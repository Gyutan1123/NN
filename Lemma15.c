#include "nnmodified.h"

void print(int m, int n, const float *x) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.4f ", x[i * n + j]);
        }
        printf("\n");
    }
}

void fc(int m, int n, const float *x, const float *A, const float *b,
        float *y) {
    for (int k = 0; k < m; k++) {
        for (int j = 0; j < n; j++) {
            y[k] += A[k * n + j] * x[j];
        }
        y[k] += b[k];
    }
}

void relu(int n, const float *x, float *y) {
    for (int k = 0; k < n; k++) {
        if (x[k] > 0) {
            y[k] = x[k];
        } else {
            y[k] = 0;
        }
    }
}

void softmax(int n, const float *x, float *y) {
    float xmax = x[0];
    for (int k = 0; k < n; k++) {
        if (x[k] > xmax) {
            xmax = x[k];
        }
    }
    float sum = 0;
    for (int k = 0; k < n; k++) {
        sum += exp(x[k] - xmax);
    }
    for (int k = 0; k < n; k++) {
        y[k] = exp(x[k] - xmax) / sum;
    }
}
int inference3(const float *A, const float *b, const float *x,float *y) {
    
    for (int i = 0; i < 10; i++)
        y[i] = 0.0f;
    fc(10, 784, x, A, b, y);
    relu(10, y, y);
    softmax(10, y, y);
    float ymax = y[0];
    int ymax_index = 0;
    for (int i = 0; i < 10; i++) {
        if (y[i] > ymax) {
            ymax = y[i];
            ymax_index = i;
        }
    }
    return ymax_index;
}

void softmaxwithloss_bwd(int n, const float *y, unsigned char t, float *dEdx) {
    for (int i = 0; i < n; i++) {
        dEdx[i] = y[i];
        if (i == t) {
            dEdx[i] -= 1;
        }
    }
}

void relu_bwd(int n, const float *x, const float *dEdy, float *dEdx) {
    for (int i = 0; i < n; i++) {
        if (x[i] > 0) {
            dEdx[i] = dEdy[i];
        } else {
            dEdx[i] = 0;
        }
    }
}

void fc_bwd(int m, int n, const float *x, const float *dEdy, const float *A,
            float *dEdA, float *dEdb, float *dEdx) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            dEdA[n * i + j] = dEdy[i] * x[j];
        }
        dEdb[i] = dEdy[i];
    }
    for (int i = 0; i < n; i++) {
        dEdx[i] = 0;
        for (int j = 0; j < m; j++) {
            dEdx[i] += A[i + n * j] * dEdy[j];
        }
    }
}

void backward3(const float *A, const float *b, const float *x, unsigned char t,
               float *y, float *dEdA, float *dEdb) {
    for (int i = 0; i < 10; i++)
        y[i] = 0.0f;
    float *FC_in = malloc(sizeof(float) * 784);
    for (int i = 0; i < 784; i++) {
        FC_in[i] = x[i];
    }
    fc(10, 784, x, A, b, y);
    float *ReLU_in = malloc(sizeof(float) * 10);
    for (int i = 0; i < 10; i++) {
        ReLU_in[i] = y[i];
    }
    relu(10, y, y);
    softmax(10, y, y);
    float *dEdx_soft = malloc(sizeof(float) * 10);
    float *dEdx_relu = malloc(sizeof(float) * 10);
    float *dEdx_fc = malloc(sizeof(float) * 784);
    softmaxwithloss_bwd(10, y, t, dEdx_soft);
    relu_bwd(10, ReLU_in, dEdx_soft, dEdx_relu);
    fc_bwd(10, 784, FC_in, dEdx_relu, A, dEdA, dEdb, dEdx_fc);
    free(dEdx_soft);
    free(dEdx_relu);
    free(FC_in);
    free(ReLU_in);
    free(dEdx_fc);
}

float cross_entropy_error(const float *y, int t) { return -log(y[t] + 1e-7); }

void add(int n, const float *x, float *o) {
    for (int i = 0; i < n; i++) {
        o[i] += x[i];
    }
}

void scale(int n, float x, float *o) {
    for (int i = 0; i < n; i++) {
        o[i] *= x;
    }
}

void init(int n, float x, float *o) {
    for (int i = 0; i < n; i++) {
        o[i] = x;
    }
}

void rand_init(int n, float *o) {
    for (int i = 0; i < n; i++) {
        float x = (float)rand()/RAND_MAX*2-1;
        o[i] = x;
    }
}

void shuffle(int n, int *x) {
    for (int i = 0; i < n; i++) {
        int j = rand() % n;
        int tmp = x[i];
        x[i] = x[j];
        x[j] = tmp;
    }
}

int main() {
    srand(time(NULL));
    float *train_x = NULL;
    unsigned char *train_y = NULL;
    int train_count = -1;
    float *test_x = NULL;
    unsigned char *test_y = NULL;
    int test_count = -1;
    int width = -1;
    int height = -1;
    load_mnist(&train_x, &train_y, &train_count, &test_x, &test_y, &test_count,
               &width, &height);
    int epoch = 10;
    int n = 100;
    float eta = 0.1;
    float *A = malloc(sizeof(float) * 10 * 784);
    float *b = malloc(sizeof(float) * 10);
    rand_init(784 * 10, A);
    rand_init(10, b);
    int *index = malloc(sizeof(int) * train_count);
    for (int i = 0; i < train_count; i++) {
        index[i] = i;
    }
    for (int i = 0; i < epoch; i++) {
        shuffle(train_count, index);
        for (int j = 0; j < train_count / n; j++) {
            float *dEdA_ave = malloc(sizeof(float) * 10 * 784);
            float *dEdb_ave = malloc(sizeof(float) * 10);
            init(10 * 784, 0, dEdA_ave);
            init(10, 0, dEdb_ave);
            for (int k = 0; k < n; k++) {
                float *y = malloc(sizeof(float) * 10);
                float *dEdA = malloc(sizeof(float) * 10 * 784);
                float *dEdb = malloc(sizeof(float) * 10);
                backward3(A, b, train_x + 784 * index[j * n + k],
                          train_y[index[j * n + k]], y, dEdA, dEdb);
                add(784 * 10, dEdA, dEdA_ave);
                add(10, dEdb, dEdb_ave);
                free(dEdA);
                free(dEdb);
                free(y);
            }
            scale(784 * 10, (float)-eta / n, dEdA_ave);
            scale(10, (float)-eta / n, dEdb_ave);
            add(784 * 10, dEdA_ave, A);
            add(10, dEdb_ave, b);
            free(dEdA_ave);
            free(dEdb_ave);
        }
        int correct = 0;
        float e = 0;
        for (int i = 0; i < test_count; i++) {
            float *y = malloc(sizeof(float) * 10);
            if (inference3(A, b, test_x + i * width * height,y) ==
                test_y[i]) {
                correct++;
            }
            e += cross_entropy_error(y,test_y[i]);
            free(y);
        }

        printf("epoch%2d 正解率 : %f%%\n", i+1,correct * 100.0 / test_count);
        printf("        損失関数 : %f\n", e / test_count);
    }

    return 0;
}