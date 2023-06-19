#include "nn.h"
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
int inference3(const float *A, const float *b, const float *x) {
    float *y = malloc(sizeof(float) * 10);
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
        float x = (float)(rand() % 3 - 1);
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

int inference6(const float *A1, const float *b1, const float *A2,
               const float *b2, const float *A3, const float *b3,
               const float *x, float *y) {
    float *y1 = malloc(sizeof(float) * 50);
    init(50, 0, y1);
    fc(50, 784, x, A1, b1, y1);
    relu(50, y1, y1);
    float *y2 = malloc(sizeof(float) * 100);
    init(100, 0, y2);
    fc(100, 50, y1, A2, b2, y2);

    relu(100, y2, y2);
    fc(10, 100, y2, A3, b3, y);

    softmax(10, y, y);
    float ymax = y[0];
    int ymax_index = 0;
    for (int i = 0; i < 10; i++) {
        if (y[i] > ymax) {
            ymax = y[i];
            ymax_index = i;
        }
    }
    free(y1);
    free(y2);
    return ymax_index;
};

void equalize(int n, const float *x, float *y) {
    for (int i = 0; i < n; i++) {
        y[i] = x[i];
    }
}

void backward6(const float *A1, const float *b1, const float *A2,
               const float *b2, const float *A3, const float *b3,
               const float *x, unsigned char t, float *dEdA1, float *dEdb1,
               float *dEdA2, float *dEdb2, float *dEdA3, float *dEdb3) {
    float *y1 = malloc(sizeof(float) * 50);
    init(50, 0, y1);
    float *FC1_in = malloc(sizeof(float) * 784);
    equalize(784, x, FC1_in);
    fc(50, 784, x, A1, b1, y1);
    float *ReLU1_in = malloc(sizeof(float) * 50);
    equalize(50, y1, ReLU1_in);
    relu(50, y1, y1);
    float *y2 = malloc(sizeof(float) * 100);
    init(100, 0, y2);
    float *FC2_in = malloc(sizeof(float) * 50);
    equalize(50, y1, FC2_in);
    fc(100, 50, y1, A2, b2, y2);
    float *ReLU2_in = malloc(sizeof(float) * 100);
    equalize(100, y2, ReLU2_in);
    relu(100, y2, y2);
    float *y3 = malloc(sizeof(float) * 10);
    init(10, 0, y3);
    float *FC3_in = malloc(sizeof(float) * 100);
    equalize(100, y2, FC3_in);
    fc(10, 100, y2, A3, b3, y3);
    softmax(10, y3, y3);

    float *dEdx_soft = malloc(sizeof(float) * 10);
    float *dEdx_fc3 = malloc(sizeof(float) * 100);
    float *dEdx_relu2 = malloc(sizeof(float) * 100);
    float *dEdx_fc2 = malloc(sizeof(float) * 50);
    float *dEdx_relu1 = malloc(sizeof(float) * 50);
    float *dEdx_fc1 = malloc(sizeof(float) * 784);
    softmaxwithloss_bwd(10, y3, t, dEdx_soft);
    fc_bwd(10, 100, FC3_in, dEdx_soft, A3, dEdA3, dEdb3, dEdx_fc3);
    relu_bwd(100, ReLU2_in, dEdx_fc3, dEdx_relu2);
    fc_bwd(100, 50, FC2_in, dEdx_relu2, A2, dEdA2, dEdb2, dEdx_fc2);
    relu_bwd(50, ReLU1_in, dEdx_fc2, dEdx_relu1);
    fc_bwd(50, 784, FC1_in, dEdx_relu1, A1, dEdA1, dEdb1, dEdx_fc1);

    free(y1);
    free(y2);
    free(y3);
    free(FC1_in);
    free(FC2_in);
    free(FC3_in);
    free(ReLU1_in);
    free(ReLU2_in);
    free(dEdx_soft);
    free(dEdx_fc1);
    free(dEdx_fc2);
    free(dEdx_fc3);
    free(dEdx_relu1);
    free(dEdx_relu2);
}

void save(const char *filename, int m, int n, const float *A, const float *b) {
    FILE *write;
    write = fopen(filename,"w");
    fwrite(A, sizeof(float), m * n, write);
    fwrite(b, sizeof(float), m, write);
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
    float *A1 = malloc(sizeof(float) * 784 * 50);
    float *A2 = malloc(sizeof(float) * 50 * 100);
    float *A3 = malloc(sizeof(float) * 100 * 10);
    float *b1 = malloc(sizeof(float) * 50);
    float *b2 = malloc(sizeof(float) * 100);
    float *b3 = malloc(sizeof(float) * 10);
    rand_init(784 * 50, A1);
    rand_init(50 * 100, A2);
    rand_init(100 * 10, A3);
    rand_init(50, b1);
    rand_init(100, b2);
    rand_init(10, b3);
    int *index = malloc(sizeof(int) * train_count);
    for (int i = 0; i < train_count; i++) {
        index[i] = i;
    }
    for (int i = 0; i < epoch; i++) {
        shuffle(train_count, index);
        for (int j = 0; j < train_count / n; j++) {
            float *dEdA1_ave = malloc(sizeof(float) * 50 * 784);
            float *dEdA2_ave = malloc(sizeof(float) * 50 * 100);
            float *dEdA3_ave = malloc(sizeof(float) * 100 * 10);
            float *dEdb1_ave = malloc(sizeof(float) * 50);
            float *dEdb2_ave = malloc(sizeof(float) * 100);
            float *dEdb3_ave = malloc(sizeof(float) * 10);
            init(784 * 50, 0, dEdA1_ave);
            init(50 * 100, 0, dEdA2_ave);
            init(100 * 10, 0, dEdA3_ave);
            init(50, 0, dEdb1_ave);
            init(100, 0, dEdb2_ave);
            init(10, 0, dEdb3_ave);
            for (int k = 0; k < n; k++) {
                float *dEdA1 = malloc(sizeof(float) * 50 * 784);
                float *dEdA2 = malloc(sizeof(float) * 50 * 100);
                float *dEdA3 = malloc(sizeof(float) * 100 * 10);
                float *dEdb1 = malloc(sizeof(float) * 50);
                float *dEdb2 = malloc(sizeof(float) * 100);
                float *dEdb3 = malloc(sizeof(float) * 10);
                init(50 * 784, 0, dEdA1);
                init(100 * 50, 0, dEdA2);
                init(100 * 10, 0, dEdA3);
                init(50, 0, dEdb1);
                init(100, 0, dEdb2);
                init(10, 0, dEdb3);
                backward6(A1, b1, A2, b2, A3, b3,
                          train_x + 784 * index[j * n + k],
                          train_y[index[j * n + k]], dEdA1, dEdb1, dEdA2, dEdb2,
                          dEdA3, dEdb3);
                add(50 * 784, dEdA1, dEdA1_ave);
                add(50 * 100, dEdA2, dEdA2_ave);
                add(100 * 10, dEdA3, dEdA3_ave);
                add(50, dEdb1, dEdb1_ave);
                add(100, dEdb2, dEdb2_ave);
                add(10, dEdb3, dEdb3_ave);
                free(dEdA1);
                free(dEdA2);
                free(dEdA3);
                free(dEdb1);
                free(dEdb2);
                free(dEdb3);
            }
            scale(784 * 50, (float)-eta / n, dEdA1_ave);
            scale(50 * 100, (float)-eta / n, dEdA2_ave);
            scale(10 * 100, (float)-eta / n, dEdA3_ave);
            scale(50, (float)-eta / n, dEdb1_ave);
            scale(100, (float)-eta / n, dEdb2_ave);
            scale(10, (float)-eta / n, dEdb3_ave);
            add(784 * 50, dEdA1_ave, A1);
            add(50 * 100, dEdA2_ave, A2);
            add(10 * 100, dEdA3_ave, A3);
            add(50, dEdb1_ave, b1);
            add(100, dEdb2_ave, b2);
            add(10, dEdb3_ave, b3);
            free(dEdA1_ave);
            free(dEdA2_ave);
            free(dEdA3_ave);
            free(dEdb1_ave);
            free(dEdb2_ave);
            free(dEdb3_ave);
        }
        int correct = 0;
        float e = 0;
        for (int i = 0; i < test_count; i++) {
            float *y = malloc(sizeof(float) * 10);
            init(10, 0, y);
            if (inference6(A1, b1, A2, b2, A3, b3, test_x + i * width * height,
                           y) == test_y[i]) {
                correct++;
            }
            e += cross_entropy_error(y, test_y[i]);
            free(y);
        }
        printf("epoch%2d 正解率 : %f%%\n", i + 1, correct * 100.0 / test_count);
        printf("        損失関数 : %f\n", e / test_count);
    }
    save("fc1.dat", 50, 784, A1, b1);
    save("fc2.dat", 100, 50, A2, b2);
    save("fc3.dat", 10, 100, A3, b3);
    return 0;
    
}