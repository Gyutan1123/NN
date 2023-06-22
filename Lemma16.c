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
    for (int i = 0; i < 10; i++) y[i] = 0.0f;
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
    for (int i = 0; i < 10; i++) y[i] = 0.0f;
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
        float x = (float)rand() / RAND_MAX * 2 - 1;
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

void load(const char *filename, int m, int n, float *A, float *b) {
    FILE *read;
    read = fopen(filename, "rb");
    if (!read) {
        printf("Cannot open %c.\n", *filename);
    } else {
        fread(A, sizeof(float), m * n, read);
        fread(b, sizeof(float), m, read);
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
    float *A1 = malloc(sizeof(float) * 50 * 784);
    float *A2 = malloc(sizeof(float) * 100 * 50);
    float *A3 = malloc(sizeof(float) * 10 * 100);
    float *b1 = malloc(sizeof(float) * 50);
    float *b2 = malloc(sizeof(float) * 100);
    float *b3 = malloc(sizeof(float) * 10);
    load("fc1.dat", 50, 784, A1, b1);
    load("fc2.dat", 100, 50, A2, b2);
    load("fc3.dat", 10, 100, A3, b3);
    int correct = 0;
    float e = 0;
    for (int i = 0; i < test_count; i++) {
        float *y = malloc(sizeof(float) * 10);
        init(10, 0, y);
        if (inference6(A1, b1, A2, b2, A3, b3, test_x + i * width * height,
                       y) == test_y[i]) {
            correct++;
            e += cross_entropy_error(y, test_y[i]);
        }

        /* if (inference6(A1_784_50_100_10, b1_784_50_100_10, A2_784_50_100_10,
                       b2_784_50_100_10, A3_784_50_100_10, b3_784_50_100_10,
                       test_x + i * width * height, y) == test_y[i]) {
            correct++;
            e += cross_entropy_error(y, test_y[i]);
        } */
        free(y);
    }
    printf("正解率%f%%\n", correct * 100.0 / test_count);
    printf("損失関数%f\n", e / test_count);
    return 0;
}