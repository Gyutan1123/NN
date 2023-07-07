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
        y[k] = exp(x[k] - xmax) / (sum+1e-7);
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
        float x = (float)rand() /RAND_MAX *2-1;
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

void copy(int n, const float *x, float *y) {
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
    copy(784, x, FC1_in);
    fc(50, 784, x, A1, b1, y1);
    float *ReLU1_in = malloc(sizeof(float) * 50);
    copy(50, y1, ReLU1_in);
    relu(50, y1, y1);
    float *y2 = malloc(sizeof(float) * 100);
    init(100, 0, y2);
    float *FC2_in = malloc(sizeof(float) * 50);
    copy(50, y1, FC2_in);
    fc(100, 50, y1, A2, b2, y2);
    float *ReLU2_in = malloc(sizeof(float) * 100);
    copy(100, y2, ReLU2_in);
    relu(100, y2, y2);
    float *y3 = malloc(sizeof(float) * 10);
    init(10, 0, y3);
    float *FC3_in = malloc(sizeof(float) * 100);
    copy(100, y2, FC3_in);
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
    write = fopen(filename, "wb");
    fwrite(A, sizeof(float), m * n, write);
    fwrite(b, sizeof(float), m, write);
}

void load (const char * filename, int m, int n, float *A, float *b){
    FILE *read;
    read = fopen(filename, "rb");
    if (!read){
        printf("Cannot open %s.\n", *filename);
    } else{
        fread(A, sizeof(float), m * n, read);
        fread(b, sizeof(float), m, read);
    }
}

int main(int argc, char *argv[]) {
    float *A1 = malloc(sizeof(float) * 784 * 50);
    float *b1 = malloc(sizeof(float) * 50);
    float *A2 = malloc(sizeof(float) * 50 * 100);
    float *b2 = malloc(sizeof(float) * 100);
    float *A3 = malloc(sizeof(float) * 100 * 10);
    float *b3 = malloc(sizeof(float) * 10);
    float *x = load_mnist_bmp(argv[4]);
    float *y = malloc(sizeof(float) * 10);
    init(10, 0, y);
    load(argv[1], 50, 784, A1, b1);
    load(argv[2], 100, 50, A2, b2);
    load(argv[3], 10, 100, A3, b3);
    printf("この数字は%dです。\n", inference6(A1, b1, A2, b2, A3, b3, x, y));
    return 0;
}