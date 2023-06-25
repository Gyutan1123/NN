#include "nn.h"

#include "MT.h"
#define pi 3.14159265358979323846264338327950288 /* 円周率*/

/* 動的メモリ確保された配列 n 個 をまとめてfreeする */
void free_all(int n, ...) {
    va_list ap;
    va_start(ap, n);
    for (int i = 0; i < n; i++) {
        free(va_arg(ap, float *));
    }
    va_end(ap);
}

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

    free_all(5, dEdx_soft, dEdx_relu, dEdx_fc, ReLU_in, FC_in);
}

float cross_entropy_error(const float *y, int t) { return -log(y[t] + 1e-7); }

void add(int n, const float *x, float *o) {
    for (int i = 0; i < n; i++) {
        o[i] += x[i];
    }
}

void add_ave(int size, int n, const float *x, float *o) {
    for (int i = 0; i < size; i++) {
        o[i] += (float)1 / n * x[i];
    }
}

void scale_and_add(int n, float scale, float *x, float *y) {
    for (int i = 0; i < n; i++) {
        y[i] += scale * x[i];
    }
}

void scale(int n, float x, float *o) {
    for (int i = 0; i < n; i++) {
        o[i] *= x;
    }
}

void sqrt_and_div(int n, float *x, float *y, float *o) {
    for (int i = 0; i < n; i++) {
        o[i] = y[i] / sqrt(x[i] + 1e-7);
    }
}

void init(int n, float x, float *o) {
    for (int i = 0; i < n; i++) {
        o[i] = x;
    }
}

float rand01() {
    return (float)genrand_real1();
    /* return rand() / RAND_MAX * 2 - 1; */
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
    free_all(2, y1, y2);

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

    free_all(13, y1, y2, FC1_in, FC2_in, FC3_in, ReLU1_in, ReLU2_in, dEdx_soft,
             dEdx_fc1, dEdx_fc2, dEdx_fc3, dEdx_relu1, dEdx_relu2);
}

void save(const char *filename, int m, int n, const float *A, const float *b) {
    FILE *write;
    write = fopen(filename, "wb");
    fwrite(A, sizeof(float), m * n, write);
    fwrite(b, sizeof(float), m, write);
}

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

void gaussian_rand_init(int n, float *o) {
    for (int i = 0; i < n; i++) {
        float u1 = (float)genrand_real1();
        float u2 = (float)genrand_real1();
        /* float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX; */
        o[i] =
            sqrt((float)2 / n) * (float)sqrt(-2 * log(u1)) * cos(2 * pi * u2);
    }
}

void generate(int m, int n, const float *x, float *o) {
    float shift_i = rand01() * 4 - 2;
    float shift_j = rand01() * 4 - 2;
    float scale_i = rand01() * 0.2 + 0.9;
    float scale_j = rand01() * 0.2 + 0.9;
    float theta_xx = rand01() * 20 - 10;
    float theta_xy = rand01() * 20 - 10;
    float theta_yx = rand01() * 20 - 10;
    float theta_yy = rand01() * 20 - 10;
    theta_xx *= pi / 180;
    theta_xy *= pi / 180;
    theta_yx *= pi / 180;
    theta_yy *= pi / 180;
    init(m * n, 0, o);
    int ci = m / 2;
    int cj = n / 2;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int i_new = ((i - ci + 0.5) * cos(theta_xx) -
                         (j - cj + 0.5) * sin(theta_xy)) *
                            scale_i +
                        shift_i + ci;
            int j_new = ((i - ci + 0.5) * sin(theta_yx) +
                         (j - cj + 0.5) * cos(theta_yy)) *
                            scale_j +
                        shift_j + cj;
            if ((i_new >= 0) && (i_new < m) && (j_new >= 0) && (j_new < n)) {
                o[i_new * n + j_new] = x[i * n + j];
            }
        }
    }
}

/* サイズ n の配列で表されるパラメーター w を　w <- w -eta*dEdw + alpha v
 * で更新する */
void momentum_update(int n, float *w, float eta, float alpha, float *dEdw,
                     float *v) {
    scale(n, -eta, dEdw);
    scale(n, alpha, v);
    add(n, dEdw, v);
    add(n, v, w);
}
/* momentum SGD を一回分行う つまり
1.訓練データをn個ずつ取り出し、それぞれに対しA1 ~ b3 の 勾配を計算し、平均をとる
2.その勾配と前回の更新量によってA1~b3を更新する
3. 1,2を訓練データの個数 / n 回繰り返す
*/

void momentum_SGD(float *train_x, unsigned char *train_y, int train_count,
                  int n, int *index, float *A1, float *b1, float *A2, float *b2,
                  float *A3, float *b3, float *v_A1, float *v_b1, float *v_A2,
                  float *v_b2, float *v_A3, float *v_b3, float eta,
                  float alpha) {
    init(784 * 50, 0, v_A1);
    init(50 * 100, 0, v_A2);
    init(100 * 10, 0, v_A3);
    init(50, 0, v_b1);
    init(100, 0, v_b2);
    init(10, 0, v_b3);

    for (int i = 0; i < train_count / n; i++) {
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

        for (int j = 0; j < n; j++) {
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
            float *train_x_new = malloc(sizeof(float) * 784);
            generate(28, 28, train_x + 784 * index[i * n + j], train_x_new);
            backward6(A1, b1, A2, b2, A3, b3, train_x_new,
                      train_y[index[i * n + j]], dEdA1, dEdb1, dEdA2, dEdb2,
                      dEdA3, dEdb3);
            add_ave(50 * 784, n, dEdA1, dEdA1_ave);
            add_ave(50 * 100, n, dEdA2, dEdA2_ave);
            add_ave(100 * 10, n, dEdA3, dEdA3_ave);
            add_ave(50, n, dEdb1, dEdb1_ave);
            add_ave(100, n, dEdb2, dEdb2_ave);
            add_ave(10, n, dEdb3, dEdb3_ave);
            free_all(7, dEdA1, dEdA2, dEdA3, dEdb1, dEdb2, dEdb3, train_x_new);
        }
        momentum_update(784 * 50, A1, eta, alpha, dEdA1_ave, v_A1);
        momentum_update(50 * 100, A2, eta, alpha, dEdA2_ave, v_A2);
        momentum_update(100 * 10, A3, eta, alpha, dEdA3_ave, v_A3);
        momentum_update(50, b1, eta, alpha, dEdb1_ave, v_b1);
        momentum_update(100, b2, eta, alpha, dEdb2_ave, v_b2);
        momentum_update(10, b3, eta, alpha, dEdb3_ave, v_b3);
        free_all(6, dEdA1_ave, dEdA2_ave, dEdA3_ave, dEdb1_ave, dEdb2_ave,
                 dEdb3_ave);
    }
}

void test(int epoch, float *A1, float *b1, float *A2, float *b2, float *A3,
          float *b3, int test_count, float *test_x, unsigned char *test_y) {
    int correct = 0;
    float e = 0;
    for (int i = 0; i < test_count; i++) {
        float *y = malloc(sizeof(float) * 10);
        init(10, 0, y);
        if (inference6(A1, b1, A2, b2, A3, b3, test_x + i * 784, y) ==
            test_y[i]) {
            correct++;
        }
        e += cross_entropy_error(y, test_y[i]);
        free(y);
    }
    printf("Accuracy = %f%% , Loss Avg = %f ",correct*100.0/test_count,e/test_count);
    
}

/* 配列で表された画像データを平行移動させる */
void shift(int m, int n, const float *x, float *o, int shift_i, int shift_j) {
    init(m * n, 0, o);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if ((i - shift_i >= 0) && (i - shift_i < m) && (j - shift_j >= 0) &&
                (j - shift_j < n)) {
                o[i * n + j] = x[(i - shift_i) * n + (j - shift_j)];
            }
        }
    }
}

/* 画像データを拡大・縮小する */
void scaling(int m, int n, const float *x, float *o, float scale_i,
             float scale_j) {
    int ci = m / 2;
    int cj = n / 2;

    init(m * n, 0, o);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int i_new = (i - ci) * scale_i + ci;
            int j_new = (j - cj) * scale_j + cj;
            if ((i_new >= 0) && (i_new < m) && (j_new >= 0) && (j_new < n)) {
                o[i_new * n + j_new] = x[i * n + j];
            }
        }
    }
}

/* 画像データの回転*/
void rotation(int m, int n, const float *x, float *o, float theta) {
    theta *= pi / 180;
    int ci = m / 2;
    int cj = n / 2;
    init(m * n, 0, o);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int i_new = (i - ci) * cos(theta) - (j - cj) * sin(theta) + ci;
            int j_new = (i - ci) * sin(theta) + (j - cj) * cos(theta) + cj;
            if ((i_new >= 0) && (i_new < m) && (j_new >= 0) && (j_new < n)) {
                o[i_new * n + j_new] = x[i * n + j];
            }
        }
    }
}

/* xのアダマール積を計算し、y に計算結果記録*/
void Hadamard(int n, const float *x, float *y) {
    for (int i = 0; i < n; i++) {
        y[i] = x[i] * x[i];
    }
}

void Adam_update(int n, float *v, float *h, float *w, float *dEdw, float eta) {
    float beta1 = 0.9;
    float beta2 = 0.999;
    scale(n, beta1, v);
    scale_and_add(n, (1 - beta1), dEdw, v);
    scale(n, beta2, h);
    Hadamard(n, dEdw, dEdw);
    scale_and_add(n, (1 - beta2), dEdw, h);
    float *dw = malloc(sizeof(float) * n);
    sqrt_and_div(n, h, v, dw);
    scale(n, -eta * sqrt(1 - beta2) / (1 - beta1), dw);
    add(n, dw, w);
    free(dw);
}

void Adam(float *train_x, unsigned char *train_y, int train_count, int n,
          int *index, float *A1, float *b1, float *A2, float *b2, float *A3,
          float *b3, float *v_A1, float *v_b1, float *v_A2, float *v_b2,
          float *v_A3, float *v_b3, float eta, float *h_A1, float *h_b1,
          float *h_A2, float *h_b2, float *h_A3, float *h_b3) {
    for (int i = 0; i < train_count / n; i++) {
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
        for (int j = 0; j < n; j++) {
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
            float *train_x_new = malloc(sizeof(float) * 784);
            generate(28, 28, train_x + 784 * index[i * n + j], train_x_new);
            backward6(A1, b1, A2, b2, A3, b3, train_x_new,
                      train_y[index[i * n + j]], dEdA1, dEdb1, dEdA2, dEdb2,
                      dEdA3, dEdb3);
            add_ave(50 * 784, n, dEdA1, dEdA1_ave);
            add_ave(50 * 100, n, dEdA2, dEdA2_ave);
            add_ave(100 * 10, n, dEdA3, dEdA3_ave);
            add_ave(50, n, dEdb1, dEdb1_ave);
            add_ave(100, n, dEdb2, dEdb2_ave);
            add_ave(10, n, dEdb3, dEdb3_ave);
            free_all(7, dEdA1, dEdA2, dEdA3, dEdb1, dEdb2, dEdb3, train_x_new);
        }
        Adam_update(50 * 784, v_A1, h_A1, A1, dEdA1_ave, eta);
        Adam_update(50 * 100, v_A2, h_A2, A2, dEdA2_ave, eta);
        Adam_update(10 * 100, v_A3, h_A3, A3, dEdA2_ave, eta);
        Adam_update(50, v_b1, h_b1, b1, dEdb1_ave, eta);
        Adam_update(100, v_b2, h_b2, b2, dEdb2_ave, eta);
        Adam_update(10, v_b3, h_b3, b3, dEdb3_ave, eta);
        free_all(6, dEdA1_ave, dEdA2_ave, dEdA3_ave, dEdb1_ave, dEdb2_ave,
                 dEdb3_ave);
    }
}

int main(int argc, char *argv[]) {
    init_genrand(time(NULL));
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
    float eta = 0.001;
    float alpha = 0.9;
    float *A1 = malloc(sizeof(float) * 784 * 50);
    float *A2 = malloc(sizeof(float) * 50 * 100);
    float *A3 = malloc(sizeof(float) * 100 * 10);
    float *b1 = malloc(sizeof(float) * 50);
    float *b2 = malloc(sizeof(float) * 100);
    float *b3 = malloc(sizeof(float) * 10);
    load(argv[1], 50, 784, A1, b1);
    load(argv[2], 100, 50, A2, b2);
    load(argv[3], 10, 100, A3, b3);
    /* gaussian_rand_init(784 * 50, A1);
    gaussian_rand_init(50 * 100, A2);
    gaussian_rand_init(100 * 10, A3);
    gaussian_rand_init(50, b1);
    gaussian_rand_init(100, b2);
    gaussian_rand_init(10, b3); */
    int *index = malloc(sizeof(int) * train_count);
    for (int i = 0; i < train_count; i++) {
        index[i] = i;
    }
    /* 変化量を表す変数*/
    float *v_A1 = malloc(sizeof(float) * 50 * 784);
    float *v_A2 = malloc(sizeof(float) * 50 * 100);
    float *v_A3 = malloc(sizeof(float) * 100 * 10);
    float *v_b1 = malloc(sizeof(float) * 50);
    float *v_b2 = malloc(sizeof(float) * 100);
    float *v_b3 = malloc(sizeof(float) * 10);
    init(784 * 50, 0, v_A1);
    init(50 * 100, 0, v_A2);
    init(100 * 10, 0, v_A3);
    init(50, 0, v_b1);
    init(100, 0, v_b2);
    init(10, 0, v_b3);
    float *h_A1 = malloc(sizeof(float) * 50 * 784);
    float *h_A2 = malloc(sizeof(float) * 50 * 100);
    float *h_A3 = malloc(sizeof(float) * 100 * 10);
    float *h_b1 = malloc(sizeof(float) * 50);
    float *h_b2 = malloc(sizeof(float) * 100);
    float *h_b3 = malloc(sizeof(float) * 10);
    init(784 * 50, 0, h_A1);
    init(50 * 100, 0, h_A2);
    init(100 * 10, 0, h_A3);
    init(50, 0, h_b1);
    init(100, 0, h_b2);
    init(10, 0, h_b3);
    for (int i = 0; i < epoch; i++) {
        shuffle(train_count, index);
        momentum_SGD(train_x, train_y, train_count, n, index, A1, b1, A2, b2,
                     A3, b3, v_A1, v_b1, v_A2, v_b2, v_A3, v_b3, eta, alpha);
        /* Adam(train_x, train_y, train_count, n, index, A1, b1, A2, b2, A3, b3,
             v_A1, v_b1, v_A2, v_b2, v_A3, v_b3, eta, h_A1, h_b1, h_A2, h_b2,
             h_A3, h_b3); */
        printf("Epoch %3d/%d\n", i + 1, epoch);
        printf("Train : ");
        test(i + 1, A1, b1, A2, b2, A3, b3, train_count, train_x, train_y);
        printf("Test : ");
        test(i + 1, A1, b1, A2, b2, A3, b3, test_count, test_x, test_y);
        printf("\n");
    }
    free_all(6, v_A1, v_A2, v_A3, v_b1, v_b2, v_b3);
    save("fc1.dat", 50, 784, A1, b1);
    save("fc2.dat", 100, 50, A2, b2);
    save("fc3.dat", 10, 100, A3, b3);

    return 0;
}
