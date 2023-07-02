#include "nn.h"
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
/* サイズm*nの配列xをm*n行列とみなし、そのように出力する*/
void print(int m, int n, const float *x) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.4f ", x[i * n + j]);
        }
        printf("\n");
    }
}
/*m*n行列A, n*1行列 x, m*1行列 b, y について、 y = Ax + b とする*/
void fc(int m, int n, const float *x, const float *A, const float *b,
        float *y) {
    for (int k = 0; k < m; k++) {
        for (int j = 0; j < n; j++) {
            y[k] += A[k * n + j] * x[j];
        }
        y[k] += b[k];
    }
}
/*サイズnの配列xにrelu関数を適用し、その結果をyに記録する*/
void relu(int n, const float *x, float *y) {
    for (int k = 0; k < n; k++) {
        if (x[k] > 0) {
            y[k] = x[k];
        } else {
            y[k] = 0;
        }
    }
}
/*サイズnの配列xにsoftmax関数を適用し、その結果をyに記録する*/
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

/* n*1行列である、softmax層の出力yおよび、NNの推論による出力の理想値であるtを用いて
下流への勾配dEdxを計算する */
void softmaxwithloss_bwd(int n, const float *y, unsigned char t, float *dEdx) {
    for (int i = 0; i < n; i++) {
        dEdx[i] = y[i];
        if (i == t) {
            dEdx[i] -= 1;
        }
    }
}
/* n*1行列 x を入力とする relu層において、上流からの勾配dEdy によって下流への勾配dEdxを計算する */
void relu_bwd(int n, const float *x, const float *dEdy, float *dEdx) {
    for (int i = 0; i < n; i++) {
        if (x[i] > 0) {
            dEdx[i] = dEdy[i];
        } else {
            dEdx[i] = 0;
        }
    }
}
/* m*n行列A, m*1行列b について、fc層に入力されるn*1行列x, m*1行列で表される上流の層からの勾配dEdyを用いてdEdA,dEdbを計算
また、下流への勾配dEdxを計算する */
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

/*損失関数E=-Σt_klog(y_k) を計算し、それを戻り値とするとする。0の対数をとってしまわないように、微小量1e-7を加える*/
float cross_entropy_error(const float *y, int t) { return -log(y[t] + 1e-7); }

/*サイズnの配列x,oに対し、oの各要素にxの各要素を足す*/
void add(int n, const float *x, float *o) {
    for (int i = 0; i < n; i++) {
        o[i] += x[i];
    }
}

/*サイズ size の配列x,o について、oの各要素に、xの各要素の 1/n倍を足す*/
void add_ave(int size, int n, const float *x, float *o) {
    for (int i = 0; i < size; i++) {
        o[i] += (float)1 / n * x[i];
    }
}

/* サイズ n の配列 o の各要素を x倍する */
void scale(int n, float x, float *o) {
    for (int i = 0; i < n; i++) {
        o[i] *= x;
    }
}

/* サイズnの配列oを各要素 x で初期化する */
void init(int n, float x, float *o) {
    for (int i = 0; i < n; i++) {
        o[i] = x;
    }
}

/* [0:1]の一様乱数に従うfloat型の値を返却する */
float rand01() {
    return (float)rand() / RAND_MAX ;
}

/*サイズnの配列oを各要素[0:1]の一様乱数で初期化する*/
void rand_init(int n, float *o) {
    for (int i = 0; i < n; i++) {
        float x = rand01();
        o[i] = x;
    }
}

/*0~n-1 について、x[i] を x[j] (jは0~n-1の乱数)と入れ替えることで、サイズ n の配列 x をシャッフルする*/
void shuffle(int n, int *x) {
    for (int i = 0; i < n; i++) {
        int j = rand() % n;
        int tmp = x[i];
        x[i] = x[j];
        x[j] = tmp;
    }
}

/* 各fc層のパラメータ、入力 x、出力結果ｙを表す配列を入力として、
順番に計算していきyを求める。
その後、yの要素のうち最大のものの index を求め、それを返り値とする。 */
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

/* サイズ n の配列 x, y について、yの各要素を x の各要素と等しくする */
void copy(int n, const float *x, float *y) {
    for (int i = 0; i < n; i++) {
        y[i] = x[i];
    }
}

/* 各fc層のパラメータ、NNに対する入力 x 答え t , 損失関数EのA1 ~ b3 の偏微分を表す配列を入力とし、
各層への入力をそれぞれ記録しながら、順にNNの構成にしたがって計算していく。
その後、最終的な計算結果から、逆方向に勾配を計算していき、EのA1~b3の偏微分を計算する*/
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

/* 指定したファイル名に、ポインタA,bが指しているサイズ m*n , mの配列を、m*n , m*1 行列として保存する */
void save(const char *filename, int m, int n, const float *A, const float *b) {
    FILE *write;
    write = fopen(filename, "wb");
    fwrite(A, sizeof(float), m * n, write);
    fwrite(b, sizeof(float), m, write);
}

/* 指定したバイナリファイルに保存されている、m*n行列A,m*1行列b を、ポインタA,bが示す配列に格納する
ファイルが存在しない場合はエラー文を表示する*/
void load(const char *filename, int m, int n, float *A, float *b) {
    FILE *read;
    read = fopen(filename, "rb");
    if (!read) {
        printf("Cannot open file %s.\n", filename);
    } else {
        fread(A, sizeof(float), m * n, read);
        fread(b, sizeof(float), m, read);
    }
}

/* Box-Muller法と変数変換により、標準偏差\sqrt{\frac{2}{n}} のガウス分布を実装*/
void gaussian_rand_init(int n, float *o) {
    for (int i = 0; i < n; i++) {
        float u1 = rand01();
        float u2 = rand01();
        o[i] = sqrt((float)2 / n) * (float)sqrt(-2 * log(u1)) * cos(2 * pi * u2);
    }
}

/* m*n ビットで表される画像 x を、ランダムに 平行移動、拡大縮小、回転させ、新たな画像を生成し、それを o に保存する。 */
void generate(int m, int n, const float *x, float *o) {
    float shift_i = rand01() * 4 - 2; /*上下左右に±2ビット平行移動される*/
    float shift_j = rand01() * 4 - 2;
    float scale_i = rand01() * 0.2 + 0.9; /*上下左右に 0.9 ~ 1.1 倍される*/
    float scale_j = rand01() * 0.2 + 0.9;
    float theta_xx = rand01() * 20 - 10; /* -10 ~ 20度回転 */
    float theta_xy = rand01() * 20 - 10; /*ねじれた画像も作るために、回転のパラメータを4つにしている*/
    float theta_yx = rand01() * 20 - 10;
    float theta_yy = rand01() * 20 - 10;
    theta_xx *= pi / 180; /*ラジアン*/
    theta_xy *= pi / 180;
    theta_yx *= pi / 180;
    theta_yy *= pi / 180;
    init(m * n, 0, o);
    int ci = m / 2; /*画像の中心*/
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
3. 1,2を　(訓練データの個数/n) 回繰り返す
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
        float *dEdA1_ave = malloc(sizeof(float) * 50 * 784); /*ミニバッチに含まれる各パラメータによる、*/
        float *dEdA2_ave = malloc(sizeof(float) * 50 * 100); /*勾配の平均*/
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
            /*訓練データをそのまま使うのではなく、generate関数を用いて新しくデータを作成し、それによって
            勾配を計算する*/
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

/* 各fc層のパラメータ、テストケースの個数、テストケース、答えを渡して、正答率及び損失関数の平均を計算*/
void test(float *A1, float *b1, float *A2, float *b2, float *A3,
          float *b3, int test_count, float *test_x, unsigned char *test_y) {
    int correct = 0; /*正解した個数*/
    float e = 0; /*損失関数の和*/
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

int main(int argc, char *argv[]) {
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

    int epoch = 10; /*エポック数*/
    int n = 100; /*ミニバッチサイズ*/
    float eta = 0.1; /* 学習率*/
    float alpha = 0.9; /* momentumの係数*/
    /*Ai bi はfci層のパラメータ*/
    float *A1 = malloc(sizeof(float) * 784 * 50);
    float *A2 = malloc(sizeof(float) * 50 * 100);
    float *A3 = malloc(sizeof(float) * 100 * 10);
    float *b1 = malloc(sizeof(float) * 50);
    float *b2 = malloc(sizeof(float) * 100);
    float *b3 = malloc(sizeof(float) * 10);
    /* load(argv[1], 50, 784, A1, b1);
    load(argv[2], 100, 50, A2, b2);
    load(argv[3], 10, 100, A3, b3); */
    gaussian_rand_init(784 * 50, A1);
    gaussian_rand_init(50 * 100, A2);
    gaussian_rand_init(100 * 10, A3);
    gaussian_rand_init(50, b1);
    gaussian_rand_init(100, b2);
    gaussian_rand_init(10, b3);
    int *index = malloc(sizeof(int) * train_count);
    for (int i = 0; i < train_count; i++) {
        index[i] = i;
    }
    /* 各パラメータA1~b3について、前回の変化量を表す変数*/
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
    for (int i = 0; i < epoch; i++) { /*以下をepoch回繰り返す*/
        shuffle(train_count, index); /*配列indexをシャッフル*/
        momentum_SGD(train_x, train_y, train_count, n, index, A1, b1, A2, b2, /*momentum SGDの更新式によってA1~b3を更新*/
                     A3, b3, v_A1, v_b1, v_A2, v_b2, v_A3, v_b3, eta, alpha);
        printf("Epoch %3d/%d\nTrain : ", i + 1, epoch); 
        test(A1, b1, A2, b2, A3, b3, train_count, train_x, train_y); /*訓練データに対する正答率及び損失関数*/
        printf("Test : ");
        test(A1, b1, A2, b2, A3, b3, test_count, test_x, test_y); /*テストデータに対する正答率及び損失関数*/
        printf("\n");
    }
    free_all(6, v_A1, v_A2, v_A3, v_b1, v_b2, v_b3); 
    save("fc1.dat", 50, 784, A1, b1); /*各パラメータを保存*/ 
    save("fc2.dat", 100, 50, A2, b2); 
    save("fc3.dat", 10, 100, A3, b3);
    free_all(6, A1, b1, A2, b2, A3, b3);
    return 0;
}
