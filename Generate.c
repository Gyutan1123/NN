#include "nn.h"
#define pi 3.14159265358979323846264338327950288 /* 円周率*/
void init(int n, float x, float *o) {
    for (int i = 0; i < n; i++) {
        o[i] = x;
    }
}

void scaling(int m, int n, const float *x, float *o,float scale_i , float scale_j) {
    int ci = m / 2;
    int cj = n / 2;
   
    for (int i = 0; i < m;i++){
        for (int j = 0; j < n;j++){
            int i_new = (i - ci+0.5) * scale_i + ci;
            int j_new = (j - cj+0.5) * scale_j + cj;
            if ((i_new >= 0) && (i_new < m) && (j_new >= 0) && (j_new <n)){
                o[i_new * n + j_new] = x[i * n + j];
            }
        }
    }
}

void shift(int m, int n, const float *x, float *o, int shift_i,int shift_j) {
    
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if ((i - shift_i >= 0) && (i - shift_i < m) && (j - shift_j >= 0) &&
                (j - shift_j < n)) {
                o[i * n + j] = x[(i - shift_i) * n + (j - shift_j)];
            }
        }
    }
}

void rotation(int m, int n, const float *x, float *o, float theta){
    theta *= pi / 180;
    int ci = m / 2;
    int cj = n / 2;
   
    for (int i = 0; i < m;i++){
        for (int j = 0; j < n;j++){
            int i_new = (i-ci+0.5)*cos(theta) - (j-cj+0.5)*sin(theta) + ci;
            int j_new = (i-ci+0.5)*sin(theta) + (j-cj+0.5)*cos(theta) + cj;
            if ((i_new >= 0) && (i_new < m) && (j_new >= 0) && (j_new <n)){
                o[i_new * n + j_new] = x[i * n + j];
            }
        }
    }
}

void generate(int m, int n, const float *x, float *o){
    float shift_i = (float)rand() / RAND_MAX * 4 -2;
    float shift_j = (float)rand() / RAND_MAX * 4 -2;
    float scale_i = (float)rand() / RAND_MAX * 0.1 + 0.7;
    float scale_j = (float)rand() / RAND_MAX * 0.1 + 0.7;
    float theta_xx = (float)rand() / RAND_MAX * 20 - 10;
    float theta_xy = (float)rand() / RAND_MAX * 20 - 10;
    float theta_yx = (float)rand() / RAND_MAX * 20 - 10;
    float theta_yy = (float)rand() / RAND_MAX * 20 - 10;
    theta_xx *= pi / 180;
    theta_xy *= pi / 180;
    theta_yx *= pi / 180;
    theta_yy *= pi / 180;
    init(m * n, 0, o);
    int ci = m / 2;
    int cj = n / 2;
    for (int i = 0; i < m;i++){
        for (int j = 0; j < n;j++){
            int i_new = ((i-ci+0.5)*cos(theta_xx) - (j-cj+0.5)*sin(theta_xy))*scale_i+shift_i + ci;
            int j_new = ((i-ci+0.5)*sin(theta_yx) + (j-cj+0.5)*cos(theta_yy))*scale_j+shift_j + cj;
            if ((i_new >= 0) && (i_new < m) && (j_new >= 0) && (j_new <n)){
                o[i_new * n + j_new] = x[i * n + j];
            }
        }
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
    float *y = malloc(sizeof(float) * 784);
    for (int i = 0; i < 10; i++){
        generate(28, 28, train_x+784, y);
        save_mnist_bmp(y, "Image/train_00001random%d.bmp", i);
    } 
    return 0;
}