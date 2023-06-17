#include "nn.h"

void print(int m, int n, const float * x){
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            printf("%.4f ", x[i*n + j]);
        }
        printf("\n");
    }
}

void fc(int m, int n, const float * x, const float * A, const float * b, float *y){
    for (int k = 0; k < m; k++ ){
        for (int j = 0; j < n; j++ ){
            y[k] += A[k*n + j] * x[j];
        }
        y[k] += b[k];
    }
}

int main(){
    float * train_x = NULL;
    unsigned char * train_y = NULL;
    int train_count = -1;
    float *test_x = NULL;
    unsigned char *test_y = NULL;
    int test_count = -1;
    int width = -1;
    int height = -1;
    load_mnist(&train_x, &train_y, &train_count, &test_x, &test_y, &test_count, &width, &height);
    float y[10];
    fc(10, 784, train_x, A_784x10, b_784x10, y);
    
    print(1, 10, y);
    return 0;
}