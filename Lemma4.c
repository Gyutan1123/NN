#include "nnmodified.h"

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


void relu(int n, const float *x, float*y){
    for (int k = 0; k < n; k++){
        if (x[k]>0){
            y[k] = x[k];
        } else{
            y[k] = 0;
        }
    }
}

void softmax(int n, const float *x, float *y){
    float xmax = x[0];
    for (int k = 0; k < n; k++){
        if (x[k] > xmax){
            xmax = x[k];
        }
    }
    float sum = 0;
    for (int k = 0; k < n; k++){
        sum += exp(x[k] - xmax);
    }
    for (int k = 0; k < n; k++){
        y[k] = exp(x[k] - xmax) / sum;
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
    load_mnist(&train_x, &train_y, &train_count, 
    &test_x, &test_y, &test_count, &width, &height);
    float *y = malloc(sizeof(float) * 10);
    for (int i = 0; i < 10;i++)  y[i] = 0.0f;
    fc(10, 784, train_x, A_784x10, b_784x10, y);
    relu(10, y, y);
    softmax(10, y, y);
    print(1, 10, y);
    return 0;
}