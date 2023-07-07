#include "nnmodified.h"

void print(int m, int n, const float * x){
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            printf("%.4f ", x[i*n + j]);
        }
        printf("\n");
    }
}

int main(){
    print(1, 10, b_784x10);
    return 0;
}