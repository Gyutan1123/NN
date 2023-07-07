#include "nnmodified.h"

void softmaxwithloss_bwd(int n, const float * y, unsigned char t, float * dEdx){
    for (int i = 0; i < n; i++){
        dEdx[i] = y[i];
        if (i == t){
            dEdx[i] -= 1;
        }
    }
}