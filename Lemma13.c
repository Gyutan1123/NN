#include "nn.h"
float cross_entropy_error(const float * y, int t) {
    return  -log(y[t]+1e-7); // １行で書けるはず
}