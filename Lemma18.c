#include "nnmodified.h"

void save(const char *filename, int m, int n, const float *A ,const float *b){
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
