#include <stdlib.h>
#include <stdio.h>
#include <x86intrin.h>


void a(int N)
{
    float (*A)[N] = malloc(sizeof(float[N][N]));
    float (*B)[N] = malloc(sizeof(float[N][N]));
    float (*C)[N] = malloc(sizeof(float[N][N]));

    int i,j,k;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                C[i][j] = C[i][j] + A[i][k] * B[k][j];
            }
        }
    }

    free(A);
    free(B);
    free(C);
}


int main(int args, char *argv[]) 
{
    float (*A)[4] = malloc(sizeof(double[4][4]));
    A[0][0] = 1.0;
    A[2][2] = 2.0;
    __m64 x;
    printf("first: %f \n", A[0][0]);
    printf("second: %f \n", A[2][2]);
    free(A);
    return 0;
}
