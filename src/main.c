#include <stdlib.h>
#include <stdio.h>
#include <x86intrin.h>
#include <string.h>


void a(int N)
{
    float (*A)[N] = malloc(sizeof(float[N][N]));
    float (*B)[N] = malloc(sizeof(float[N][N]));
    float (*C)[N] = malloc(sizeof(float[N][N]));

    int NU, MU = 64/sizeof(float);
    printf("nu: %d\n", NU);
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
    /* Data & parameter initialization */
    int i, j, k, m, n;
    int NU = 4;
    int MU = 4;
    int N = 8;
    float (*A)[N] = malloc(sizeof(float[N][N]));
    float (*B)[N] = malloc(sizeof(float[N][N]));
    float (*C)[N] = malloc(sizeof(float[N][N]));
    for(i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = 1;
            B[i][j] = 2;
        }
    }

    /* Register tiling, no vectorization */
    for (i = 0; i < N; i += MU ) {
        for (j = 0; j < N; j += NU) {
            for (k = 0; k < N; k++) {
                // Register tiling
                // No vectorization
                for (m = i; m < i + MU; m++) {
                   for(n = j; n < j + NU; n++) {
                       __m128 rX = _mm_load_ss(&A[m][k]);
                       __m128 rY = _mm_load_ss(&B[k][n]);
                       __m128 rZ = _mm_mul_ss(rX, rY);
                       rZ = _mm_add_ss(rZ, _mm_load_ss(&C[m][n]));
                       _mm_store_ss(&C[m][n], rZ);
                    }
                }
            }
        }
    }
    /* Checking output */
    for(i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            printf("%f ", C[i][j]);
        }
        printf("\n");
    }

    free(A);
    free(B);
    free(C);
    return 0;
}
