#include <stdlib.h>
#include <stdio.h>
#include <x86intrin.h>
#include <string.h>
#include <papi.h>

void printCounters(int q, long long* counters){
    printf("Part %c:", 'a' + q);
    printf("Number of flops: %lld\n", counters[0]);
    printf("Number of L1 data/ins cache misses: %lld\n", counters[1]);
}


/* Part a  */
void mmm(int N, int* PAPI_events){
    /* Initialize PAPI counter */
    long long counters[2];
    memset(counters, 0, 2*sizeof(long long));

    int i, j, k;
    float (*A)[N] = malloc(sizeof(float[N][N]));
    float (*B)[N] = malloc(sizeof(float[N][N]));
    float (*C)[N] = malloc(sizeof(float[N][N]));
    memset(A, 0, N*N*sizeof(float));
    memset(B, 0, N*N*sizeof(float));
    memset(C, 0, N*N*sizeof(float));

    //Start counters
    PAPI_start_counters(PAPI_events, 2);

    /* Register tiling, no vectorization */
    for (i = 0; i < N; i++ ) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < N; k++) {
                C[i][j] = C[i][j] + A[i][k] * B[k][j];
            }
        }
    }

    //Stop counters
    PAPI_stop_counters(counters, 2);
    printCounters(0, counters);

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
}



/* Part b */
void register_tiling(int N, int* PAPI_events){

    /* Initialize PAPI counter */
    long long counters[2];
    memset(counters, 0, 2*sizeof(long long));

    int i, j, k, m, n;
    int NU = 4;
    int MU = 4;
    float (*A)[N] = malloc(sizeof(float[N][N]));
    float (*B)[N] = malloc(sizeof(float[N][N]));
    float (*C)[N] = malloc(sizeof(float[N][N]));
    memset(A, 0, N*N*sizeof(float));
    memset(B, 0, N*N*sizeof(float));
    memset(C, 0, N*N*sizeof(float));


    //Start counters
    PAPI_start_counters(PAPI_events, 2);

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

    //Stop counters
    PAPI_stop_counters(counters, 2);
    printCounters(1, counters);


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
}

/* Part c */
void vector_intrinsics(int N, int* PAPI_events){
    /* Initialize PAPI counter */
    long long counters[2];
    memset(counters, 0, 2*sizeof(long long));

    int i, j, k, m, n;
    int NU = 4;
    int MU = 4;
    float (*A)[N] = malloc(sizeof(float[N][N]));
    float (*B)[N] = malloc(sizeof(float[N][N]));
    float (*C)[N] = malloc(sizeof(float[N][N]));
    memset(A, 0, N*N*sizeof(float));
    memset(B, 0, N*N*sizeof(float));
    memset(C, 0, N*N*sizeof(float));


    //Start counters
    PAPI_start_counters(PAPI_events, 2);

    /* vectorization */

    for (i = 0; i < N; i += MU ) {
        for (j = 0; j < N; j += NU) {
            for (k = 0; k < N; k++) {
                __m128 rX = _mm_load_ps(&A[m]);
                __m128 rY = _mm_load_ps(&B[k]);
                __m128 rZ = _mm_mul_ps(rX, rY);
                rZ = _mm_add_ps(rZ, _mm_load_ps(&C[m]));
                _mm_store_ps(&C[m], rZ);
            }
        }
    }

    //Stop counters
    PAPI_stop_counters(counters, 2);
    printCounters(2, counters);

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
}


int main(int args, char *argv[]) {
    /* Initalize PAPI counters */
    int PAPI_events[] = {
            PAPI_FP_OPS,
            PAPI_L1_DCM,
    };

    PAPI_library_init(PAPI_VER_CURRENT);

    /* Data & parameter initialization */
    int n;
    for(n = 4; n < 5; n += 4) {
        //mmm(n, PAPI_events);
        //register_tiling(n, PAPI_events);
        vector_intrinsics(n, PAPI_events);
    }

    return 0;
}
