#include <stdlib.h>
#include <stdio.h>
#include <x86intrin.h>
#include <string.h>
#include <papi.h>

void printCounters(int q, int size, long long* counters){
  printf("Part %c: matrix size = %d\n", 'a' + q, size);
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
    printCounters(0, N, counters);

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
		      register float rX = A[m][k];
		      register float rY = B[k][n];
		      register float rZ = C[i][j];
		      rY = rX * rY;
		      rZ += rY;
		      C[i][j] = rZ;
                    }
                }
            }
        }
    }

    //Stop counters
    PAPI_stop_counters(counters, 2);
    printCounters(1, N, counters);

    free(A);
    free(B);
    free(C);
}

/* Part c */
void vector_intrinsics(int N, int* PAPI_events){
    /* Initialize PAPI counter */
   long long counters[2];
    memset(counters, 0, 2*sizeof(long long));

    int i, j, k, n;
    int NU = 4;
    int MU = 4;
    float (*A)[N] = malloc(sizeof(float[N][N]));
    float (*B)[N] = malloc(sizeof(float[N][N]));
    float (*C)[N] = malloc(sizeof(float[N][N]));
    memset(A, 0, N*N*sizeof(float));
    memset(B, 0, N*N*sizeof(float));
    memset(C, 0, N*N*sizeof(float));

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = 2;
            B[i][j] = 1;
        }
    }

    //Start counters
    PAPI_start_counters(PAPI_events, 2);

    /* vectorization */
    for (i = 0; i < N; i += MU ) {
        for (j = 0; j < N; j += NU) {
            for (k = 0; k < 4; k++) {
                //Load C by row
                float *c_addr = ((float *) C + (k + i) * N + j);
                __m128 rZ = _mm_loadu_ps(c_addr);

                for (n = j; n < j + NU; n++) {
                    float *a_addr = ((float *) A + n * N + k + i);
                    float *b_addr = ((float *) B + (k + i) * N + j);

                    __m128 rX = _mm_load1_ps(a_addr);
                    __m128 rY = _mm_loadu_ps(b_addr);
                    rY = _mm_mul_ps(rX, rY);
                    rZ = _mm_add_ps(rZ, rY);
                }
                //Store C by row
                _mm_storeu_ps(c_addr, rZ);
            }
        }
    }

    //Stop counters
    PAPI_stop_counters(counters, 2);
    printCounters(2, N, counters);

    free(A);
    free(B);
    free(C);
}

/* Part d */
void cache_blocking(int N, int* PAPI_events){
    /* Initialize PAPI counter */
    long long counters[2];
    memset(counters, 0, 2*sizeof(long long));

    int i, j, bi, bj, bk, n;
  
    int MU = 4;
    int NU = 4;
    int NB = 64;
    float (*A)[N] = malloc(sizeof(float[N][N]));
    float (*B)[N] = malloc(sizeof(float[N][N]));
    float (*C)[N] = malloc(sizeof(float[N][N]));
    memset(A, 0, N*N*sizeof(float));
    memset(B, 0, N*N*sizeof(float));
    memset(C, 0, N*N*sizeof(float));

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = 2;
            B[i][j] = 1;
        }
    }

    //Start counters
    PAPI_start_counters(PAPI_events, 2);

    /* Cache blocking loops */
    for (i = 0; i < N; i += NB ) {
      for (j = 0; j < N; j += NB) {

        /* Register tiling loops */
        for (bi = i; bi < (i + NB); bi += MU) {
            for(bj = j; bj < (j + NB); bj += NU){
                //printf("REG corner[%d][%d]\n", bi,bj);
                for (bk = 0; bk < MU; bk++) {
                    //Load C by row
                    float *c_addr = ((float *) C + (bk + bi) * N + bj);
                    __m128 rZ = _mm_loadu_ps(c_addr);

                    for (n = bj; n < bj + NU; n++) {
                        float *a_addr = ((float *) A + n * N + bk + bi);
                        float *b_addr = ((float *) B + (bk + bi) * N + bj);

                        __m128 rX = _mm_load1_ps(a_addr);
                        __m128 rY = _mm_loadu_ps(b_addr);
                        rY = _mm_mul_ps(rX, rY);
                        rZ = _mm_add_ps(rZ, rY);
                    }
                    //Store C by row
                    _mm_storeu_ps(c_addr, rZ);
                }
            }
        }
      }
    }


    //Stop counters
    PAPI_stop_counters(counters, 2);
    printCounters(3, N, counters);

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

    //Part a
    /*
    mmm(16, PAPI_events);
    mmm(64, PAPI_events);
    mmm(128, PAPI_events);
    mmm(160, PAPI_events);
    mmm(188, PAPI_events);
    mmm(200, PAPI_events);
    */

    //Part b
    
    register_tiling(16, PAPI_events);
    register_tiling(64, PAPI_events);
    register_tiling(128, PAPI_events);
    register_tiling(160, PAPI_events);
    register_tiling(188, PAPI_events);
    register_tiling(200, PAPI_events);
    

    //Part c
    /*
    vector_intrinsics(16, PAPI_events);
    vector_intrinsics(64, PAPI_events);
    vector_intrinsics(128, PAPI_events);
    vector_intrinsics(160, PAPI_events);
    vector_intrinsics(188, PAPI_events);
    vector_intrinsics(200, PAPI_events);
    */

    //Part d
    /* cache_blocking(16, PAPI_events);
    cache_blocking(64, PAPI_events);
    cache_blocking(128, PAPI_events);
    cache_blocking(160, PAPI_events);
    cache_blocking(184, PAPI_events);*/
    
    //cache_blocking(192, PAPI_events);

    return 0;
}
