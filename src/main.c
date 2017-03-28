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
    
    for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
	A[i][j] = 2;
	B[i][j] = 1;
      }
    }
    
    //Start counters
    PAPI_start_counters(PAPI_events, 2);

    /* vectorization */
    //for (i = 0; i < N; i += MU ) {
    //for (j = 0; j < N; j += NU) {
    //	  printf("i = %d, j = %d\n", i, j);
    i = 4;
    j = 0;
            for (k = 0; k < N; k++) {
                for (m = i; m < i + MU; m+=4) {
		  //Load C by row
		  __m128 rZ = _mm_loadu_ps((float *)(C) + m*N + k + j);
		  printf("C offset = %d\n", m*N + k +j);

                    for (n = j; n < j + NU; n++) {
		      //print X
		      //printf("Aptr: %p Bptr: %p\n", A + m*N + k*N, B + k*N + n);
		      __m128 rX = _mm_load1_ps((float *)(A) + m*N + k*N);
		      //printf("A offset = %d\n", m*N + k*N);
		      float *val = (float*) &rX;
		      //printf("A = %f \n", val[0]);
		      
		       //print Y
		       __m128 rY = _mm_loadu_ps((float *)(B) + N + n);
		       
		       printf("A offset = %d, B offset = %d\n", (n - j)*N + i, N + n);
		      

		       val = (float*) &rY;
		      //printf("B = (%f, %f, %f, %f) \n", val[0], val[1], val[2], val[3]);
		     
		      rY = _mm_mul_ps(rX, rY);
		      rZ = _mm_add_ps(rZ, rY);
                    }

		    printf("\n\n");
		    //Store C by row
		    _mm_storeu_ps((float *)(C) + m*N +(k + j), rZ);
		}
            }
	    //       }
	    //}

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

    //free(A);
    //free(B);
    //free(C);
}

void test() {
    int N = 8;
    int NU = 4;
    int MU = NU;
    int i, j, k, m, n;
    for (i = 0; i < N; i += NU) {
        for (j = 0; j < N; j += MU) {
            for (k = 0; k < N; k++) {
                for (m = i; m < (i + MU); m += 4) {
                    // not sure about the below address
                    c_addr = ((float * )C + m * NUMBER_OF_COLUMNS + j;

                    for (n = j; n < (j + NU); n++) {
                        a_addr = ((float *)A + m * NUMBER_OF_COLUMNS + k);
                        b_addr = ((float *)B + k * NUMBER_OF_COLUMNS + n);
                    }
                }
                /* What we are trying to accomplish */

                // A[i...MU-1, k]
                // A[i][k]
                // A[i + 1][k]
                // A[i + 2][k]
                // A[i + ...][k]
                // A[i + MU-1][k]
                // x = *((float *)A + i * NUMBER_OF_COLUMNS + k);
                // y = *((float *)B + k * NUMBER_OF_COLUMNS + i);
            }
        }
    }

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
        vector_intrinsics(8, PAPI_events);
    }

    return 0;
}
