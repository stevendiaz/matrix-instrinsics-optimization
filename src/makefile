CC=gcc
CFLAGS := -L$(TACC_PAPI_LIB) -lpapi -I$(TACC_PAPI_INC) -Wall -Werror -O0



end_to_end: multiply.c no_papi_multiply.c
	$(CC) -L$(TACC_PAPI_LIB) -lpapi -I$(TACC_PAPI_INC) -Wall -Werror -O3 -fp-model precise multiply.c -o multiply.o
	$(CC) -Wall -Werror -O3 -fp-model precise no_papi_multiply.c -o no_papi.o -lrt
	sbatch slurm_matrix_multiply
	sbatch slurm_no_papi

main: main.c
	$(CC) $(CFLAGS) main.c -o main.o

run:
	./main.o

clean:
	rm *.o