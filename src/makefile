CC=gcc
CFLAGS := -L$(TACC_PAPI_LIB) -lpapi -I$(TACC_PAPI_INC) -Wall -O0

main: main.c
	$(CC) $(CFLAGS) main.c -o main.o

clean:
	rm *.o
