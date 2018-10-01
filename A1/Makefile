
CC = gcc-8
CFLAGS = -std=gnu99 -O3 -fopenmp -Wall

all: pi integral test pi_falsesharing

pi: pi.c utility.h
	$(CC) $(CFLAGS) $< -o $@

pi_falsesharing: pi_falsesharing.c utility.h
	$(CC) $(CFLAGS) $< -o $@

integral: integral.c function.o utility.h
	$(CC) $(CFLAGS) $< -o $@

test: omp_hello.c
	$(CC) $(CFLAGS) $< -o $@
	./test

clean:
	rm -f pi integral function.o test
