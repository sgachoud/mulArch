
CC = gcc
CFLAGS = -std=gnu99 -O3 -fopenmp -Wall

all: assignment2

assignment2: assignment2.c utility.h algorithm.c
	$(CC) $(CFLAGS) $< -o $@

assignment_martino: assignment2.c utility.h algorithm_Martino.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f assignment2 assignment_martino
