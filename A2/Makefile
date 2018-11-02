
CC = gcc
CFLAGS = -std=gnu99 -O3 -fopenmp -Wall

all: assignment2 assignment_martino assignment_sequential assignment_horizontal_rectangles assignment_vertical_rectangles

assignment2: assignment2.c utility.h algorithm.c
	$(CC) $(CFLAGS) $< -o $@

assignment_martino: assignment2_martino.c utility.h algorithm_Martino.c
	$(CC) $(CFLAGS) $< -o $@

assignment_sequential: assignment2_sequential.c utility.h algorithm_sequential.c
	$(CC) $(CFLAGS) $< -o $@

assignment_horizontal_rectangles: assignment2_horizontal_rectangles.c utility.h algorithm_horizontal_rectangles.c
	$(CC) $(CFLAGS) $< -o $@

assignment_vertical_rectangles: assignment2_vertical_rectangles.c utility.h algorithm_vertical_rectangles.c
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f assignment2 assignment_martino assignment_sequential
