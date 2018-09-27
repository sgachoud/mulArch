/*
============================================================================
Filename    : pi.c
Author      : Sébastien Gachoud /
SCIPER		: 250083 /
============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include "utility.h"

double calculate_pi (int num_threads, int samples);

int main (int argc, const char *argv[]) {

    int num_threads, num_samples;
    double pi;

    if (argc != 3) {
		printf("Invalid input! Usage: ./pi <num_threads> <num_samples> \n");
		return 1;
	} else {
        num_threads = atoi(argv[1]);
        num_samples = atoi(argv[2]);
	}

    set_clock();
    pi = calculate_pi (num_threads, num_samples);

    printf("- Using %d threads: pi = %.15g computed in %.4gs.\n", num_threads, pi, elapsed_time());

    return 0;
}


double calculate_pi (int num_threads, int samples) {
    double pi;
    unsigned int in = 0;
    rand_gen gen = init_rand();

    #pragma omp parallel for num_threads (num_threads)
    for(int i = 0; i < samples; i++)
    {
    	double x; double y;

    	#pragma omp critical
    	{
    		x = next_rand(gen);
    		y = next_rand(gen);
    	}

    	if(x*x + y*y <= 1)
    	{
    		#pragma omp atomic
    		in++;
    	}
    }
    free_rand(gen);
    pi = (in << 2) / (double)samples;
    return pi;
}
