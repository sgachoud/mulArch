/*
============================================================================
Filename    : pi.c
Author      : Sébastien Gachoud / Martino Milania
SCIPER		: 250083 / 000000
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

    rand_gen gens[num_threads];
    int sums[num_threads];

    #pragma omp parallel for num_threads (num_threads)
    for (int i = 0; i < num_threads; ++i)
    {
        gens[i] = init_rand();  //use of i instead of omp_get_thread_num() because 
                                //we don't care whitch gen is used by a specific 
                                //thread as long they all use a diffrent one
        sums[i] = 0;
    }

    #pragma omp parallel for num_threads (num_threads)
    for(int i = 0; i < samples; i++)
    {
        rand_gen gen = gens[omp_get_thread_num()];
    	double x = next_rand(gen);
    	double y = next_rand(gen);

    	if(x*x + y*y <= 1)
    	{
    		sums[omp_get_thread_num()]++;
    	}
    }

    for (int i = 0; i < num_threads; ++i)
    {
        free_rand(gens[i]);
        in += sums[i];
    }

    pi = (in << 2) / (double)samples;
    return pi;
}
