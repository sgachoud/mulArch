/*
============================================================================
Filename    : integral.c
Author      : Sébastien Gachoud / Martino Milani
SCIPER      : 250083 / 286204
============================================================================
*/

#include <stdio.h>
#include <stdlib.h>
#include "utility.h"
#include "function.c"

double integrate (int num_threads, int samples, int a, int b, double (*f)(double));

int main (int argc, const char *argv[]) {

    int num_threads, num_samples, a, b;
    double integral;

    if (argc != 5) {
		printf("Invalid input! Usage: ./integral <num_threads> <num_samples> <a> <b>\n");
		return 1;
	} else {
        num_threads = atoi(argv[1]);
        num_samples = atoi(argv[2]);
        a = atoi(argv[3]);
        b = atoi(argv[4]);
	}

    set_clock();

    /* You can use your self-defined funtions by replacing identity_f. */
    integral = integrate (num_threads, num_samples, a, b, square_f);

    printf("- Using %d threads: integral on [%d,%d] = %.15g computed in %.4gs.\n", num_threads, a, b, integral, elapsed_time());

    return 0;
}


double integrate (int num_threads, int samples, int a, int b, double (*f)(double)) {
    double integral;

    double sum = 0;
    int l = b-a;

    rand_gen gens[num_threads];
    double sums[num_threads];

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
    	double random = next_rand(gens[omp_get_thread_num()]);

    	double area = l * f(a + random * l);

    	sums[omp_get_thread_num()] += area;
    }

    for (int i = 0; i < num_threads; ++i)
    {
        free_rand(gens[i]);
        sum += sums[i];
    }
    integral = sum / (double)samples;

    return integral;
}
