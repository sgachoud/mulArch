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
	  }
    else {
        num_threads = atoi(argv[1]);  //  int atoi(const char*) converts string to int
        num_samples = atoi(argv[2]);
	  }
    set_clock();
    pi = calculate_pi (num_threads, num_samples);

    printf("- Using %d threads: pi = %.15g computed in %.4gs.\n", num_threads, pi, elapsed_time());

    return 0;
}


double calculate_pi (int num_threads, int samples) {
    double pi;
    unsigned long in = 0;
    int sums[num_threads];  // this stays on the same cahe line: causes FALSE SHARING
                            // if used inside the parallel code many times
    omp_set_num_threads(num_threads);
    printf("calculate pi");
    // parallel code
    #pragma omp parallel
    {
      int thread_num = omp_get_thread_num();
      sums[thread_num] = 0;
      rand_gen gen = init_rand((unsigned short)thread_num);  // by having modified init_rand
                                             // we assure to have different random
                                             // generators

      for(int i=0; i<samples/num_threads; i++){
        double x = next_rand(gen);
      	double y = next_rand(gen);
        if(x*x + y*y <= 1)
      		sums[thread_num]++;
      }
      free_rand(gen);
    }

    // serial code
    for (int i=0; i < num_threads; ++i)
        in += sums[i];
    pi = (in << 2) / (double)samples;
    return pi;
}
