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
    // serial code: initialization of variables
    double pi = 0.;
    unsigned long int in = 0;  // counter of the samples fallen inside the quarter
                               // of the circle
    omp_set_num_threads(num_threads);
    // parallel code
    #pragma omp parallel
    {
      int cont = 0;         // to avoid FALSE SHARING we declare a private counter
                            // allocated on the stack of each thread
      rand_gen gen = init_rand();  // initiate differently each random generators
      // taking care of max_iter
      unsigned long int max_iter = samples/num_threads;
      int whoami = omp_get_thread_num();
      if (!whoami)
        max_iter += samples % num_threads;
      // core code
      for(int i=0; i < max_iter; i++){
        double x = next_rand(gen);
      	double y = next_rand(gen);
        if(x*x + y*y <= 1)
      		cont++;
      }
      // incrementing the shared variable
      #pragma omp atomic
      in += cont;
      // freeing memory
      free_rand(gen);
    }
    // serial code
    pi = (in << 2) / (double)samples;
    return pi;
}
