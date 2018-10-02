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
  unsigned int in = 0;
  const double l = b-a;

  omp_set_num_threads(num_threads);
  // parallel code
  #pragma omp parallel
  {
    long double sum = 0.;         // to solve FALSE SHARING we declare a local counter     
    rand_gen gen = init_rand();  // by having modified init_rand
                                           // we assure to have different random
                                           // generators

    int max_iter = samples/num_threads;
	if(omp_get_thread_num() == 0)
		max_iter += samples%num_threads;

    for(int i=0; i < max_iter; i++){
      double x = next_rand(gen);
      sum += l * f(a + x * l);
    }

    #pragma omp atomic
    in += sum;

    free_rand(gen);
  }

  // serial code
  integral = in / (double)samples;
  return integral;
}
