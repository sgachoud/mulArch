/*
============================================================================
Filename    : algorithm.c
Author      : Martino Milani / Sébastien Gachoud
SCIPER      : 286204 / 250083

============================================================================
*/
#include <math.h>
#include <omp.h>

#define INPUT(I,J) input[(I)*length+(J)]
#define OUTPUT(I,J) output[(I)*length+(J)]

void simulate(double *input, double *output, int threads, int length, int iterations)
{
    // INPUT IS A ROW-MAJOR ORDERED MATRIX (implicit in line 12)
    double *temp;
    omp_set_num_threads(threads);
    // Parallelize this!!

    //L1 cache size in B 
    const int L1_CACHE_SIZE = 32000;
    const long int DOUBLE_SIZE = sizeof(double);


    // checking whether we are lucky and we can exactly split in blocks
    #pragma omp parallel
    {

    // initializing private variables
    int whoami = omp_get_thread_num();
    int _i = length/threads*whoami;
    int how_many_i = length/threads;
    if (!whoami){
        _i = 1;
        how_many_i -= 1;
    }
    else if (whoami == threads-1)
        how_many_i += length%threads-1;

    // starting the temporal iterations
    for(int n=0; n < iterations; n++){
        for(int i=_i; i<_i+how_many_i; i++){
            for(int j=1; j<length-1; j++){
                // implementation of the algorithm
                if ( ((i == length/2-1) || (i== length/2))
                    && ((j == length/2-1) || (j == length/2)) )
                    continue;
                OUTPUT(i,j) = (INPUT(i-1,j-1) + INPUT(i-1,j) + INPUT(i-1,j+1) +
                               INPUT(i,j-1)   + INPUT(i,j)   + INPUT(i,j+1)   +
                               INPUT(i+1,j-1) + INPUT(i+1,j) + INPUT(i+1,j+1) )/9;
            }
        }
        // exchanging pointers
        #pragma omp barrier
        // we need a barrier at the end of the exchange of pointers too!!
        // for this reason a #pragma omp master (no implicit barrier does not work)
        #pragma omp single
        {
		temp = input;
        input = output;
        output = temp;
        }
    }
    }
}
