/*
============================================================================
Filename    : algorithm.c
Author      : Your names go here
SCIPER      : Your SCIPER numbers

============================================================================
*/
#include <math.h>
#include <omp.h>

#define INPUT(I,J) input[(I)*length+(J)]
#define OUTPUT(I,J) output[(I)*length+(J)]

// big assumption: cache lines of 64Bytes (that's the dimension on my Mac, you can change it
#define L1SIZE 32
// 32 KBytes


void simulate(double *input, double *output, int threads, int length, int iterations)
{
    // INPUT IS A ROW-MAJOR ORDERED MATRIX (implicit in line 12)
    double *temp;
    omp_set_num_threads(threads);
    // Parallelize this!!


    // checking whether we are lucky and we can exactly split in blocks
    #pragma omp parallel shared(input, output, temp)
    {
    int whoami = omp_get_thread_num();
    int _i = length/threads*whoami;
    int how_many_i = length/threads;
    if (!whoami){
        _i = 1;
        how_many_i -= 1;
    }
    if (whoami == threads-1)
        how_many_i += length%threads-1;
    for(int n=0; n < iterations; n++)
    {
        // looping over rows.
        for(int i=_i; i<_i+how_many_i; i++)
        {
            // looping over columns
            for(int j=1; j<length-1; j++)
            {
                // implementation of the algorithm
                if ( ((i == length/2-1) || (i== length/2))
                    && ((j == length/2-1) || (j == length/2)) )
                    continue;
                OUTPUT(i,j) = (INPUT(i-1,j-1) + INPUT(i-1,j) + INPUT(i-1,j+1) +
                               INPUT(i,j-1)   + INPUT(i,j)   + INPUT(i,j+1)   +
                               INPUT(i+1,j-1) + INPUT(i+1,j) + INPUT(i+1,j+1) )/9;
            }
        }
        #pragma omp barrier
        #pragma omp single
        {
        temp = input;
        input = output;
        output = temp;
        }
    }
    }
}
