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
#define CACHELINESIZE 64
#define CACHELINENUMBER 512
// 64 Bytes * 512 = 32KB


void simulate(double *input, double *output, int threads, int length, int iterations)
{
    // input is a row-major ordered matrix
    double *temp;
    omp_set_num_threads(threads);
    // Parallelize this!!
    int doubles_per_block = CACHELINESIZE/8;
    int j = 0;
    // Parallelize this!!
    for(int n=0; n < iterations; n++)
    {
        // dividing into vertical blocks whose width fit in a 64B line cache
        int blocks = length / doubles_per_block;

        // initialize block_width with N
        int block_width = doubles_per_block;

        // looping over blocks
        for(int b=0; b<blocks; b++){
            // correcting the last block_width
            if(b==n-1)
                block_width += length % doubles_per_block;

            // looping over rows
            for(int i=1; i<n; i++)
            {
                // looping over columns
                for(int h=1; h<block_width; h++)
                {
                    j = b*doubles_per_block + h;
                    if ( ((i == length/2-1) || (i== length/2))
                        && ((j == length/2-1) || (j == length/2)) )
                        continue;

                    OUTPUT(i,j) = (INPUT(i-1,j-1) + INPUT(i-1,j) + INPUT(i-1,j+1) +
                                   INPUT(i,j-1)   + INPUT(i,j)   + INPUT(i,j+1)   +
                                   INPUT(i+1,j-1) + INPUT(i+1,j) + INPUT(i+1,j+1) )/9;
                }
            }
        }

        temp = input;
        input = output;
        output = temp;
    }
}
