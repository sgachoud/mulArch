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
#define CACHELINESIZE 64
#define CACHELINENUMBER 512
// 64 Bytes * 512 = 32KB


void simulate(double *input, double *output, int threads, int length, int iterations)
{
    // INPUT IS A ROW-MAJOR ORDERED MATRIX (implicit in line 12)
    double *temp;
    omp_set_num_threads(threads);
    // Parallelize this!!
    int doubles_per_block = CACHELINESIZE/8; // equal to 8, I did the computation only for
                                             // readability of the code
    int j = 0;
    // Parallelize this!!
    for(int n=0; n < iterations; n++)
    {
        // dividing into vertical blocks whose heigth fit in a 64B line cache
        int blocks = length / doubles_per_block;

        // initialize block_heigth with doubles_per_block
        int block_heigth = doubles_per_block;

        // looping over blocks
        for(int b=0; b<blocks; b++){
            // correcting the last block_heigth
            if(b==blocks-1)
                block_heigth += length % doubles_per_block - 1;
                // the -1 at the end is useful to correctly implement boundary
                // conditions and not modify the last column of the last block

            // looping over columns. The first row is left untouched, also the last one

            // PROBLEM: NESTING THIS LOOP IN THE BLOCK CREATES A LOT OF OVERHEAD
            // SINCE I GOES FROM 1 TO length A NUMBER OF TIMES CORRESPONDING TO
            // THE NUMBER OF BLOCKS IN THE MATRIX

            for(int i=1; i<length-1; i++)
            {
                // looping over rows
                // h identifies the h-th columns of the block b
                for(int h=0; h<block_heigth; h++)
                {
                    // j identifies the corresponding index inside the whole matrix
                    
                    // PROBLEM: THIS IS AN OVERHEAD OF 20% OF FLOPS!
                    j = b*doubles_per_block + h;

                    // implementation of the algorithm
                    if ( ((i == length/2-1) || (i== length/2))
                        && ((j == length/2-1) || (j == length/2)) )
                        continue;
                    OUTPUT(j,i) = (INPUT(j-1,i-1) + INPUT(j-1,i) + INPUT(j-1,i+1) +
                                   INPUT(j,i-1)   + INPUT(j,i)   + INPUT(j,i+1)   +
                                   INPUT(j+1,i-1) + INPUT(j+1,i) + INPUT(j+1,i+1) )/9;
                }
            }
        }

        temp = input;
        input = output;
        output = temp;
    }
}
