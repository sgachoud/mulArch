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
    int l = 16;
    int block_width = l;
    int block_heigth = l;
    int j = 0;
    int i = 0;

    // checking whether we are lucky and we can exactly split in blocks
    int blocks = length/l;
    int extra_l = length%l;
    int max_k, max_h;
    int h, k;
    for(int n=0; n < iterations; n++)
    {


        // looping over blocks
        for(int b1=0; b1<blocks; b1++){
            for(int b2=0; b2<blocks; b2++){

                // fixing dimensions of the block
                h = 0;
                k = 0;
                max_h = l;
                max_k = l;
                block_width = l;
                block_heigth = l;
                if(b1==blocks-1){
                    block_width += extra_l;
                    max_h = block_width-1;
                }
                if(!b1)
                    h = 0;

                if(b2==blocks-1){
                    block_heigth += extra_l;
                    max_k = block_heigth-1;
                }
                if(!b2)
                    k = 0;

                // looping over rows.
                // k identifies the k-th row of the block (b1,b2)
                for(; k<max_k; k++)
                {
                    // looping over columns
                    // h identifies the h-th columns of the block (b1,b2)
                    for(; h<max_h; h++)
                    {
                        // (i,j) identifies the corresponding index inside the whole matrix
                        i = b2*l + k;
                        j = b1*l + h;
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
        }

        temp = input;
        input = output;
        output = temp;
    }
}
