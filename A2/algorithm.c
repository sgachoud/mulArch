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
    const int l = 16;
    const int block_width = l;
    const int block_heigth = l;
    int j = 0;
    int i = 0;

    // checking whether we are lucky and we can exactly split in blocks
    const int blocks = length/l;
    const int extra_l = length%l;
    int max_k, max_h;
    int _h, _k;
    printf("blocks = %d\nextra_l = %d\n", blocks, extra_l);
    for(int n=0; n < iterations; n++)
    {
        // looping over blocks
        for(int b1=0; b1<blocks; b1++){
            for(int b2=0; b2<blocks; b2++){

                // fixing dimensions of the block
                _h = 0;
                _k = 0;
                max_h = l;
                max_k = l;

                if(b1==blocks-1)
                    max_h = block_width + extra_l - 1;

                if(!b1)
                    _h = 1;

                if(b2==blocks-1)
                    max_k = block_heigth + extra_l - 1;

                if(!b2)
                    _k = 1;

                // looping over rows.
                // k identifies the k-th row of the block (b1,b2)
                for(int k=_k; k<max_k; k++)
                {
                    // looping over columns
                    // h identifies the h-th columns of the block (b1,b2)
                    for(int h=_h; h<max_h; h++)
                    {
                        // (i,j) identifies the corresponding index inside the whole matrix
                        i = b2*l + k;
                        j = b1*l + h;
                        //printf("(b1,b2) = (%d,%d)\n", b1, b2);
                        //printf("(k,h) = (%d,%d)\n", k, h);
                        //printf("(i,j) = (%d,%d)\n", i, j);
                        // implementation of the algorithm
                        if ( ((i == length/2-1) || (i== length/2))
                            && ((j == length/2-1) || (j == length/2)) )
                            continue;
                        OUTPUT(i,j) = (INPUT(i-1,j-1) + INPUT(i-1,j) + INPUT(i-1,j+1) +
                                       INPUT(i,j-1)   + INPUT(i,j)   + INPUT(i,j+1)   +
                                       INPUT(i+1,j-1) + INPUT(i+1,j) + INPUT(i+1,j+1) )/9;
                    }
                }
            }
        }

        temp = input;
        input = output;
        output = temp;
    }
}
