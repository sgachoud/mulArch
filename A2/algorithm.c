/*
============================================================================
Filename    : algorithm.c
Author      : Your names go here
SCIPER      : Your SCIPER numbers

============================================================================
*/
#include <math.h>

#define INPUT(I,J) input[(I)*length+(J)]
#define OUTPUT(I,J) output[(I)*length+(J)]

void simulate(double *input, double *output, int threads, int length, int iterations)
{
    double *temp;
    omp_set_num_threads(threads);

    // Parallelize this!!
    #pragma omp parallel num_threads(threads)
    {
    	printf("%u\n",omp_get_thread_num() );
	    for(int n=0; n < iterations; n++)
	    {
	    	#pragma omp for
	        for(int i=1; i<length-1; i++)
	        {
	            for(int j=1; j<length-1; j++)
	            {
	                    if ( ((i == length/2-1) || (i== length/2))
	                        && ((j == length/2-1) || (j == length/2)) )
	                    {}
	            		else
	            		{
							OUTPUT(i,j) = (INPUT(i-1,j-1) + INPUT(i-1,j) + INPUT(i-1,j+1) +
	                                   INPUT(i,j-1)   + INPUT(i,j)   + INPUT(i,j+1)   +
	                                   INPUT(i+1,j-1) + INPUT(i+1,j) + INPUT(i+1,j+1) )/9;
	            		}
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

void simulate_sequencial(double *input, double *output, int threads, int length, int iterations)
{
    double *temp;
    
    // Parallelize this!!
    for(int n=0; n < iterations; n++)
    {
        for(int i=1; i<length-1; i++)
        {
            for(int j=1; j<length-1; j++)
            {
                    if ( ((i == length/2-1) || (i== length/2))
                        && ((j == length/2-1) || (j == length/2)) )
                        continue;

                    OUTPUT(i,j) = (INPUT(i-1,j-1) + INPUT(i-1,j) + INPUT(i-1,j+1) +
                                   INPUT(i,j-1)   + INPUT(i,j)   + INPUT(i,j+1)   +
                                   INPUT(i+1,j-1) + INPUT(i+1,j) + INPUT(i+1,j+1) )/9;
            }
        }

        temp = input;
        input = output;
        output = temp;
    }
}
