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

int min(int a, int b){
	if(a < b){
		return a;
	}
	return b;
}

void caresOf(int from_i, int to_i, int from_j, int to_j, double *input, double *output, int length){
	for (int i = from_i; i < to_i; ++i)
	{
		for (int j = from_j; j < to_j; ++j)
		{
			if ( ((i == length/2-1) || (i== length/2))
	                    && ((j == length/2-1) || (j == length/2)) )
	                    continue;
	                OUTPUT(i,j) = (INPUT(i-1,j-1) + INPUT(i-1,j) + INPUT(i-1,j+1) +
	                               INPUT(i,j-1)   + INPUT(i,j)   + INPUT(i,j+1)   +
	                               INPUT(i+1,j-1) + INPUT(i+1,j) + INPUT(i+1,j+1) )/9;
		}
	}
}

void simulate(double *input, double *output, int threads, int length, int iterations)
{
    // INPUT IS A ROW-MAJOR ORDERED MATRIX (implicit in line 12)
    double *temp;
    omp_set_num_threads(threads);
    // Parallelize this!!

    //L1 cache size in B 
    int L1_CACHE_SIZE = 32000;
    int DOUBLE_SIZE = sizeof(double);
    int NB_DOUBLE_IN_L1 = L1_CACHE_SIZE / DOUBLE_SIZE;
    int SQUARE_SIZE = sqrt(NB_DOUBLE_IN_L1/3);

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
        for(int i=_i; i<_i+how_many_i; i += SQUARE_SIZE){
        	int imax = min(_i+how_many_i, i + SQUARE_SIZE);
        	for (int j = 1; j < length - 1; j += SQUARE_SIZE)
        	{
        		int jmax = min(length - 1, j + SQUARE_SIZE);
	            caresOf(i, imax, j, jmax, input, output, length);
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
