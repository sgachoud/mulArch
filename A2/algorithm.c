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
    #pragma omp parallel
    {

    // initializing private variables
    int whoami = omp_get_thread_num();
    int _i = 0;
    int _j = 0;
    int how_many_i = 0;
    int how_many_j = 0;
    int l = 0;
    // easy:
    if (threads == 16){
        l = length/4;
        // taking care of the i
        if(whoami<4){
            _i = 1;
            how_many_i = l-1;}
        else if(whoami<8){
            _i = l;
            how_many_i = l;}
        else if(whoami<12){
            _i = 2*l;
            how_many_i = l;}
        else{
            _i = 3*l;
            how_many_i = l-1 + length % threads;}

        // taking care of the j
        if(whoami == 0 || whoami == 4 || whoami == 8 || whoami == 12){
            _j = 1;
            how_many_j = l-1;}
        else if(whoami == 1 || whoami == 5 || whoami == 9 || whoami == 13){
            _j = l;
            how_many_j = l;}
        else if(whoami == 2 || whoami == 6 || whoami == 10 || whoami == 14){
            _j = 2*l;
            how_many_j = l;}
        else if(whoami == 3 || whoami == 7 || whoami == 11 || whoami == 15){
            _j = 3*l;
            how_many_j = l-1 + length % threads;}
    }
    fprintf(stderr, "whoami = %d, _i=%d, how_many_i=%d, _j=%d, how_many_j=%d\n", whoami, _i, how_many_i, _j, how_many_j);
    // starting the temporal iterations
    for(int n=0; n < iterations; n++){
        for(int i=_i; i<_i+how_many_i; i++){
            for(int j=_j; j<_j+how_many_j; j++){
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
