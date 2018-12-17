/*
============================================================================
Filename    : algorithm.c
Author      : Your name goes here
SCIPER      : Your SCIPER number
============================================================================
*/

#include <iostream>
#include <iomanip>
#include <sys/time.h>
#include <cuda_runtime.h>
using namespace std;

// CPU Baseline
void array_process(double *input, double *output, int length, int iterations)
{
    double *temp;

    for(int n=0; n<(int) iterations; n++)
    {
        for(int i=1; i<length-1; i++)
        {
            for(int j=1; j<length-1; j++)
            {
                output[(i)*(length)+(j)] = (input[(i-1)*(length)+(j-1)] +
                                            input[(i-1)*(length)+(j)]   +
                                            input[(i-1)*(length)+(j+1)] +
                                            input[(i)*(length)+(j-1)]   +
                                            input[(i)*(length)+(j)]     +
                                            input[(i)*(length)+(j+1)]   +
                                            input[(i+1)*(length)+(j-1)] +
                                            input[(i+1)*(length)+(j)]   +
                                            input[(i+1)*(length)+(j+1)] ) / 9;

            }
        }
        output[(length/2-1)*length+(length/2-1)] = 1000;
        output[(length/2)*length+(length/2-1)]   = 1000;
        output[(length/2-1)*length+(length/2)]   = 1000;
        output[(length/2)*length+(length/2)]     = 1000;

        temp = input;
        input = output;
        output = temp;
    }
}

__global__
void gpu_computation(double* input, double* output, int length);

// GPU Optimized function
void GPU_array_process(double *input, double *output, int length, int iterations)
{
    //Cuda events for calculating elapsed time
    cudaEvent_t cpy_H2D_start, cpy_H2D_end, comp_start, comp_end, cpy_D2H_start, cpy_D2H_end;
    cudaEventCreate(&cpy_H2D_start);
    cudaEventCreate(&cpy_H2D_end);
    cudaEventCreate(&cpy_D2H_start);
    cudaEventCreate(&cpy_D2H_end);
    cudaEventCreate(&comp_start);
    cudaEventCreate(&comp_end);

    /* Preprocessing goes here */

    /*----- What I did -----*/
    const long SIZE = length * length * sizeof(double);
    double* gpu_input;
    double* gpu_output;
    dim3 threadsPerBlock(32,32);
    dim3 nbBlocks(length / threadsPerBlock.x + 1, length / threadsPerBlock.y + 1);
    const long PADDED_SIZE = (nbBlocks.x+1) * threadsPerBlock.x * (nbBlocks.y+1) * threadsPerBlock.y * sizeof(double); //+1 to avoid going out of the input
    cudaSetDevice(0);
    if(cudaMalloc((void**)&gpu_input, PADDED_SIZE) != cudaSuccess){
        cerr << "Error allocating input" << endl;
    }
    if(cudaMalloc((void**)&gpu_output, PADDED_SIZE) != cudaSuccess){
        cerr << "Error allocating output" << endl;
    }
    /*----------------------*/

    cudaEventRecord(cpy_H2D_start);
    /* Copying array from host to device goes here */

    /*----- What I did -----*/
    if(cudaMemcpy(gpu_input, input, SIZE, cudaMemcpyHostToDevice) != cudaSuccess){
        cerr << "Error copying input to gpu" << endl;
    }

    if(cudaMemcpy(gpu_output, output, SIZE, cudaMemcpyHostToDevice) != cudaSuccess){
        cerr << "Error copying output to gpu" << endl;
    }
    /*----------------------*/

    cudaEventRecord(cpy_H2D_end);
    cudaEventSynchronize(cpy_H2D_end);

    //Copy array from host to device
    cudaEventRecord(comp_start);
    /* GPU calculation goes here */

    /*----- What I did -----*/
    for(int iter(0); iter < iterations; iter++){
        if(iter%2){ 
            gpu_computation <<< nbBlocks, threadsPerBlock >>> (gpu_output, gpu_input, length);
        }
        else{
            gpu_computation <<< nbBlocks, threadsPerBlock >>> (gpu_input, gpu_output, length);
        }
        cudaThreadSynchronize();
    }
    /*----------------------*/

    cudaEventRecord(comp_end);
    cudaEventSynchronize(comp_end);

    cudaEventRecord(cpy_D2H_start);
    /* Copying array from device to host goes here */

    /*----- What I did -----*/
    if(iterations%2==0)
    {
        if(cudaMemcpy(output, gpu_input, SIZE, cudaMemcpyDeviceToHost) != cudaSuccess){
            cerr << "failed to retrieve gpu_input" << endl;
        }
    }
    else{
        if(cudaMemcpy(output, gpu_output, SIZE, cudaMemcpyDeviceToHost) != cudaSuccess){
            cerr << "failed to retrieve gpu_output" << endl;
        }
    }
    /*----------------------*/

    cudaEventRecord(cpy_D2H_end);
    cudaEventSynchronize(cpy_D2H_end);

    /* Postprocessing goes here */

    /*----- What I did -----*/
    cudaFree(&gpu_input);
    cudaFree(&gpu_output);
    /*----------------------*/

    float time;
    cudaEventElapsedTime(&time, cpy_H2D_start, cpy_H2D_end);
    cout<<"Host to Device MemCpy takes "<<setprecision(4)<<time/1000<<"s"<<endl;

    cudaEventElapsedTime(&time, comp_start, comp_end);
    cout<<"Computation takes "<<setprecision(4)<<time/1000<<"s"<<endl;

    cudaEventElapsedTime(&time, cpy_D2H_start, cpy_D2H_end);
    cout<<"Device to Host MemCpy takes "<<setprecision(4)<<time/1000<<"s"<<endl;
}

/*With & instead of branches single kernel*/
//DOES NOT PRODUCE CORRECT RESULT : need to extend ones
__global__
void gpu_computation(double* input, double* output, int length){
    int x_glob = (blockIdx.x * blockDim.x) + threadIdx.x + 1;   //+1 to avoid first column
    int y_glob = (blockIdx.y * blockDim.y) + threadIdx.y + 1;   //+1 to avoid first row
    int element_id = (y_glob * length) + x_glob;
    bool isCenter = ((x_glob == length/2-1) || (x_glob == length/2)) && ((y_glob == length/2-1) || (y_glob == length/2));
    bool isBorder = x_glob == 0 || y_glob == 0 || x_glob >= length - 1 || y_glob >= length-1;

    double temp((input[(y_glob-1)*(length)+(x_glob-1)] +
                 input[(y_glob-1)*(length)+(x_glob)]   +
                 input[(y_glob-1)*(length)+(x_glob+1)] +
                 input[(y_glob)*(length)+(x_glob-1)]   +
                 input[(y_glob)*(length)+(x_glob)]     +
                 input[(y_glob)*(length)+(x_glob+1)]   +
                 input[(y_glob+1)*(length)+(x_glob-1)] +
                 input[(y_glob+1)*(length)+(x_glob)]   +
                input[(y_glob+1)*(length)+(x_glob+1)]) / 9);
    output[element_id] = (isCenter & 1000) + (isBorder & 0) + (!(isCenter || isBorder) & temp);
}
/**/

/*With mult instead of branches single kernel
__global__
void gpu_computation(double* input, double* output, int length){
    int x_glob = (blockIdx.x * blockDim.x) + threadIdx.x + 1;   //+1 to avoid first column
    int y_glob = (blockIdx.y * blockDim.y) + threadIdx.y + 1;   //+1 to avoid first row
    int element_id = (y_glob * length) + x_glob;
    int isCenter = ((x_glob == length/2-1) || (x_glob == length/2)) && ((y_glob == length/2-1) || (y_glob == length/2));
    int isBorder = x_glob == 0 || y_glob == 0 || x_glob >= length - 1 || y_glob >= length-1;

    output[element_id] = isCenter * 1000 + isBorder * 0 + !(isCenter || isBorder) *
                                            ((input[(y_glob-1)*(length)+(x_glob-1)] +
                                            input[(y_glob-1)*(length)+(x_glob)]   +
                                            input[(y_glob-1)*(length)+(x_glob+1)] +
                                            input[(y_glob)*(length)+(x_glob-1)]   +
                                            input[(y_glob)*(length)+(x_glob)]     +
                                            input[(y_glob)*(length)+(x_glob+1)]   +
                                            input[(y_glob+1)*(length)+(x_glob-1)] +
                                            input[(y_glob+1)*(length)+(x_glob)]   +
                                            input[(y_glob+1)*(length)+(x_glob+1)] ) / 9);
}
*/

/*Without branches single kernel
__global__
void gpu_computation(double* input, double* output, int length){
    int x_glob = (blockIdx.x * blockDim.x) + threadIdx.x + 1;   //+1 to avoid first column
    int y_glob = (blockIdx.y * blockDim.y) + threadIdx.y + 1;   //+1 to avoid first row
    int element_id = (y_glob * length) + x_glob;
    int isCenter = ((x_glob == length/2-1) || (x_glob == length/2)) && ((y_glob == length/2-1) || (y_glob == length/2));
    int isBorder = x_glob == 0 || y_glob == 0 || x_glob >= length - 1 || y_glob >= length-1;

    output[element_id] = isCenter ? 1000 : (isBorder ? 0 : (input[(y_glob-1)*(length)+(x_glob-1)] +
                                            input[(y_glob-1)*(length)+(x_glob)]   +
                                            input[(y_glob-1)*(length)+(x_glob+1)] +
                                            input[(y_glob)*(length)+(x_glob-1)]   +
                                            input[(y_glob)*(length)+(x_glob)]     +
                                            input[(y_glob)*(length)+(x_glob+1)]   +
                                            input[(y_glob+1)*(length)+(x_glob-1)] +
                                            input[(y_glob+1)*(length)+(x_glob)]   +
                                            input[(y_glob+1)*(length)+(x_glob+1)] ) / 9);
}
*/

/*With branches single kernel
__global__
void gpu_computation(double* input, double* output, int length){
    int x_glob = (blockIdx.x * blockDim.x) + threadIdx.x + 1;   //+1 to avoid first column
    int y_glob = (blockIdx.y * blockDim.y) + threadIdx.y + 1;   //+1 to avoid first row
    int element_id = (y_glob * length) + x_glob;
    if ( ((x_glob == length/2-1) || (x_glob == length/2)) && ((y_glob == length/2-1) || (y_glob == length/2)) 
        || x_glob >= length - 1 || y_glob >= length-1)
    {
        return;
    }
    output[element_id] = (input[(y_glob-1)*(length)+(x_glob-1)] +
                                            input[(y_glob-1)*(length)+(x_glob)]   +
                                            input[(y_glob-1)*(length)+(x_glob+1)] +
                                            input[(y_glob)*(length)+(x_glob-1)]   +
                                            input[(y_glob)*(length)+(x_glob)]     +
                                            input[(y_glob)*(length)+(x_glob+1)]   +
                                            input[(y_glob+1)*(length)+(x_glob-1)] +
                                            input[(y_glob+1)*(length)+(x_glob)]   +
                                            input[(y_glob+1)*(length)+(x_glob+1)] ) / 9;
}
*/