#include <iostream>
#include "cuda-by-example/src/common/book.h"

#define imin(a, b) (a < b?a:b)
#define sum_squares(x) (x * (x+1) * (2*x + 1)/6)

const int N = 1024;
const int threadsPerBlock = 256;


__global__ void dot(float *a, float *b, float *c){
    __shared__ float cache[threadsPerBlock];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    float temp = 0;
    while (tid < N)
    {
        temp += a[tid] * b[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp;

    // This call makes sure that every thread in the block has completed instructions 
    // prior to this statement before the hardware will execute the next instruction on
    // any thread.
    __syncthreads();

    // for this block to run, the threadsPerBlock must be a multiple of 2, becuase the 
    // cahceIndex dependds on it and its getting decremented by two everytime
    int i = blockDim.x / 2;
    while (i != 0)
    {
        if (cacheIndex < i){
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0){ // doesnt matter which index, it can be any index, since any
        // one of the threads inside the block can write back to the array
        // for each block there is only one number that resides in the 0th position of the
        // cache for that block, so we write it to the output array
        c[blockIdx.x] = cache[0];
    }
    
}

// the second part makes sure that the number of blocks is at least 1
const int blocksPerGrid = imin(32, (N + threadsPerBlock-1)/threadsPerBlock);


int main(){

    float *a, *b, c, *partial_c;
    float *dev_a, *dev_b, *dev_partial_c;

    a = new float[N];
    b = new float[N];
    partial_c = new float[blocksPerGrid];

    HANDLE_ERROR(cudaMalloc((void **) &dev_a, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **) &dev_b, N * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **) &dev_partial_c, blocksPerGrid * sizeof(float)));

    for(int i = 0; i < N; i++){
        a[i] = i;
        b[i] = i * 2;
    }

    HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice));

    dot<<<blocksPerGrid, threadsPerBlock>>>(dev_a, dev_b, dev_partial_c);

    HANDLE_ERROR(cudaMemcpy(partial_c, dev_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));

    c = 0;
    for (int i = 0; i< blocksPerGrid; i++){
        c += partial_c[i];
    }

    printf("Does GPU value %lf = %lf ?\n", c, 2 * sum_squares((float) (N-1) ));

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_partial_c);

    
}