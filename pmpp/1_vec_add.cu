#include "cuda.h"
#include "stdio.h"

// the addition kernel
__global__
void vecAddKernel(float *A, float *B, float *C, int n){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    // since the vec dimension might not be a multiple of the threads launched in a grid,
    // this makes sure that only the threads that are actually responsible operate on the 
    // data. For eg: n= 100, blockDim = 32, then the minm number of threads that needs to be
    // launched is 128, but last 28 threads should not operate on the vector
    if(i < n) C[i] = A[i] + B[i];
}

void vec_add(float *A, float *B, float *C, int n){
    
    // allocation of sizes for the arrays in device global memory
    float *A_d, *B_d, *C_d;
    int size = sizeof(float) * n;

    cudaMalloc((void **)&A_d, size);
    cudaMalloc((void **)&B_d, size);
    cudaMalloc((void **)&C_d, size);

    // copy the data from the host to the device
    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    // number of blocks, number of threads per block
    vecAddKernel<<<ceil(n / 256.0), 256>>>(A_d, B_d, C_d, n);

    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    // free the memory on the device global memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

}


int main(){

    int n = 100;
    float A_h[n], B_h[n], C_h[n];
    for (int i = 0; i < n; i ++){
        A_h[i] = i * 1.0f;
        B_h[i] = i * 2.3f;
    }
    vec_add(A_h, B_h, C_h, n);

    for(int i =0; i < n; i++){
        printf("%lf \n", C_h[i]);
    }

    return 0;
}