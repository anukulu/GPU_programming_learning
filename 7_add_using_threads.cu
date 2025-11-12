#include <iostream>

#define DIM (1024 * 3)

__global__ void kernel(int *a, int *b, int *c){
    //maximum number of threads that can be instantiated in a block is 512 in old GPU, dont know about new ones
    int tid = threadIdx.x + blockIdx.x * blockDim.x;//threadIdx.x; if we were only using 1 block and multiple threads
    // if(tid < DIM){ //this now ensures that the index does not go above the array size
    //     c[tid] = a[tid] + b[tid];
    // }
    // we use this when the size of the array is very large. In that case, we dont want to create threads for every
    // single element. So, we use grid stride so that each thread is responsible for multiple elements in the
    // array (specifically elemnts that are total number of threads -> blockDim.x * gridDim.x apart from each other on the array)
    while(tid < DIM){
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}
/*
    threadIdx.x → which thread inside the block
    blockIdx.x → which block you’re in
    blockDim.x → how many threads per block
    gridDim.x → how many blocks in total(used in grid-stride loop)
*/

int main()
{
    int a[DIM], b[DIM], c[DIM];
    int *dev_a, *dev_b, *dev_c;

    cudaMalloc((void**)&dev_a, DIM * sizeof(int));
    cudaMalloc((void**)&dev_b, DIM * sizeof(int));
    cudaMalloc((void **)&dev_c, DIM * sizeof(int));


    for (int i =0 ; i< DIM; i++){
        a[i] = i;
        b[i] = i * i; 
    }

    cudaMemcpy(dev_a, a, sizeof(int) * DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, sizeof(int) * DIM, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c, c, sizeof(int) * DIM, cudaMemcpyHostToDevice);

    // (DIM + (127) / 128) fails when number of blocks exceed a certain number. So, we would also like to keep it small.
    // anyways the grid stride will handle any large array
    kernel<<<128, 128>>>(dev_a, dev_b, dev_c); // we are running a round up of the number of blocks, and 128 threads

    cudaMemcpy(c, dev_c, DIM * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < DIM; i++){
        printf("%d\n", c[i]);
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}