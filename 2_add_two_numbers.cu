#include <iostream>

__global__ void add(int a, int b, int *c){ // device code
    *c = a + b;
}

int main(){
    int c;
    int *dev_c;

    cudaMalloc((void**)&dev_c, sizeof(int)); // dev_c is a pointer to a pointer ie : int** so it is casted to void**
    // any pointer allocated with cudaMalloc cannot be used to read/write memory using code that executes on the host 
    add<<<1,1>>>(2,3, dev_c);
    cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost); // copying from the device to the host
    printf("2 + 3 = %d\n", c);
    cudaFree(dev_c); // freeing on the device 

    return 0;
}