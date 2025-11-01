#include <iostream>

int main(){

    cudaDeviceProp prop;

    int count;
    cudaGetDeviceCount(&count);

    for(int i = 0; i < count; i++){
        cudaGetDeviceProperties(&prop, i);
        printf("%s\n", prop.name);
        printf("%ld\n", prop.totalConstMem);
    }

    return 0;
}