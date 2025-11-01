#include <iostream>

__global__ void kernel(void) {}

int main(){
    kernel<<<1,1>>>();
    printf("Hello World!");
    return 0;
}