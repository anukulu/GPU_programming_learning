#include <iostream>

int main(){
    cudaDeviceProp prop;
    int dev;

    cudaGetDevice(&dev);
    printf("The current device is %d\n", dev);

    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;
    prop.minor = 3;
    cudaChooseDevice(&dev, &prop); // basically add constraints using cudadevice prop struct to get the device id that satisfies this constraints
    cudaGetDeviceProperties(&prop, dev);
    printf("ID of the device closest to the revision 1.3 is %d and its name is %s\n", dev, prop.name);
    cudaSetDevice(dev);
}