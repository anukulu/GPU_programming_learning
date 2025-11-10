#include <iostream>
#include "cuda-by-example/src/common/book.h"
#include "cuda-by-example/src/common/cpu_bitmap.h"

#define DIM 1000

struct myComplex {
    float r;
    float i;
    __device__ myComplex(float a, float b) : r(a) , i(b) {}
    __device__ float magnitude2(void) const{
        return r * r + i * i;
    }
    __device__ myComplex operator*(const myComplex& other) const{
        return myComplex(r * other.r - i * other.i,  i * other.r + r * other.i);
    }
    __device__ myComplex operator+(const myComplex& other) const{
        return myComplex(r + other.r , i + other.i);
    }
};

__device__ int julia(int x, int y) {
    const float scale = 1.5;
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    myComplex c(-0.8, 0.156);
    myComplex a(jx, jy);

    for (int i = 0; i < 200; i++){
        a = a * a + c;
        if(a.magnitude2() > 1000){
            return 0;
        }
    }
    return 1;
}

__global__ void kernel(unsigned char *ptr){
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x; //gridDim is a global variable that holds the dimension of the grid that was launched

    int julVal = julia(x, y);
    ptr[offset * 4 + 0] = 255 * julVal;
    ptr[offset * 4 + 1] = 0;
    ptr[offset * 4 + 2] = 0;
    ptr[offset * 4 + 3] = 255;
}

int main(void){

    CPUBitmap bitmap(DIM, DIM);
    unsigned char* dev_bitmap;

    cudaMalloc((void**)&dev_bitmap, bitmap.image_size());

    // we pass a grid only because its easier to get the offest, the bitmap is laid out sequentially only
    dim3 grid(DIM, DIM);

    // this launches a (DIM * DIM) grid with z axis = 1, thtats why grid is dim3
    kernel<<<grid, 1>>>(dev_bitmap);
    cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);

    bitmap.display_and_exit();

    cudaFree(dev_bitmap);

    return 0;
}