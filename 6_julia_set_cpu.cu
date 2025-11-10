#include <iostream>
#include "cuda-by-example/src/common/book.h"
#include "cuda-by-example/src/common/cpu_bitmap.h"

#define DIM 1000

int julia(int, int);
void kernel(unsigned char*);

struct myComplex
{
    float r;
    float i;
    myComplex(float a, float b) : r(a), i(b) {} // r(a), i(b) is called the initializer list and this function is basically the constructor for the struct
    float magnitude2(void) { return r * r + i * i; }
    myComplex operator*(const myComplex &a)
    {
        return myComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }
    myComplex operator+(const myComplex &a)
    {
        return myComplex(r + a.r, i + a.i);
    }
};

int julia(int x, int y){
    const float scale = 1.5;
    float jx = scale * (float) (DIM/2 - x) / (DIM/2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    myComplex c(-0.8, 0.156);
    myComplex a(jx, jy);

    for(int i =0; i < 200; i++){
        a = a * a + c;
        if(a.magnitude2() > 1000){
            return 0;
        }
    }
    return 1;
}

void kernel(unsigned char *ptr)
{
    for (int i = 0; i < DIM; i++)
    {
        for (int j = 0; j < DIM; j++)
        {

            int offset = i + j * DIM;
            int juliaVal = julia(i, j);

            // the product with 4 is basically because, each pixel has (R, G, B, alpha) values and they are laid out sequentially
            ptr[offset * 4 + 0] = 0;
            ptr[offset * 4 + 1] = 0;
            ptr[offset * 4 + 2] = 255 * juliaVal;
            ptr[offset * 4 + 3] = 255;
        }
    }
}

int main(int argc, char *argv[])
{

    CPUBitmap bitmap(DIM, DIM);
    unsigned char *ptr = bitmap.get_ptr();
    kernel(ptr);
    bitmap.display_and_exit();

    return 0;
}