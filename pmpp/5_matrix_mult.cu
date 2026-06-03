#include "cuda.h"
#include "stdio.h"


struct Matrix {
    int height;
    int width;
    float *elements;
};

__global__
void mat_mul(float *A, float *B, float *C, int A_ht, int A_wd, int B_wd){
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    
    if(row < A_ht && col < B_wd){
        int op_offset = col + row * B_wd;

        float summ = 0.0f;

        for (int i = 0; i < A_wd; i++){
            summ += (A[i + row * A_wd] * B[i * B_wd + col]);
        }
        C[op_offset] = summ;
    }

}





int main() {


    Matrix A, B, C;

    A.height = 2;
    A.width = 2;
    A.elements = (float *)malloc(A.height * A.width * sizeof(float));

    B.height = 2;
    B.width = 2;
    B.elements = (float *)malloc(B.height * B.width * sizeof(float));

    C.height = A.height;
    C.width = B.width;
    C.elements = (float *) malloc(A.height * B.width * sizeof(float));


    for (int i = 0; i < A.height ; i++){
        for(int j = 0; j < A.width; j++){
            int offset = i * A.width + j * 1.0f;
            A.elements[offset] = offset;
        }
    }

    for (int i = 0; i < B.height ; i++){
        for(int j = 0; j < B.width; j++){
            int offset = i * B.width + j * 1.0f;
            B.elements[offset] = offset;
        }
    }

    float *A_d, *B_d, *C_d;

    cudaMalloc((void **)&A_d, A.height * A.width * sizeof(float));
    cudaMalloc((void **)&B_d, B.height * B.width * sizeof(float));
    cudaMalloc((void **)&C_d, C.height * C.width * sizeof(float));


    cudaMemcpy(A_d, A.elements, A.height * A.width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B.elements, B.height * B.width * sizeof(float), cudaMemcpyHostToDevice);


    dim3 block(16, 16, 1);
    dim3 grid(ceil(C.width/16.0), ceil(C.width/16.0), 1);
    //run the kernel here
    mat_mul<<<grid, block>>>(A_d, B_d, C_d, A.height, A.width, B.width);

    cudaMemcpy(C.elements, C_d, C.height * C.width * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < C.height ; i++){
        for(int j = 0; j < C.width; j++){
            int offset = i * C.width + j;
            printf("%lf\t", C.elements[offset]);
        }
        printf("\n");
    }


    free(A.elements);
    free(B.elements);
    free(C.elements);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;

}
