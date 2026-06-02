#include "cuda.h"

#define STB_IMAGE_IMPLEMENTATION
#include "libs/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "libs/stb_image_write.h"

#define FILTER_SIZE 3
#define NUM_THREADS 16.0f


__constant__ int SOBEL_X[9] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
};

__constant__ int SOBEL_Y[9] = {
    -1, -2, -1,
    0, 0, 0,
    1, 2, 1
};


__global__
void conv_img(unsigned char *img_in, unsigned char *img_out, int width, int height){

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if(col < width && row < height){

        // normally the grid is always defined based on the output's dimension
        // and same dimensions are used inside the kernel

        int offset = col + row * width;
        // we get back the original image shape because the last thread 
        // in the output image's width or height (this is the same as the dim 
        // we have defined for the grid) will be working on parts of the input
        // image that is outside the boundary of the thread grid (or the output image dimensions) by FILTER_SIZE - 1  
        int width_in = width + FILTER_SIZE - 1;
        int height_in = height + FILTER_SIZE - 1;
        
        float mag_x = 0.0f;
        float mag_y = 0.0f;

        for(int i = 0; i < FILTER_SIZE; ++i){
            for(int j = 0; j < FILTER_SIZE; ++j){

                int nbr_row = row + i;
                int nbr_col = col + j;

                int filter_offset = i * FILTER_SIZE + j;

                if(nbr_row >= 0 && nbr_row < height_in && nbr_col >= 0 && nbr_col < width_in){
                    // the width_in is used here to index the part of the image currently being
                    // worked on by the "output image" shape thread grid
                    mag_x += SOBEL_X[filter_offset] * img_in[nbr_row * width_in + nbr_col];
                    mag_y += SOBEL_Y[filter_offset] * img_in[nbr_row * width_in + nbr_col];
                }
            }
        }
        // E = sqrt(Ex^2 + Ey^2) for sobel filters
        float mag = sqrtf(mag_x * mag_x + mag_y * mag_y);
        img_out[offset] = (unsigned char) fminf(255.0f, fabsf(mag));

    }
}

int main(){

    int width, height, actual_channels;
    int used_channels = 1;

    unsigned char *img_in = stbi_load(
        "images/grayed.jpg",
        &width,
        &height,
        &actual_channels,
        used_channels
    );

    unsigned char *img_out, *img_in_d, *img_out_d;


    int size_in = width * height * used_channels;
    int width_out = width - FILTER_SIZE + 1;
    int height_out = height - FILTER_SIZE + 1; 
    int size_out = width_out * height_out;

    img_out = (unsigned char *)malloc(size_out * sizeof(unsigned char));

    cudaMalloc((void **)&img_in_d, size_in * sizeof(unsigned char));
    cudaMalloc((void **)&img_out_d, size_out * sizeof(unsigned char));

    cudaMemcpy(img_in_d, img_in, size_in * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blocks(16, 16, 1);
    dim3 grids(ceil(width_out / NUM_THREADS), ceil(height_out / NUM_THREADS), 1);

    //running the kernel here
    conv_img<<<grids, blocks>>>(img_in_d, img_out_d, width_out, height_out);

    cudaError_t err;
    err = cudaGetLastError();
    printf("Launch: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    printf("Execution: %s\n", cudaGetErrorString(err));

    cudaMemcpy(img_out, img_out_d, size_out * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    stbi_write_jpg(
        "images/conv_filter.jpg",
        width_out,
        height_out,
        used_channels,
        img_out,
        95
    );

    free(img_out);
    cudaFree(img_in_d);
    cudaFree(img_out_d);

    return 0;
}