#define STB_IMAGE_IMPLEMENTATION
#include "libs/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "libs/stb_image_write.h"

__global__
void blurImg(unsigned char *img_in, unsigned char *img_out, int width, int height, int channels){

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    
    if(col < width && row < height){

        int grayOffset = col + row * width;
        int rgbOffset = channels * grayOffset;

        unsigned char r = img_in[rgbOffset];
        unsigned char g = img_in[rgbOffset + 1];
        unsigned char b = img_in[rgbOffset + 2];

        img_out[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }

}

int main(){
    int width, height, channels;

    unsigned char *img_in = stbi_load(
        "images/woman.jpg",
        &width,
        &height,
        &channels,
        0
    );

    if (!img_in){
        printf("Failed to load the file. \n");
        return 1;
    }

    printf("width : %d, height %d, channels %d \n", width, height, channels);

    unsigned char *img_in_d, *img_out_d;
    unsigned char *img_out;
    int size_out = width * height;
    int size_in = size_out * channels;

    img_out = (unsigned char *) malloc(size_out * sizeof(unsigned char));
    cudaMalloc((void **)&img_in_d, size_in * sizeof(unsigned char));
    cudaMalloc((void **)&img_out_d, size_out * sizeof(unsigned char));

    cudaMemcpy(img_in_d, img_in, size_in * sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    dim3 grid(ceil(width/ 32.0), ceil(height / 32.0), 1);
    dim3 block(32, 32, 1);

    // kernel operations happen here
    blurImg<<<grid, block>>>(img_in_d, img_out_d, width, height, channels);

    cudaMemcpy(img_out, img_out_d, size_out * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    stbi_write_jpg("images/grayed.jpg", width, height, 1, img_out, 95);
    stbi_image_free(img_in);
    free(img_out);
    cudaFree(img_in_d);
    cudaFree(img_out_d);

    return 0;
}