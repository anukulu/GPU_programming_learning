#define STB_IMAGE_IMPLEMENTATION
#include "libs/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "libs/stb_image_write.h"

#define BLUR_AREA 1


__global__
void blur_img(unsigned char *img_in, unsigned char *img_out, int height, int width, int channels){

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    if(col < width && row < height){
        int offset = row * width + col;
        
        int pixVal = 0;
        int n_pixels = 0; 

        for(int i = -BLUR_AREA; i < BLUR_AREA + 1; ++i){
            for(int j = -BLUR_AREA; j < BLUR_AREA + 1; ++j){
                
                int nbrRow = row + i;
                int nbrCol = col + j;

                if(nbrCol < width && nbrCol >= 0 && nbrRow < height && nbrRow >= 0){
                    pixVal += img_in[nbrRow * width + nbrCol];
                    ++n_pixels;
                }
            }
        }
        img_out[offset] = (unsigned char) (pixVal / n_pixels);
    }
}


int main(){

    int width, height, ori_channels;
    int used_channels = 1;

    unsigned char *img_in = stbi_load(
        "images/grayed.jpg",
        &width,
        &height,
        &ori_channels,
        used_channels
    );

    printf("ori_channels = %d\n", ori_channels);
    printf("width=%d height=%d\n", width, height);

    int size = height * width * used_channels;

    if (!img_in){
        printf("Failed to load the image. \n");
        return 1;
    }

    unsigned char *img_out = (unsigned char *)malloc(size * sizeof(unsigned char));

    unsigned char *img_in_d, *img_out_d;
    cudaMalloc((void **)&img_in_d, size * sizeof(unsigned char));
    cudaMalloc((void **)&img_out_d, size * sizeof(unsigned char));
    
    cudaMemcpy(img_in_d, img_in, size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 grid(ceil(width/16.0), ceil(height/16.0), 1);
    dim3 block(16, 16, 1);

    // run the kernel here
    blur_img<<<grid, block>>>(img_in_d, img_out_d, height, width, used_channels);

    cudaError_t err = cudaGetLastError();
    printf("Launch error %s\n", cudaGetErrorString(err));
    
    cudaDeviceSynchronize();

    err = cudaGetLastError();
    printf("Runtime Error : %s\n", cudaGetErrorString(err));


    cudaMemcpy(img_out, img_out_d, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    stbi_write_jpg(
        "images/blurred.jpg",
        width,
        height,
        used_channels,
        img_out,
        95
    );

    free(img_out);
    cudaFree(img_in_d);
    cudaFree(img_out_d);

    return 0;
}