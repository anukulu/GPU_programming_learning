#include "cuda.h"
#include "cuda-by-example/src/common/book.h"
#include "cuda-by-example/src/common/cpu_bitmap.h"

#define SPHERES 20
#define INF 2e10f
#define rnd(x) (x * rand() / RAND_MAX)
#define DIM 1024

struct Sphere_ {

    float x, y, z;
    float r, g, b;
    float radius;

    __device__ float hit(float ox, float oy, float *n){
        float dx = ox - x;
        float dy = oy - y;
        // this places the point relative to the circle in the x-y plane. 
        if(dx*dx + dy*dy < radius*radius){ // this is simple x^2 + y^2 < r^2 
            float dz = sqrtf(radius * radius - dx * dx - dy * dy); // this is the z axis width of the point 
            //of the sphere, becomes smaller as we move towards the radius
            *n = dz / sqrtf(radius * radius); // this is the normal realtive to the centre. It becomes smaller 
            //as we move towards the radius
            return z + dz; // this is done to find the z of each point on the projected sphere. the larger it is
            // the closer to the screen
        }
        return -INF;
    }
};

// using constant memory has two benifits:
// - boradcasts memory to half warp (collection of 32/2 threads woven together)
// - caches the memory in GPU, so the number of reads is less for multiple
// threads accessing the same memory location
__constant__ Sphere_ dev_s[SPHERES];

__global__ void kernel(unsigned char *ptr){//, Sphere_ *dev_s){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    // the origin is now located at the centre of the screen
    float ox = x - (DIM/2);
    float oy = y - (DIM/2);

    float r = 0, g = 0, b = 0;
    float maxd = -INF;
    for (int i = 0; i < SPHERES; i++){
        float n;
        float hit_d = dev_s[i].hit(ox, oy, &n);
        if(hit_d > maxd){
            float fscale = n;
            r = dev_s[i].r * fscale;
            g = dev_s[i].g * fscale;
            b = dev_s[i].b * fscale;
            maxd = hit_d;
        }
    }

    ptr[offset * 4 + 0] = (int)(r * 255);
    ptr[offset * 4 + 1] = (int)(g * 255);
    ptr[offset * 4 + 2] = (int)(b * 255);
    ptr[offset * 4 + 3] = 255;

}

int main(){
    // Sphere_ *dev_s;

    cudaEvent_t start, stop;

    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));
    
    CPUBitmap bitmap(DIM, DIM);
    unsigned char *dev_bitmap;

    Sphere_ *temp_spheres = (Sphere_*) malloc(sizeof(Sphere_) * SPHERES);
    for (int i = 0; i < SPHERES; i++){
        temp_spheres[i].r = rnd(1.0f);
        temp_spheres[i].g = rnd(1.0f);
        temp_spheres[i].b = rnd(1.0f);
        temp_spheres[i].x = rnd(1000.0f) - 500;
        temp_spheres[i].y = rnd(1000.0f) - 500;
        temp_spheres[i].z = rnd(1000.0f) - 500;
        temp_spheres[i].radius = rnd(100.0f) + 20;
    }

    HANDLE_ERROR(cudaMalloc((void **)&dev_bitmap, bitmap.image_size()));

    HANDLE_ERROR(cudaMemcpyToSymbol(dev_s, temp_spheres, sizeof(Sphere_) * SPHERES));
    // HANDLE_ERROR(cudaMalloc((void **)&dev_s, sizeof(Sphere_) * SPHERES));
    // HANDLE_ERROR(cudaMemcpy(dev_s, temp_spheres, sizeof(Sphere_) * SPHERES, cudaMemcpyHostToDevice));

    free(temp_spheres);

    dim3 grids(DIM/16, DIM/16);
    dim3 threads(16, 16);

    // have to pass the dev_s address to the kernel because its a host variable 
    // pointing to a gpu location and cant be accessed inside the kernel
    kernel<<<grids, threads>>>(dev_bitmap);//, dev_s);

    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaFree(dev_bitmap));
    
    //since we are using a constant memory, we do not need to free the allocated space anymore
    // HANDLE_ERROR(cudaFree(dev_s));
    
    // printing the time taken to generated the frame
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    // the event command given to the gpu is normally asynchronous, meaning once the coomand is given 
    // to the gpu, the cpu might return some value in stop (since the cpu does not wait for the gpu 
    // to finish its task) while the gpu is still doing its work.
    // to make sure that the stop is recorded only when gpu has completely finished its task, we
    // use the event synchronization, basically making it blocking, so that stop is actually recorded
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    printf("Time to generate : %3.1f ms\n", elapsedTime);
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    bitmap.display_and_exit();

}