#include <iostream>
#include "stdio.h"
#define R 1 
#define C 32
#define H 1
#define numThrdx 32
#define numThrdy 1
#define numThrdz 1
 
#define ITERS 1000000000

#define g 64

__device__ uint get_smid(void) {
     uint ret;
     asm("mov.u32 %0, %smid;" : "=r"(ret) );
     return ret;
}

void DisplayHeader()
{
    const int kb = 1024;
    const int mb = kb * kb;
    std::cout << "NBody.GPU" <<"\n" << "=========" <<"\n" <<"\n";

    std::cout << "CUDA version:   v" << CUDART_VERSION <<"\n";    
    //std::cout << "Thrust version: v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION <<"\n" <<"\n"; 

    int devCount;
    cudaGetDeviceCount(&devCount);
    std::cout << "CUDA Devices: " <<"\n" <<"\n";

    for(int i = 0; i < devCount; ++i)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        std::cout << i << ": " << props.name << ": " << props.major << "." << props.minor <<"\n";
        std::cout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" <<"\n";
        std::cout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" <<"\n";
        std::cout << "  Constant memory: " << props.totalConstMem / kb << "kb" <<"\n";
        std::cout << "  Block registers: " << props.regsPerBlock <<"\n" <<"\n";

        std::cout << "  Warp size:         " << props.warpSize <<"\n";
        std::cout << "  Threads per block: " << props.maxThreadsPerBlock <<"\n";
        std::cout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", "<< props.maxThreadsDim[1]  << ", " << props.maxThreadsDim[2]<<" ]" <<"\n";
        std::cout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1]  << ", " << props.maxGridSize[2] << " ]" <<"\n";
        std::cout <<"\n";
    }
}


__device__ void iteratively_divergent(int *a, int* w, int *k, int i) {

   __syncthreads();
   int gtid = gridDim.x*blockDim.x*gridDim.y*blockDim.y*threadIdx.z + (gridDim.x*blockIdx.y + blockIdx.x)*blockDim.x*blockDim.y + blockDim.x*threadIdx.y + threadIdx.x;
   
   for(int i=0; i<32 ;i++)
      if(gtid <= i)
      {
         continue; 
      }      
      else
      {
     //   atomicAdd(&k[0],1);
        k[0]+=1;
        a[gtid] = k[0];
        w[gtid] = clock();
      }

}


__global__ void full_divergent(int *a, int* w, int *k) {
__syncthreads();

iteratively_divergent(a,w,k,0);
//store last value of k in a[0]
int stime = clock();
a[0] = 1;
int ftime = clock();
a[0] = ftime - stime;
}

__global__ void zero_divergent(int *a, int *k) {
int gtid = gridDim.x*blockDim.x*gridDim.y*blockDim.y*threadIdx.z + (gridDim.x*blockIdx.y + blockIdx.x)*blockDim.x*blockDim.y + blockDim.x*threadIdx.y + threadIdx.x;

int j = 0;
while(j++ < ITERS);
   a[gtid] = (99 + get_smid());

}


int main() {

DisplayHeader();

int a[R][C][H], k[1], w[R][C][H];
int *dev_a, *dev_k, *dev_warp;

cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaMalloc((void **) &dev_a, R*C*H*sizeof(int));
cudaMalloc((void **) &dev_warp, R*C*H*sizeof(int));
cudaMalloc((void **) &dev_k, sizeof(int));

// Fill Arrays
for (int k = 0; k < H; k++) {
   for (int i = 0; i < R; i++) {
      for (int j = 0; j < C; j++) {
    a[k][i][j] = 0;
    w[k][i][j] = 0;
    }
   }
}
k[0] = 0;

cudaMemcpy(dev_a, a, R*C*sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(dev_warp, w, R*C*sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(dev_k, k, sizeof(k), cudaMemcpyHostToDevice);

// Kernel invocation 
dim3 threadsPerBlock(numThrdx,numThrdy,numThrdz); 
dim3 numBlocks( C/threadsPerBlock.x, R/threadsPerBlock.y, H/threadsPerBlock.z );

std::cout<<"numBlocks.x="<<numBlocks.x<<" numBlocks.y="<<numBlocks.y<<" numBlocks.z="<<numBlocks.z<<"\n";

cudaEventRecord(start);
full_divergent<<<numBlocks, threadsPerBlock>>>(dev_a, dev_warp, dev_k);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

cudaMemcpy(a, dev_a, R*C*sizeof(int), cudaMemcpyDeviceToHost);
cudaMemcpy(w, dev_warp, R*C*sizeof(int), cudaMemcpyDeviceToHost);
cudaMemcpy(k, dev_k, sizeof(int), cudaMemcpyDeviceToHost);

float elapsed_time = 0;
cudaEventElapsedTime(&elapsed_time, start, stop);

std::cout<<"results:\n";
for (int k = 0; k < H; k++) {
   for (int i = 0; i < R; i++) {
	for (int j = 0; j < C; j++) {
            std::cout << a[k][i][j] <<" ";
    }
std::cout<<"\n";
   }
}

std::cout<<"clocks:\n";
for (int k = 0; k < H; k++) {
   for (int i = 0; i < R; i++) {
      for (int j = 0; j < C; j++) {
         std::cout << w[k][i][j] <<" ";
    }
std::cout<<"\n";
   }
}


std::cout<<"numBlocks.x="<<numBlocks.x<<" numBlocks.y="<<numBlocks.y<<" numBlocks.z="<<numBlocks.z<<"\n";
std::cout<<"Elapsed time = "<<elapsed_time<<" ms\n";

cudaDeviceReset();
return 0;
}
