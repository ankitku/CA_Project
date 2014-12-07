#include <iostream>
#include "stdio.h"
#define R 128 
#define C 128
#define ITERS 1000000

#define g 64

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
__global__ void add(int *a, int *b, int *c) {
int gtid = (gridDim.x*blockIdx.y + blockIdx.x)*blockDim.x*blockDim.y + blockDim.x*threadIdx.y + threadIdx.x;

if (gtid < R*C)
    for(int i=0; i<ITERS; i++) 
       c[gtid] = sqrt( (float) (a[gtid] + b[gtid]));
}

__global__ void add_divergent(int *a, int *b, int *c) {
int gtid = (gridDim.x*blockIdx.y + blockIdx.x)*blockDim.x*blockDim.y + blockDim.x*threadIdx.y + threadIdx.x;

if (gtid < R*C)
  if(gtid & g)
    for(int i=0; i<ITERS; i++) 
       c[gtid] = sqrt( (float) (a[gtid] + b[gtid]));
  else
    for(int i=0; i<ITERS; i++) 
       c[gtid] = sqrt( (float) (a[gtid] - b[gtid]));
}

int main() {

//DisplayHeader();

int a[R][C] , b[R][C] , c[R][C];
int *dev_a, *dev_b, *dev_c;

cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaMalloc((void **) &dev_a, R*C*sizeof(int));
cudaMalloc((void **) &dev_b, R*C*sizeof(int));
cudaMalloc((void **) &dev_c, R*C*sizeof(int));

// Fill Arrays
for (int i = 0; i < R; i++) {
	for (int j = 0; j < C; j++) {
    a[i][j] = C*i + j,
    b[i][j] = R*C - a[i][j];
    }
}

cudaMemcpy(dev_a, a, R*C*sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(dev_b, b, R*C*sizeof(int), cudaMemcpyHostToDevice);

// Kernel invocation 
dim3 threadsPerBlock(8,8); 
dim3 numBlocks( C/threadsPerBlock.x , R/threadsPerBlock.y);

std::cout<<"numBlocks.x="<<numBlocks.x<<" numBlocks.y="<<numBlocks.y<<"\n";

cudaEventRecord(start);
add<<<numBlocks, threadsPerBlock>>>(dev_a,dev_b,dev_c);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

cudaMemcpy(c, dev_c, R*C*sizeof(int), cudaMemcpyDeviceToHost);

float elapsed_time = 0;
cudaEventElapsedTime(&elapsed_time, start, stop);

for (int i = 0; i < R; i++) {
	for (int j = 0; j < C; j++) {
//    std::cout << c[i][j] <<" ";
    }
//std::cout<<"\n";
}

std::cout<<"divergence/"<<g<<" : Elapsed time = "<<elapsed_time<<" ms\n";
return 0;
}
