#include <iostream>
#include "stdio.h"
#include "time.h"
#define R 1 
#define C 32 
#define ITERS 100000

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

__global__ void zero_divergent(int *a, unsigned long long *time) {
//int gtid = (gridDim.x*blockIdx.y + blockIdx.x)*blockDim.x*blockDim.y + blockDim.x*threadIdx.y + threadIdx.x;
unsigned long long start = clock();

for(int i=0; i<R*C; i++)
   {
/*   if (gtid != i)
      {  
         continue;
      } 
      else
      {
         a[gtid] = (get_smid());
      }
*/   }

unsigned long long finish = clock();
*time = (finish - start);
}

int main() {

//DisplayHeader();

int a[R][C];
int *dev_a;
unsigned long long ticks = 0;
unsigned long long *time;

cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaMalloc((void **) &dev_a, R*C*sizeof(int));
cudaMalloc (&time, sizeof(unsigned long long));
// Fill Arrays
for (int i = 0; i < R; i++) {
	for (int j = 0; j < C; j++) {
    a[i][j] = 0; //C*i + j;
    }
}

cudaMemcpy(dev_a, a, R*C*sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(time, &ticks, sizeof(unsigned long long) ,cudaMemcpyHostToDevice);

// Kernel invocation 
dim3 threadsPerBlock(32,1); 
dim3 numBlocks( C/threadsPerBlock.x , R/threadsPerBlock.y);

std::cout<<"numBlocks.x="<<numBlocks.x<<" numBlocks.y="<<numBlocks.y<<"\n";

cudaEventRecord(start);
zero_divergent<<<numBlocks, threadsPerBlock>>>(dev_a, &ticks);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

cudaMemcpy(a, dev_a, R*C*sizeof(int), cudaMemcpyDeviceToHost);
cudaMemcpy(&ticks, time, sizeof(unsigned long long) ,cudaMemcpyDeviceToHost);

float elapsed_time = 0;
cudaEventElapsedTime(&elapsed_time, start, stop);

for (int i = 0; i < R; i++) {
	for (int j = 0; j < C; j++) {
  std::cout << a[i][j] <<" ";
    }
std::cout<<"\n";
}

std::cout<<"Elapsed time = "<<elapsed_time<<" ms\n";
std::cout<<"Elapsed ticks= "<<(ticks - 14)/32<<" \n";
return 0;
}
