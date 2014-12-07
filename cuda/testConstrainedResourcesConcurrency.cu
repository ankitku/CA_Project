#include <iostream>
#include <stdint.h>
#include "stdio.h"
#include "helper_cuda.h"
#include "helper_string.h"

#define R1 1 
#define C1 8
#define H1 1
#define R2 8
#define C2 1
#define H2 1
#define numThrdx 1
#define numThrdy 1
#define numThrdz 1
#define NUM_STREAMS 2

__global__ void wait_kernel(uint64_t delta) {
uint64_t start = clock64();
uint64_t stop = start + delta;
while(clock64() < stop);  // you will need '-arch=sm_20' for this.
}

int main() {

cudaStream_t stream[NUM_STREAMS];
cudaDeviceProp deviceProp;

for(int i=0; i<NUM_STREAMS; i++) {
   cudaStreamCreate(&stream[i]);
}

float kernel_time = 100; // time the kernel should run in ms
clock_t time_clocks = (clock_t)(kernel_time * deviceProp.clockRate);

// Kernel invocation 
dim3 threadsPerBlock(numThrdx,numThrdy,numThrdz); 
dim3 numBlocks1( C1/threadsPerBlock.x, R1/threadsPerBlock.y, H1/threadsPerBlock.z );
std::cout<<"numBlocks1.x="<<numBlocks1.x<<" numBlocks1.y="<<numBlocks1.y<<" numBlocks1.z="<<numBlocks1.z<<"\n";

// Kernel invocation 
dim3 numBlocks2(C2/threadsPerBlock.x, R2/threadsPerBlock.y, H2/threadsPerBlock.z );
std::cout<<"numBlocks2.x="<<numBlocks2.x<<" numBlocks2.y="<<numBlocks2.y<<" numBlocks2.z="<<numBlocks2.z<<"\n";

wait_kernel<<<numBlocks1, threadsPerBlock, 0, stream[0]>>>(time_clocks);
wait_kernel<<<numBlocks2, threadsPerBlock, 0, stream[1]>>>(time_clocks);

cudaDeviceReset();
return 0;
}
