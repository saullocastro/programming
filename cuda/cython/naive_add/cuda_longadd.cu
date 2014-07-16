#include "cuda.h"
#include "stdio.h"
#include "stdlib.h"

__global__ void longadd(float *a, float *b, float *out, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < n){
        out[tid] = a[tid] + b[tid];
        tid += gridDim.x * blockDim.x;
    }
}

extern "C" int longmain(float *a, float *b, float *out, int n) {
    float *dev_a, *dev_b, *dev_out;

    // allocate the memory on the GPU
    cudaMalloc((void**)&dev_a, n*sizeof(float));
    cudaMalloc((void**)&dev_b, n*sizeof(float));
    cudaMalloc((void**)&dev_out, n*sizeof(float));

    // copy the arrays 'a' and 'b' to the GPU
    cudaMemcpy(dev_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    longadd<<<256, 256>>>(dev_a, dev_b, dev_out, n);

    // copy the array 'out' back from the GPU to the CPU
    cudaMemcpy(out, dev_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    // free the memory allocated on the GPU
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_out);

    return 0;
}
