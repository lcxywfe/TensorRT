#include "kernel.h"
#include <cuda_runtime_api.h>

__global__ void float2half_kern(const float* input, __half* output,
                             const int data_size) {
    int global_offset = blockIdx.x * blockDim.x + threadIdx.x;
    while (global_offset < data_size) {
        output[global_offset] = __float2half(input[global_offset]);
        global_offset += gridDim.x * blockDim.x;
    }
}


void dcnFloat2half(const float* input, __half* output, int data_size) {
    const int block_size = 256;
    const int task_per_thread = 8;
    const int task_per_block = (block_size * task_per_thread);
    int grid_size = (data_size + task_per_block - 1) / task_per_block;
    float2half_kern<<<grid_size, block_size>>>(input, output, data_size);
}

