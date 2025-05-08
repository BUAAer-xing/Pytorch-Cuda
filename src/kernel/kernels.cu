#include <cuda_runtime.h>
#include <cstdint>

// 简单的加法 CUDA kernel（与前相同）
__global__ void add_kernel(const float* a, const float* b, float* out, int64_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

// 裸指针 CUDA kernel wrapper，供 benchmark.cpp 调用
void add_cuda_raw(const float* d_a, const float* d_b, float* d_out, int64_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(d_a, d_b, d_out, size);
}