#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
using namespace std;

void add_cuda_raw(const float* d_a, const float* d_b, float* d_out, int64_t size);

// 将 PyTorch Tensor 转为原始指针方式进行 GPU 运算
torch::Tensor add_torch(torch::Tensor a_cpu, torch::Tensor b_cpu) {
    TORCH_CHECK(!a_cpu.is_cuda(), "Expected CPU tensor");
    TORCH_CHECK(!b_cpu.is_cuda(), "Expected CPU tensor");
    TORCH_CHECK(a_cpu.sizes() == b_cpu.sizes(), "Input sizes must match");
    TORCH_CHECK(a_cpu.dtype() == torch::kFloat32, "Only float32 supported");

    int64_t size = a_cpu.numel();
    size_t bytes = size * sizeof(float);

    // 提取原始 CPU 数据指针
    const float* h_a = a_cpu.data_ptr<float>();
    const float* h_b = b_cpu.data_ptr<float>();

    // 分配 GPU 内存
    float* d_a; float* d_b; float* d_out;
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_out, bytes);

    // 传输数据到 GPU
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // 调用 CUDA kernel
    add_cuda_raw(d_a, d_b, d_out, size);

    // 分配 CPU 输出并从 GPU 读取结果
    torch::Tensor out_cpu = torch::empty_like(a_cpu);
    float* h_out = out_cpu.data_ptr<float>();
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // 释放显存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return out_cpu;
}


// 使用 PYBIND11_MODULE 宏将函数注册为 Python 可调用模块
// TORCH_EXTENSION_NAME 是自动从 setup.py 中传入的模块名（这里是 FS_SpMM）
// m 是模块对象，可以注册函数
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("add_kernel", &add_torch, "Add two tensors");  // 将 C++ 函数暴露为 Python 中的 forward
}