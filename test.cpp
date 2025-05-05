// hw.cpp
#include <assert.h>
#include <stdio.h>
#include <cuda_runtime.h>

int main(){
    // 1. 选择设备
    cudaError_t err = cudaSetDevice(0);
    assert(err == cudaSuccess);

    // 2. 获取并打印设备属性
    cudaDeviceProp devProp;
    err = cudaGetDeviceProperties(&devProp, 0);
    assert(err == cudaSuccess);
    printf("Device Name                : %s\n", devProp.name);
    printf("Multi-Processor Count      : %d\n\n", devProp.multiProcessorCount);

    // 3. 查询并打印初始显存状态
    size_t freeMem, totalMem;
    err = cudaMemGetInfo(&freeMem, &totalMem);
    assert(err == cudaSuccess);
    printf("Before Allocation:\n");
    printf("  Total Global Memory      : %.2f MB\n", totalMem / 1024.0 / 1024.0);
    printf("  Available (Free) Memory  : %.2f MB\n\n", freeMem / 1024.0 / 1024.0);

    // 4. 分配 5 GB 的显存
    const size_t allocSize = 5ULL * 1024 * 1024 * 1024;  // 5 GB in bytes
    void* d_ptr = nullptr;
    err = cudaMalloc(&d_ptr, allocSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Requested Allocation       : %.2f MB\n\n", allocSize / 1024.0 / 1024.0);

    // 5. 再次查询并打印显存状态
    err = cudaMemGetInfo(&freeMem, &totalMem);
    assert(err == cudaSuccess);
    printf("After Allocation:\n");
    printf("  Total Global Memory      : %.2f MB\n", totalMem / 1024.0 / 1024.0);
    printf("  Available (Free) Memory  : %.2f MB\n\n", freeMem / 1024.0 / 1024.0);

    // 6. 释放内存并退出
    err = cudaFree(d_ptr);
    assert(err == cudaSuccess);
    return 0;
}
