#include <cstdio>
#include <cassert>
#include <cuda_runtime.h>
#include <chrono>

// Busy-wait kernel: spins for 'cycles' GPU clock ticks on each block (one block per SM)
__global__ void busy_wait_kernel(unsigned long long cycles) {
    unsigned long long start = clock64();
    // spin until elapsed cycles
    while (clock64() - start < cycles) {
        // busy-wait
    }
}

int main() {
    // Measure host wall-clock start time
    auto host_start = std::chrono::high_resolution_clock::now();

    // 1) Select device
    cudaError_t err = cudaSetDevice(0);
    assert(err == cudaSuccess);

    // 2) Query device properties to get clock rate and SM count
    cudaDeviceProp devProp;
    err = cudaGetDeviceProperties(&devProp, 0);
    assert(err == cudaSuccess);

    int numSMs = devProp.multiProcessorCount;
    // Clock rate is in kHz; compute cycles for ~5 seconds
    unsigned long long cycles = static_cast<unsigned long long>(devProp.clockRate) * 1000ULL * 2ULL;
    printf("Launching %d blocks (one per SM), each busy-waiting for ~5 s: cycles=%llu\n", numSMs, cycles);

    // 3) Create CUDA events to time GPU execution
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);
    cudaEventRecord(start_event);

    // 4) Launch busy-wait kernel with one block per SM
    busy_wait_kernel<<<numSMs, 1>>>(cycles);
    cudaEventRecord(stop_event);

    // 5) Synchronize and measure GPU elapsed time
    cudaEventSynchronize(stop_event);
    float gpu_time_ms = 0;
    cudaEventElapsedTime(&gpu_time_ms, start_event, stop_event);

    // 6) Measure host elapsed time
    auto host_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> host_elapsed = host_end - host_start;

    // Print results
    printf("GPU busy-wait time: %.2f ms\n", gpu_time_ms);
    printf("Host wall-clock time: %.2f s\n", host_elapsed.count());

    // Cleanup
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    return 0;
}
