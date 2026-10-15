#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                  << " : " << cudaGetErrorString(err) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void dotKernel(const double* a, const double* b, double* partialSums, int n)
{
    extern __shared__ double shared[];
    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tid;
    shared[tid] = (i < n) ? a[i] * b[i] : 0.0;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    if (tid == 0) partialSums[blockIdx.x] = shared[0];
}

__global__ void reduceKernel(const double* input, double* output, int n)
{
    extern __shared__ double shared[];
    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tid;
    shared[tid] = (i < n) ? input[i] : 0.0;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = shared[0];
}

void dotGPUReuse(const double* d_a, const double* d_b, double* d_result,
                 double* d_bufferA, double* d_bufferB, int n, int blockSize)
{
    int currentSize = (n + blockSize - 1) / blockSize;
    dotKernel<<<currentSize, blockSize, blockSize * sizeof(double)>>>(
        d_a, d_b, d_bufferA, n);
    CUDA_CHECK(cudaGetLastError());

    double* current = d_bufferA;
    double* next = d_bufferB;
    while (currentSize > 1) {
        const int nextSize = (currentSize + blockSize - 1) / blockSize;
        reduceKernel<<<nextSize, blockSize, blockSize * sizeof(double)>>>(
            current, next, currentSize);
        CUDA_CHECK(cudaGetLastError());
        currentSize = nextSize;
        std::swap(current, next);
    }
    CUDA_CHECK(cudaMemcpyAsync(d_result, current, sizeof(double), cudaMemcpyDeviceToDevice));
}

struct Stats {
    double average;
    double median;
    double minimum;
    double maximum;
};

Stats summarize(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    const size_t n = values.size();
    const double median = (n % 2 == 0)
        ? (values[n / 2 - 1] + values[n / 2]) / 2.0
        : values[n / 2];
    return {sum / static_cast<double>(n), median, values.front(), values.back()};
}

int main()
{
    const int N = 1'000'000;
    const int WARMUP = 20;
    const int BATCH_CALLS = 200;
    const int PAIRED_RUNS = 10;
    const int blockSizes[] = {128, 256, 512};
    const double expected = 2.0 * static_cast<double>(N);
    const int maxPartials = (N + blockSizes[0] - 1) / blockSizes[0];

    std::vector<double> h_a(N, 1.0), h_b(N, 2.0);
    double *d_a = nullptr, *d_b = nullptr, *d_result = nullptr;
    double *d_bufferA = nullptr, *d_bufferB = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_bufferA, maxPartials * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_bufferB, maxPartials * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto measureBatch = [&](int blockSize) -> double {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < BATCH_CALLS; ++i) {
            dotGPUReuse(d_a, d_b, d_result, d_bufferA, d_bufferB, N, blockSize);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsedMs = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));
        return static_cast<double>(elapsedMs) / BATCH_CALLS;
    };

    std::vector<double> timings128, timings256, timings512;
    std::vector<double>* timingSets[] = {&timings128, &timings256, &timings512};

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "==============================================\n";
    std::cout << "Controlled CUDA Dot Product Block Size Benchmark\n";
    std::cout << "N=" << N << ", warmups/config=" << WARMUP
              << ", calls/batch=" << BATCH_CALLS
              << ", paired runs=" << PAIRED_RUNS << "\n";
    std::cout << "Measures dotKernel + reduction kernels + final D2D scalar copy.\n\n";

    // Warm up every configuration before collecting measurements.
    for (int blockSize : blockSizes) {
        for (int i = 0; i < WARMUP; ++i)
            dotGPUReuse(d_a, d_b, d_result, d_bufferA, d_bufferB, N, blockSize);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Alternate the execution order on each pair to reduce ordering/clock bias.
    for (int run = 0; run < PAIRED_RUNS; ++run) {
        for (int j = 0; j < 3; ++j) {
            const int idx = (run % 2 == 0) ? j : (2 - j);
            timingSets[idx]->push_back(measureBatch(blockSizes[idx]));
        }
    }

    double finalResult = 0.0;
    CUDA_CHECK(cudaMemcpy(&finalResult, d_result, sizeof(double), cudaMemcpyDeviceToHost));
    const bool pass = std::abs(finalResult - expected) < 1e-6;

    std::cout << "Block   Average(ms)   Median(ms)    Min(ms)      Max(ms)\n";
    std::cout << "---------------------------------------------------------\n";
    Stats stats[3];
    for (int i = 0; i < 3; ++i) {
        stats[i] = summarize(*timingSets[i]);
        std::cout << std::setw(5) << blockSizes[i] << "   "
                  << std::setw(11) << stats[i].average << "   "
                  << std::setw(10) << stats[i].median << "   "
                  << std::setw(10) << stats[i].minimum << "   "
                  << std::setw(10) << stats[i].maximum << "\n";
    }

    int bestIdx = 0;
    for (int i = 1; i < 3; ++i) {
        if (stats[i].median < stats[bestIdx].median) bestIdx = i;
    }
    const double improvement = (stats[0].median - stats[bestIdx].median)
        / stats[0].median * 100.0;
    std::cout << "\nFastest median: block size " << blockSizes[bestIdx]
              << " (" << stats[bestIdx].median << " ms/call)\n";
    std::cout << "Median improvement vs block 128: " << improvement << "%\n";
    std::cout << "Dot result: " << finalResult << ", expected: " << expected
              << ", correctness=" << (pass ? "PASS" : "FAIL") << "\n";
    std::cout << "Note: cloud GPU load and clock state can affect small differences.\n";
    std::cout << "==============================================\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_bufferA));
    CUDA_CHECK(cudaFree(d_bufferB));
    return pass ? 0 : 1;
}
