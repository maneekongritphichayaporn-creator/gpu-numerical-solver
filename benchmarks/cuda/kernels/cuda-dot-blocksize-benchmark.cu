#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <iomanip>

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
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    shared[tid] = (i < n) ? a[i] * b[i] : 0.0;
    __syncthreads();

    // Valid for the power-of-two block sizes tested below.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    if (tid == 0) partialSums[blockIdx.x] = shared[0];
}

__global__ void reduceKernel(const double* input, double* output, int n)
{
    extern __shared__ double shared[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
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
    dotKernel<<<currentSize, blockSize, blockSize * sizeof(double)>>>(d_a, d_b, d_bufferA, n);
    CUDA_CHECK(cudaGetLastError());

    double* current = d_bufferA;
    double* next = d_bufferB;
    while (currentSize > 1) {
        int nextSize = (currentSize + blockSize - 1) / blockSize;
        reduceKernel<<<nextSize, blockSize, blockSize * sizeof(double)>>>(current, next, currentSize);
        CUDA_CHECK(cudaGetLastError());
        currentSize = nextSize;
        std::swap(current, next);
    }
    CUDA_CHECK(cudaMemcpyAsync(d_result, current, sizeof(double), cudaMemcpyDeviceToDevice));
}

struct Result { int block; int grid; double msPerCall; double value; };

int main()
{
    const int N = 1'000'000;
    const int WARMUP = 20;
    const int REPEATS = 200;
    const int blockSizes[] = {128, 256, 512};
    const double expected = 2.0 * N;

    std::vector<double> h_a(N, 1.0), h_b(N, 2.0);
    double *d_a = nullptr, *d_b = nullptr, *d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<Result> results;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "CUDA Dot Product Block Size Benchmark\n";
    std::cout << "N=" << N << ", warmup=" << WARMUP << ", batch calls=" << REPEATS << "\n\n";
    std::cout << "Block  Grid     Average ms/call     Result        Correctness\n";
    std::cout << "---------------------------------------------------------------\n";

    for (int blockSize : blockSizes) {
        int gridSize = (N + blockSize - 1) / blockSize;
        int maxPartials = gridSize;
        double *d_bufferA = nullptr, *d_bufferB = nullptr;
        CUDA_CHECK(cudaMalloc(&d_bufferA, maxPartials * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_bufferB, maxPartials * sizeof(double)));

        for (int i = 0; i < WARMUP; ++i)
            dotGPUReuse(d_a, d_b, d_result, d_bufferA, d_bufferB, N, blockSize);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < REPEATS; ++i)
            dotGPUReuse(d_a, d_b, d_result, d_bufferA, d_bufferB, N, blockSize);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsedMs = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));
        double msPerCall = static_cast<double>(elapsedMs) / REPEATS;
        double result = 0.0;
        CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost));
        bool pass = std::abs(result - expected) < 1e-6;

        std::cout << std::setw(5) << blockSize << "  "
                  << std::setw(5) << gridSize << "      "
                  << std::setw(12) << msPerCall << "      "
                  << std::setw(10) << result << "    "
                  << (pass ? "PASS" : "FAIL") << "\n";
        results.push_back({blockSize, gridSize, msPerCall, result});

        CUDA_CHECK(cudaFree(d_bufferA));
        CUDA_CHECK(cudaFree(d_bufferB));
    }

    auto best = std::min_element(results.begin(), results.end(),
        [](const Result& a, const Result& b) { return a.msPerCall < b.msPerCall; });
    std::cout << "\nFastest measured block size: " << best->block
              << " threads/block (" << best->msPerCall << " ms/call)\n";
    std::cout << "Note: compare repeated runs; GPU clock state and cloud load can affect timings.\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_result));
    return 0;
}
