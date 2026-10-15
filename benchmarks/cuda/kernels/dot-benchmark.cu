#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cmath>

#include <cuda_runtime.h>


// ============================================================
// CUDA Error Checking
// ============================================================

#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        std::cerr << "CUDA Error: "                           \
                  << cudaGetErrorString(err)                 \
                  << std::endl;                               \
        std::exit(EXIT_FAILURE);                              \
    }                                                         \
} while (0)


// ============================================================
// Statistics
// ============================================================

struct Statistics
{
    double average;
    double median;
    double min;
    double max;
};


Statistics calculateStatistics(
    std::vector<double> values
)
{
    std::sort(
        values.begin(),
        values.end()
    );

    double sum =
        std::accumulate(
            values.begin(),
            values.end(),
            0.0
        );

    double average =
        sum / values.size();

    double median;

    if (values.size() % 2 == 0)
    {
        median =
            (
                values[values.size() / 2 - 1]
                +
                values[values.size() / 2]
            )
            / 2.0;
    }
    else
    {
        median =
            values[values.size() / 2];
    }

    return {
        average,
        median,
        values.front(),
        values.back()
    };
}


// ============================================================
// CPU Dot Product
// ============================================================

double dotCPU(
    const std::vector<double>& a,
    const std::vector<double>& b
)
{
    double result = 0.0;

    for (int i = 0; i < a.size(); i++)
    {
        result +=
            a[i] * b[i];
    }

    return result;
}


// ============================================================
// CUDA Dot Product
// ============================================================

__global__ void dotKernel(
    const double* a,
    const double* b,
    double* partialSums,
    int n
)
{
    extern __shared__ double shared[];

    int tid = threadIdx.x;

    int i =
        blockIdx.x * blockDim.x
        + threadIdx.x;


    // --------------------------------------------------------
    // Multiply
    // --------------------------------------------------------

    if (i < n)
    {
        shared[tid] =
            a[i] * b[i];
    }
    else
    {
        shared[tid] = 0.0;
    }

    __syncthreads();


    // --------------------------------------------------------
    // Reduction
    // --------------------------------------------------------

    for (
        int stride = blockDim.x / 2;
        stride > 0;
        stride /= 2
    )
    {
        if (tid < stride)
        {
            shared[tid] +=
                shared[tid + stride];
        }

        __syncthreads();
    }


    // --------------------------------------------------------
    // Store block result
    // --------------------------------------------------------

    if (tid == 0)
    {
        partialSums[blockIdx.x] =
            shared[0];
    }
}


// ============================================================
// Main
// ============================================================

int main()
{
    const int n = 1'000'000;

    const int runs = 10;

    const int blockSize = 256;


    // ========================================================
    // Create input
    // ========================================================

    std::vector<double> a(
        n,
        1.0
    );

    std::vector<double> b(
        n,
        2.0
    );


    // Expected:
    //
    // 1*2 + 1*2 + ... + 1*2
    //
    // = 2,000,000


    // ========================================================
    // CPU Benchmark
    // ========================================================

    std::vector<double> cpuTimes;

    double cpuResult = 0.0;

    for (int run = 0; run < runs; run++)
    {
        auto start =
            std::chrono::high_resolution_clock::now();

        double result =
            dotCPU(a, b);

        auto end =
            std::chrono::high_resolution_clock::now();

        double elapsed =
            std::chrono::duration<
                double,
                std::milli
            >(end - start).count();

        cpuTimes.push_back(elapsed);

        cpuResult = result;
    }


    // ========================================================
    // GPU configuration
    // ========================================================

    int gridSize =
        (n + blockSize - 1)
        / blockSize;


    // ========================================================
    // GPU memory
    // ========================================================

    double* d_a;
    double* d_b;
    double* d_partialSums;


    CUDA_CHECK(
        cudaMalloc(
            &d_a,
            n * sizeof(double)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_b,
            n * sizeof(double)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_partialSums,
            gridSize * sizeof(double)
        )
    );


    // ========================================================
    // Copy input to GPU
    // ========================================================

    CUDA_CHECK(
        cudaMemcpy(
            d_a,
            a.data(),
            n * sizeof(double),
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_b,
            b.data(),
            n * sizeof(double),
            cudaMemcpyHostToDevice
        )
    );


    // ========================================================
    // CUDA Events
    // ========================================================

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(
        cudaEventCreate(&start)
    );

    CUDA_CHECK(
        cudaEventCreate(&stop)
    );


    // ========================================================
    // Warm-up
    // ========================================================

    dotKernel<<<
        gridSize,
        blockSize,
        blockSize * sizeof(double)
    >>>(
        d_a,
        d_b,
        d_partialSums,
        n
    );

    CUDA_CHECK(
        cudaGetLastError()
    );

    CUDA_CHECK(
        cudaDeviceSynchronize()
    );


    // ========================================================
    // GPU Kernel Benchmark
    // ========================================================

    std::vector<double> gpuKernelTimes;

    std::vector<double> partialSums(
        gridSize
    );


    double gpuResult = 0.0;


    for (int run = 0; run < runs; run++)
    {
        CUDA_CHECK(
            cudaEventRecord(start)
        );


        dotKernel<<<
            gridSize,
            blockSize,
            blockSize * sizeof(double)
        >>>(
            d_a,
            d_b,
            d_partialSums,
            n
        );


        CUDA_CHECK(
            cudaGetLastError()
        );


        CUDA_CHECK(
            cudaEventRecord(stop)
        );

        CUDA_CHECK(
            cudaEventSynchronize(stop)
        );


        float milliseconds = 0.0f;

        CUDA_CHECK(
            cudaEventElapsedTime(
                &milliseconds,
                start,
                stop
            )
        );


        gpuKernelTimes.push_back(
            milliseconds
        );
    }


    // ========================================================
    // Copy partial sums GPU → CPU
    // ========================================================

    CUDA_CHECK(
        cudaMemcpy(
            partialSums.data(),
            d_partialSums,
            gridSize * sizeof(double),
            cudaMemcpyDeviceToHost
        )
    );


    // ========================================================
    // Final reduction on CPU
    // ========================================================

    gpuResult = 0.0;

    for (double value : partialSums)
    {
        gpuResult += value;
    }


    // ========================================================
    // Correctness
    // ========================================================

    bool correct =
        std::abs(
            cpuResult - gpuResult
        ) < 1e-9;


    // ========================================================
    // Statistics
    // ========================================================

    Statistics cpuStats =
        calculateStatistics(
            cpuTimes
        );

    Statistics gpuStats =
        calculateStatistics(
            gpuKernelTimes
        );


    // ========================================================
    // Output
    // ========================================================

    std::cout << "\n";
    std::cout
        << "========================================\n";

    std::cout
        << "CUDA Dot Product Benchmark\n";

    std::cout
        << "========================================\n\n";


    std::cout
        << "Vector size : "
        << n
        << "\n";

    std::cout
        << "Block size  : "
        << blockSize
        << "\n";

    std::cout
        << "Grid size   : "
        << gridSize
        << "\n";

    std::cout
        << "Runs        : "
        << runs
        << "\n\n";


    std::cout
        << "CPU Result : "
        << cpuResult
        << "\n";

    std::cout
        << "GPU Result : "
        << gpuResult
        << "\n";

    std::cout
        << "Correctness: "
        << (correct ? "PASS" : "FAIL")
        << "\n\n";


    std::cout
        << "CPU Average : "
        << cpuStats.average
        << " ms\n";

    std::cout
        << "CPU Median  : "
        << cpuStats.median
        << " ms\n";

    std::cout
        << "CPU Min     : "
        << cpuStats.min
        << " ms\n";

    std::cout
        << "CPU Max     : "
        << cpuStats.max
        << " ms\n\n";


    std::cout
        << "GPU Kernel Average : "
        << gpuStats.average
        << " ms\n";

    std::cout
        << "GPU Kernel Median  : "
        << gpuStats.median
        << " ms\n";

    std::cout
        << "GPU Kernel Min     : "
        << gpuStats.min
        << " ms\n";

    std::cout
        << "GPU Kernel Max     : "
        << gpuStats.max
        << " ms\n";


    // ========================================================
    // Cleanup
    // ========================================================

    CUDA_CHECK(
        cudaEventDestroy(start)
    );

    CUDA_CHECK(
        cudaEventDestroy(stop)
    );

    CUDA_CHECK(
        cudaFree(d_a)
    );

    CUDA_CHECK(
        cudaFree(d_b)
    );

    CUDA_CHECK(
        cudaFree(d_partialSums)
    );


    return 0;
}