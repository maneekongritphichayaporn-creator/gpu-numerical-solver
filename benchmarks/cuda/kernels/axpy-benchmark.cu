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
        std::cerr << "CUDA Error: "                             \
                  << cudaGetErrorString(err)                 \
                  << std::endl;                               \
        std::exit(EXIT_FAILURE);                              \
    }                                                         \
} while (0)


// ============================================================
// CPU AXPY
// y = y + alpha * x
// ============================================================

void axpyCPU(
    std::vector<double>& y,
    double alpha,
    const std::vector<double>& x
)
{
    for (int i = 0; i < y.size(); i++)
    {
        y[i] += alpha * x[i];
    }
}


// ============================================================
// GPU AXPY Kernel
// ============================================================

__global__ void axpyKernel(
    double* y,
    double alpha,
    const double* x,
    int n
)
{
    int i =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (i < n)
    {
        y[i] += alpha * x[i];
    }
}


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
    std::sort(values.begin(), values.end());

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
            (values[values.size() / 2 - 1]
             + values[values.size() / 2])
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
// Main
// ============================================================

int main()
{
    const int n = 1'000'000;

    const int runs = 10;

    const double alpha = 2.0;


    // ========================================================
    // Create input
    // ========================================================

    std::vector<double> x(
        n,
        1.0
    );

    std::vector<double> yCPU(
        n,
        3.0
    );

    std::vector<double> yGPU(
        n,
        3.0
    );


    // ========================================================
    // CPU Benchmark
    // ========================================================

    std::vector<double> cpuTimes;

    for (int run = 0; run < runs; run++)
    {
        std::vector<double> y = yCPU;

        auto start =
            std::chrono::high_resolution_clock::now();

        axpyCPU(
            y,
            alpha,
            x
        );

        auto end =
            std::chrono::high_resolution_clock::now();

        double elapsed =
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();

        cpuTimes.push_back(elapsed);
    }


    // ========================================================
    // GPU Memory Allocation
    // ========================================================

    double* d_x;
    double* d_y;

    CUDA_CHECK(
        cudaMalloc(
            &d_x,
            n * sizeof(double)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_y,
            n * sizeof(double)
        )
    );


    // ========================================================
    // Copy input → GPU
    // ========================================================

    CUDA_CHECK(
        cudaMemcpy(
            d_x,
            x.data(),
            n * sizeof(double),
            cudaMemcpyHostToDevice
        )
    );


    // ========================================================
    // CUDA configuration
    // ========================================================

    int blockSize = 256;

    int gridSize =
        (n + blockSize - 1)
        / blockSize;


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

    CUDA_CHECK(
        cudaMemcpy(
            d_y,
            yGPU.data(),
            n * sizeof(double),
            cudaMemcpyHostToDevice
        )
    );

    axpyKernel<<<gridSize, blockSize>>>(
        d_y,
        alpha,
        d_x,
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

    for (int run = 0; run < runs; run++)
    {
        // Reset y
        CUDA_CHECK(
            cudaMemcpy(
                d_y,
                yGPU.data(),
                n * sizeof(double),
                cudaMemcpyHostToDevice
            )
        );


        CUDA_CHECK(
            cudaEventRecord(start)
        );


        axpyKernel<<<gridSize, blockSize>>>(
            d_y,
            alpha,
            d_x,
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
    // Copy result back to CPU
    // ========================================================

    CUDA_CHECK(
        cudaMemcpy(
            yGPU.data(),
            d_y,
            n * sizeof(double),
            cudaMemcpyDeviceToHost
        )
    );


    // ========================================================
    // Correctness Check
    // ========================================================

    bool correct = true;

    for (int i = 0; i < n; i++)
    {
        double expected = 5.0;

        if (std::abs(yGPU[i] - expected) > 1e-12)
        {
            correct = false;
            break;
        }
    }


    // ========================================================
    // Statistics
    // ========================================================

    Statistics cpuStats =
        calculateStatistics(cpuTimes);

    Statistics gpuStats =
        calculateStatistics(gpuKernelTimes);


    // ========================================================
    // Output
    // ========================================================

    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "CUDA AXPY Benchmark\n";
    std::cout << "========================================\n";

    std::cout << "\n";

    std::cout
        << "Vector size : "
        << n
        << "\n";

    std::cout
        << "Alpha       : "
        << alpha
        << "\n";

    std::cout
        << "Block size  : "
        << blockSize
        << "\n";

    std::cout
        << "Grid size   : "
        << gridSize
        << "\n";

    std::cout << "\n";

    std::cout
        << "Runs        : "
        << runs
        << "\n";

    std::cout << "\n";

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
        << " ms\n";

    std::cout << "\n";

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

    std::cout << "\n";

    std::cout
        << "GPU Correctness : "
        << (correct ? "PASS" : "FAIL")
        << "\n";

    std::cout << "\n";


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
        cudaFree(d_x)
    );

    CUDA_CHECK(
        cudaFree(d_y)
    );

    return 0;
}