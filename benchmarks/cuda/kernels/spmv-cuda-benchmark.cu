#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>

#include <cuda_runtime.h>


// ========================================
// CUDA Error Checking
// ========================================

#define CUDA_CHECK(call)                         \
do {                                             \
    cudaError_t err = call;                      \
    if (err != cudaSuccess) {                    \
        std::cerr << "CUDA Error: "              \
                  << cudaGetErrorString(err)     \
                  << std::endl;                 \
        std::exit(EXIT_FAILURE);                 \
    }                                            \
} while (0)


// ========================================
// CSR Matrix
// ========================================

struct CSRMatrix
{
    int rows;
    int cols;

    std::vector<double> values;
    std::vector<int> colIndex;
    std::vector<int> rowPtr;
};


// ========================================
// Create Tridiagonal Matrix
// ========================================

CSRMatrix createTridiagonalMatrix(int n)
{
    CSRMatrix A;

    A.rows = n;
    A.cols = n;

    A.rowPtr.resize(n + 1);

    for (int i = 0; i < n; i++)
    {
        if (i > 0)
        {
            A.values.push_back(-1.0);
            A.colIndex.push_back(i - 1);
        }

        A.values.push_back(4.0);
        A.colIndex.push_back(i);

        if (i < n - 1)
        {
            A.values.push_back(-1.0);
            A.colIndex.push_back(i + 1);
        }

        A.rowPtr[i + 1] =
            static_cast<int>(A.values.size());
    }

    return A;
}


// ========================================
// CPU SpMV
// ========================================

void spmvCPU(
    const CSRMatrix& A,
    const std::vector<double>& x,
    std::vector<double>& y
)
{
    for (int row = 0; row < A.rows; row++)
    {
        double sum = 0.0;

        for (
            int k = A.rowPtr[row];
            k < A.rowPtr[row + 1];
            k++
        )
        {
            sum +=
                A.values[k]
                * x[A.colIndex[k]];
        }

        y[row] = sum;
    }
}


// ========================================
// CUDA SpMV Kernel
// ========================================

__global__ void spmvKernel(
    const double* values,
    const int* colIndex,
    const int* rowPtr,
    const double* x,
    double* y,
    int rows
)
{
    int row =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (row < rows)
    {
        double sum = 0.0;

        for (
            int k = rowPtr[row];
            k < rowPtr[row + 1];
            k++
        )
        {
            sum +=
                values[k]
                * x[colIndex[k]];
        }

        y[row] = sum;
    }
}


// ========================================
// Statistics
// ========================================

struct Statistics
{
    double average;
    double median;
    double minimum;
    double maximum;
};


Statistics calculateStatistics(
    std::vector<double> times
)
{
    Statistics result{};

    std::sort(
        times.begin(),
        times.end()
    );

    result.minimum = times.front();
    result.maximum = times.back();

    result.average =
        std::accumulate(
            times.begin(),
            times.end(),
            0.0
        )
        / times.size();

    if (times.size() % 2 == 0)
    {
        int middle =
            times.size() / 2;

        result.median =
            (times[middle - 1]
             + times[middle])
            / 2.0;
    }
    else
    {
        result.median =
            times[times.size() / 2];
    }

    return result;
}


// ========================================
// Main
// ========================================

int main()
{
    // ====================================
    // Configuration
    // ====================================

    const int n = 1000000;

    const int runs = 10;

    const int blockSize = 256;


    // ====================================
    // Create Matrix
    // ====================================

    CSRMatrix A =
        createTridiagonalMatrix(n);


    // ====================================
    // Create Vector
    // ====================================

    std::vector<double> x(
        n,
        1.0
    );

    std::vector<double> yCPU(
        n,
        0.0
    );

    std::vector<double> yGPU(
        n,
        0.0
    );


    // ====================================
    // Print Matrix Information
    // ====================================

    std::cout
        << "================================"
        << std::endl;

    std::cout
        << "CUDA SpMV Benchmark"
        << std::endl;

    std::cout
        << "================================"
        << std::endl;

    std::cout
        << "Matrix size : "
        << n
        << " x "
        << n
        << std::endl;

    std::cout
        << "NNZ         : "
        << A.values.size()
        << std::endl;


    // ====================================
    // CPU Benchmark
    // ====================================

    std::vector<double> cpuTimes;

    cpuTimes.reserve(runs);


    // Warm-up
    spmvCPU(A, x, yCPU);


    for (int run = 0; run < runs; run++)
    {
        auto start =
            std::chrono::high_resolution_clock::now();


        spmvCPU(A, x, yCPU);


        auto end =
            std::chrono::high_resolution_clock::now();


        double milliseconds =
            std::chrono::duration<double, std::milli>(
                end - start
            ).count();


        cpuTimes.push_back(milliseconds);
    }


    Statistics cpuStats =
        calculateStatistics(cpuTimes);


    // ====================================
    // GPU Memory Allocation
    // ====================================

    double* d_values;
    int* d_colIndex;
    int* d_rowPtr;
    double* d_x;
    double* d_y;


    CUDA_CHECK(
        cudaMalloc(
            &d_values,
            A.values.size()
            * sizeof(double)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_colIndex,
            A.colIndex.size()
            * sizeof(int)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_rowPtr,
            A.rowPtr.size()
            * sizeof(int)
        )
    );

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


    // ====================================
    // Copy Matrix + Vector to GPU
    // ====================================

    CUDA_CHECK(
        cudaMemcpy(
            d_values,
            A.values.data(),
            A.values.size()
            * sizeof(double),
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_colIndex,
            A.colIndex.data(),
            A.colIndex.size()
            * sizeof(int),
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_rowPtr,
            A.rowPtr.data(),
            A.rowPtr.size()
            * sizeof(int),
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_x,
            x.data(),
            n * sizeof(double),
            cudaMemcpyHostToDevice
        )
    );


    // ====================================
    // Grid Configuration
    // ====================================

    int gridSize =
        (n + blockSize - 1)
        / blockSize;


    // ====================================
    // CUDA Events
    // ====================================

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));


    // ====================================
    // GPU Warm-up
    // ====================================

    spmvKernel<<<
        gridSize,
        blockSize
    >>>(
        d_values,
        d_colIndex,
        d_rowPtr,
        d_x,
        d_y,
        n
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


    // ====================================
    // GPU Kernel Benchmark
    // ====================================

    std::vector<double> gpuKernelTimes;

    gpuKernelTimes.reserve(runs);


    for (int run = 0; run < runs; run++)
    {
        CUDA_CHECK(
            cudaEventRecord(start)
        );


        spmvKernel<<<
            gridSize,
            blockSize
        >>>(
            d_values,
            d_colIndex,
            d_rowPtr,
            d_x,
            d_y,
            n
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


    Statistics gpuKernelStats =
        calculateStatistics(
            gpuKernelTimes
        );


    // ====================================
    // Copy GPU Result to CPU
    // ====================================

    CUDA_CHECK(
        cudaMemcpy(
            yGPU.data(),
            d_y,
            n * sizeof(double),
            cudaMemcpyDeviceToHost
        )
    );


    // ====================================
    // Correctness Check
    // ====================================

    bool correct = true;

    for (int i = 0; i < n; i++)
    {
        if (
            std::abs(
                yCPU[i] - yGPU[i]
            ) > 1e-10
        )
        {
            correct = false;
            break;
        }
    }


    // ====================================
    // Print Results
    // ====================================

    std::cout
        << std::endl
        << "CPU"
        << std::endl;

    std::cout
        << "Average : "
        << cpuStats.average
        << " ms"
        << std::endl;

    std::cout
        << "Median  : "
        << cpuStats.median
        << " ms"
        << std::endl;

    std::cout
        << "Min     : "
        << cpuStats.minimum
        << " ms"
        << std::endl;

    std::cout
        << "Max     : "
        << cpuStats.maximum
        << " ms"
        << std::endl;


    std::cout
        << std::endl
        << "GPU Kernel"
        << std::endl;

    std::cout
        << "Average : "
        << gpuKernelStats.average
        << " ms"
        << std::endl;

    std::cout
        << "Median  : "
        << gpuKernelStats.median
        << " ms"
        << std::endl;

    std::cout
        << "Min     : "
        << gpuKernelStats.minimum
        << " ms"
        << std::endl;

    std::cout
        << "Max     : "
        << gpuKernelStats.maximum
        << " ms"
        << std::endl;


    std::cout
        << std::endl
        << "Correctness : "
        << (correct ? "PASS" : "FAIL")
        << std::endl;


    // ====================================
    // Cleanup
    // ====================================

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_colIndex));
    CUDA_CHECK(cudaFree(d_rowPtr));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));


    return 0;
}