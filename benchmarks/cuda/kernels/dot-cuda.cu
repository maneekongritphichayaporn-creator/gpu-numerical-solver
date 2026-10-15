#include <iostream>
#include <vector>
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
// Dot Product Kernel
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
    // Step 1: Multiply
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


    // --------------------------------------------------------
    // Make sure all threads finished loading
    // --------------------------------------------------------

    __syncthreads();


    // --------------------------------------------------------
    // Step 2: Reduction
    // --------------------------------------------------------

    for (int stride = blockDim.x / 2;
         stride > 0;
         stride /= 2)
    {
        if (tid < stride)
        {
            shared[tid] +=
                shared[tid + stride];
        }

        __syncthreads();
    }


    // --------------------------------------------------------
    // One thread writes the block result
    // --------------------------------------------------------

    if (tid == 0)
    {
        partialSums[blockIdx.x] =
            shared[0];
    }
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
// Main
// ============================================================

int main()
{
    int n = 8;

    std::vector<double> a =
    {
        1, 2, 3, 4,
        5, 6, 7, 8
    };

    std::vector<double> b =
    {
        8, 7, 6, 5,
        4, 3, 2, 1
    };


    // ========================================================
    // CPU
    // ========================================================

    double cpuResult =
        dotCPU(a, b);


    // ========================================================
    // GPU configuration
    // ========================================================

    int blockSize = 8;

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
    // Copy CPU → GPU
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
    // Launch Kernel
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
    // Copy partial sums back
    // ========================================================

    std::vector<double> partialSums(
        gridSize
    );

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

    double gpuResult = 0.0;

    for (double value : partialSums)
    {
        gpuResult += value;
    }


    // ========================================================
    // Output
    // ========================================================

    std::cout
        << "CPU result: "
        << cpuResult
        << "\n";

    std::cout
        << "GPU result: "
        << gpuResult
        << "\n";


    if (std::abs(cpuResult - gpuResult) < 1e-12)
    {
        std::cout
            << "Correctness: PASS\n";
    }
    else
    {
        std::cout
            << "Correctness: FAIL\n";
    }


    // ========================================================
    // Cleanup
    // ========================================================

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_partialSums));


    return 0;
}