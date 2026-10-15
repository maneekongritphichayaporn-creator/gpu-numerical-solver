
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <numeric>

#include <cuda_runtime.h>


// ============================================================
// CUDA Error Checking
// ============================================================

#define CUDA_CHECK(call)                                       \
do {                                                           \
    cudaError_t err = call;                                    \
    if (err != cudaSuccess) {                                  \
        std::cerr << "CUDA Error: "                            \
                  << cudaGetErrorString(err)                   \
                  << " at " << __FILE__ << ":" << __LINE__     \
                  << std::endl;                                \
        std::exit(EXIT_FAILURE);                               \
    }                                                          \
} while (0)


// ============================================================
// CSR Matrix
// ============================================================

struct CSRMatrix
{
    int rows;
    int cols;

    std::vector<double> values;
    std::vector<int> colIndex;
    std::vector<int> rowPtr;
};


// ============================================================
// Benchmark Statistics
// ============================================================

struct Statistics
{
    double average;
    double median;
    double minimum;
    double maximum;
};


// ============================================================
// Calculate Statistics
// ============================================================

Statistics calculateStatistics(
    std::vector<double> times
)
{
    Statistics result{};

    std::sort(
        times.begin(),
        times.end()
    );

    result.minimum =
        times.front();

    result.maximum =
        times.back();

    result.average =
        std::accumulate(
            times.begin(),
            times.end(),
            0.0
        ) / times.size();

    if (times.size() % 2 == 0)
    {
        int middle =
            static_cast<int>(times.size() / 2);

        result.median =
            (
                times[middle - 1]
                + times[middle]
            ) / 2.0;
    }
    else
    {
        result.median =
            times[times.size() / 2];
    }

    return result;
}


// ============================================================
// Create Tridiagonal Matrix
//
// A =
// [ 4 -1  0  0 ...]
// [-1  4 -1  0 ...]
// [ 0 -1  4 -1 ...]
// [...            ]
// ============================================================

CSRMatrix createTridiagonalMatrix(int n)
{
    CSRMatrix A;

    A.rows = n;
    A.cols = n;

    A.rowPtr.resize(n + 1);

    for (int i = 0; i < n; i++)
    {
        // Left element
        if (i > 0)
        {
            A.values.push_back(-1.0);
            A.colIndex.push_back(i - 1);
        }

        // Diagonal
        A.values.push_back(4.0);
        A.colIndex.push_back(i);

        // Right element
        if (i < n - 1)
        {
            A.values.push_back(-1.0);
            A.colIndex.push_back(i + 1);
        }

        A.rowPtr[i + 1] =
            static_cast<int>(
                A.values.size()
            );
    }

    return A;
}


// ============================================================
// CUDA AXPY Kernel
//
// y = y + alpha * x
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
        y[i] +=
            alpha * x[i];
    }
}


// ============================================================
// CUDA Combine Kernel
//
// p = r + beta * p
// ============================================================

__global__ void combineKernel(
    double* p,
    const double* r,
    double beta,
    int n
)
{
    int i =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (i < n)
    {
        p[i] =
            r[i]
            + beta * p[i];
    }
}


// ============================================================
// CUDA CSR SpMV Kernel
//
// y = A * x
// ============================================================

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


// ============================================================
// CUDA Dot Product Kernel
//
// Computes partial dot products:
// result = sum(a[i] * b[i])
// ============================================================

__global__ void dotKernel(
    const double* a,
    const double* b,
    double* partialSums,
    int n
)
{
    extern __shared__ double shared[];

    int tid =
        threadIdx.x;

    int i =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (i < n)
    {
        shared[tid] =
            a[i] * b[i];
    }
    else
    {
        shared[tid] =
            0.0;
    }

    __syncthreads();

    // Reduction inside block
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

    if (tid == 0)
    {
        partialSums[blockIdx.x] =
            shared[0];
    }
}


// ============================================================
// 1. CUDA Reduction Kernel
//
// output[blockIdx.x] = sum(input[block elements])
// ============================================================

__global__ void reduceKernel(
    const double* input,
    double* output,
    int n
)
{
    extern __shared__ double shared[];

    int tid =
        threadIdx.x;

    int i =
        blockIdx.x * blockDim.x
        + threadIdx.x;

    if (i < n)
    {
        shared[tid] =
            input[i];
    }
    else
    {
        shared[tid] =
            0.0;
    }

    __syncthreads();

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

    if (tid == 0)
    {
        output[blockIdx.x] =
            shared[0];
    }
}


// ============================================================
// GPU Dot Product (Baseline)
//
// GPU: multiply + block reduction
// CPU: final reduction of partial sums
// ============================================================

double dotGPU(
    const double* d_a,
    const double* d_b,
    double* d_partialSums,
    int n,
    int blockSize
)
{
    int gridSize =
        (n + blockSize - 1)
        / blockSize;

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

    double result = 0.0;

    for (double value : partialSums)
    {
        result += value;
    }

    return result;
}


// ============================================================
// 2. GPU Dot Product (v2 - Multi-pass GPU Reduction)
//
// GPU:
//      1. Multiply + block reduction (dotKernel)
//      2. Tree reduction จนเหลือ 1 scalar บน GPU (reduceKernel)
// CPU:
//      Copy เพียง scalar เดี่ยว (1 double) กลับมา
// ============================================================

double dotGPU_v2(
    const double* d_a,
    const double* d_b,
    int n,
    int blockSize
)
{
    int gridSize =
        (n + blockSize - 1) / blockSize;

    // Buffer สำหรับผลลัพธ์แต่ละ block
    double* d_partialSums = nullptr;

    CUDA_CHECK(cudaMalloc(
        &d_partialSums,
        gridSize * sizeof(double)
    ));

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

    CUDA_CHECK(cudaGetLastError());

    // ลด partial sums บน GPU
    int currentSize = gridSize;

    while (currentSize > 1)
    {
        int reduceGridSize =
            (currentSize + blockSize - 1)
            / blockSize;

        double* d_next = nullptr;

        CUDA_CHECK(cudaMalloc(
            &d_next,
            reduceGridSize * sizeof(double)
        ));

        reduceKernel<<<
            reduceGridSize,
            blockSize,
            blockSize * sizeof(double)
        >>>(
            d_partialSums,
            d_next,
            currentSize
        );

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaFree(d_partialSums));

        d_partialSums = d_next;
        currentSize = reduceGridSize;
    }

    // ตอนนี้เหลือ scalar เดียวบน GPU
    double result = 0.0;

    CUDA_CHECK(cudaMemcpy(
        &result,
        d_partialSums,
        sizeof(double),
        cudaMemcpyDeviceToHost
    ));

    CUDA_CHECK(cudaFree(d_partialSums));

    return result;
}


// ============================================================
// CUDA Conjugate Gradient
// ============================================================

void conjugateGradientCUDA(
    const CSRMatrix& A,
    const std::vector<double>& b,
    std::vector<double>& x,
    int maxIterations,
    double tolerance
)
{
    const int n =
        A.rows;

    const int blockSize =
        256;

    const int gridSize =
        (n + blockSize - 1)
        / blockSize;

    // Device pointers
    double* d_values = nullptr;
    int* d_colIndex = nullptr;
    int* d_rowPtr = nullptr;

    double* d_b = nullptr;
    double* d_x = nullptr;
    double* d_r = nullptr;
    double* d_p = nullptr;
    double* d_Ap = nullptr;

    double* d_partialSums = nullptr;

    // Allocate matrix memory
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

    // Allocate vector memory
    CUDA_CHECK(
        cudaMalloc(
            &d_b,
            n * sizeof(double)
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
            &d_r,
            n * sizeof(double)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_p,
            n * sizeof(double)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_Ap,
            n * sizeof(double)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_partialSums,
            gridSize * sizeof(double)
        )
    );

    // Copy matrix CPU -> GPU
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

    // Copy b CPU -> GPU
    CUDA_CHECK(
        cudaMemcpy(
            d_b,
            b.data(),
            n * sizeof(double),
            cudaMemcpyHostToDevice
        )
    );

    // Initial x0 = 0
    CUDA_CHECK(
        cudaMemset(
            d_x,
            0,
            n * sizeof(double)
        )
    );

    // Initial r0 = b - A*x0 = b
    CUDA_CHECK(
        cudaMemcpy(
            d_r,
            d_b,
            n * sizeof(double),
            cudaMemcpyDeviceToDevice
        )
    );

    // Initial p0 = r0
    CUDA_CHECK(
        cudaMemcpy(
            d_p,
            d_r,
            n * sizeof(double),
            cudaMemcpyDeviceToDevice
        )
    );

    // Initial r^T r
    // หมายเหตุ: สลับเรียกใช้ dotGPU_v2(d_r, d_r, n, blockSize) ตรงนี้ได้เพื่อเทียบผล
    double rsold =
        dotGPU(
            d_r,
            d_r,
            d_partialSums,
            n,
            blockSize
        );

    int iterations = 0;

    double residual =
        std::sqrt(rsold);

    // Conjugate Gradient Loop
    for (
        int iter = 0;
        iter < maxIterations;
        iter++
    )
    {
        // Ap = A * p
        spmvKernel<<<
            gridSize,
            blockSize
        >>>(
            d_values,
            d_colIndex,
            d_rowPtr,
            d_p,
            d_Ap,
            n
        );

        CUDA_CHECK(
            cudaGetLastError()
        );

        CUDA_CHECK(
            cudaDeviceSynchronize()
        );

        // pAp = p^T Ap
        double pAp =
            dotGPU(
                d_p,
                d_Ap,
                d_partialSums,
                n,
                blockSize
            );

        // alpha = (r^T r) / (p^T Ap)
        double alpha =
            rsold / pAp;

        // x = x + alpha*p
        axpyKernel<<<
            gridSize,
            blockSize
        >>>(
            d_x,
            alpha,
            d_p,
            n
        );

        CUDA_CHECK(
            cudaGetLastError()
        );

        CUDA_CHECK(
            cudaDeviceSynchronize()
        );

        // r = r - alpha*Ap
        axpyKernel<<<
            gridSize,
            blockSize
        >>>(
            d_r,
            -alpha,
            d_Ap,
            n
        );

        CUDA_CHECK(
            cudaGetLastError()
        );

        CUDA_CHECK(
            cudaDeviceSynchronize()
        );

        // rsnew = r^T r
        double rsnew =
            dotGPU(
                d_r,
                d_r,
                d_partialSums,
                n,
                blockSize
            );

        residual =
            std::sqrt(rsnew);

        iterations =
            iter + 1;

        if (residual < tolerance)
        {
            break;
        }

        // beta = rsnew / rsold
        double beta =
            rsnew / rsold;

        // p = r + beta*p
        combineKernel<<<
            gridSize,
            blockSize
        >>>(
            d_p,
            d_r,
            beta,
            n
        );

        CUDA_CHECK(
            cudaGetLastError()
        );

        CUDA_CHECK(
            cudaDeviceSynchronize()
        );

        rsold =
            rsnew;
    }

    // Copy final x GPU -> CPU
    x.resize(n);

    CUDA_CHECK(
        cudaMemcpy(
            x.data(),
            d_x,
            n * sizeof(double),
            cudaMemcpyDeviceToHost
        )
    );

    std::cout
        << "Iterations : "
        << iterations
        << std::endl;

    std::cout
        << "Residual   : "
        << residual
        << std::endl;

    // Free device memory
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_colIndex));
    CUDA_CHECK(cudaFree(d_rowPtr));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_Ap));
    CUDA_CHECK(cudaFree(d_partialSums));
}


// ============================================================
// Main
// ============================================================

int main()
{
    const int n =
        1000000;

    const int maxIterations =
        100;

    const double tolerance =
        1e-10;

    CSRMatrix A =
        createTridiagonalMatrix(n);

    std::vector<double> xTrue(
        n,
        1.0
    );

    std::vector<double> b(n);

    b[0] = 3.0;

    for (int i = 1; i < n - 1; i++)
    {
        b[i] = 2.0;
    }

    b[n - 1] = 3.0;

    std::cout
        << "================================"
        << std::endl;

    std::cout
        << "CUDA Conjugate Gradient"
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

    std::cout
        << std::endl;

    std::vector<double> x;

    conjugateGradientCUDA(
        A,
        b,
        x,
        maxIterations,
        tolerance
    );

    double maxError = 0.0;

    for (int i = 0; i < n; i++)
    {
        double error =
            std::abs(
                x[i]
                - xTrue[i]
            );

        if (error > maxError)
        {
            maxError = error;
        }
    }

    std::cout
        << std::endl;

    std::cout
        << "x[0]       : "
        << x[0]
        << std::endl;

    std::cout
        << "x[last]    : "
        << x[n - 1]
        << std::endl;

    std::cout
        << "Max error  : "
        << maxError
        << std::endl;

    if (maxError < 1e-8)
    {
        std::cout
            << "Correctness : PASS"
            << std::endl;
    }
    else
    {
        std::cout
            << "Correctness : FAIL"
            << std::endl;
    }

    std::cout
        << "================================"
        << std::endl;

    return 0;
}
