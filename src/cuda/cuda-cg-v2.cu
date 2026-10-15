
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>

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
// AXPY with scalar stored on GPU
//
// y = y + alpha * x
// ============================================================

__global__ void axpyScalarKernel(
    double* y,
    const double* alpha,
    const double* x,
    int n
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        y[i] += (*alpha) * x[i];
    }
}


// ============================================================
// Combine with scalar stored on GPU
//
// p = r + beta * p
// ============================================================

__global__ void combineScalarKernel(
    double* p,
    const double* r,
    const double* beta,
    int n
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        p[i] = r[i] + (*beta) * p[i];
    }
}


// ============================================================
// CSR SpMV
//
// y = A*x
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
    int row = blockIdx.x * blockDim.x + threadIdx.x;

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
                values[k] *
                x[colIndex[k]];
        }

        y[row] = sum;
    }
}


// ============================================================
// Dot product - first reduction
//
// partialSums[block] = sum(a[i] * b[i])
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
        blockIdx.x * blockDim.x +
        threadIdx.x;

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
// Reduction kernel
//
// input -> output
// ============================================================

__global__ void reduceKernel(
    const double* input,
    double* output,
    int n
)
{
    extern __shared__ double shared[];

    int tid = threadIdx.x;

    int i =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (i < n)
    {
        shared[tid] = input[i];
    }
    else
    {
        shared[tid] = 0.0;
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
// Scalar division
//
// result = numerator / denominator
//
// Everything stays on GPU.
// ============================================================

__global__ void scalarDivideKernel(
    const double* numerator,
    const double* denominator,
    double* result
)
{
    if (
        blockIdx.x == 0 &&
        threadIdx.x == 0
    )
    {
        *result =
            *numerator /
            *denominator;
    }
}


// ============================================================
// Dot product
//
// Final result stays on GPU.
//
// d_result contains ONE double.
// ============================================================

void dotGPU(
    const double* d_a,
    const double* d_b,
    double* d_result,
    int n,
    int blockSize
)
{
    int gridSize =
        (n + blockSize - 1)
        / blockSize;

    double* d_current = nullptr;

    CUDA_CHECK(
        cudaMalloc(
            &d_current,
            gridSize * sizeof(double)
        )
    );

    // First reduction
    dotKernel<<<
        gridSize,
        blockSize,
        blockSize * sizeof(double)
    >>>(
        d_a,
        d_b,
        d_current,
        n
    );

    CUDA_CHECK(cudaGetLastError());

    int currentSize = gridSize;

    // Continue reducing until only one value remains
    while (currentSize > 1)
    {
        int nextGridSize =
            (currentSize + blockSize - 1)
            / blockSize;

        double* d_next = nullptr;

        CUDA_CHECK(
            cudaMalloc(
                &d_next,
                nextGridSize * sizeof(double)
            )
        );

        reduceKernel<<<
            nextGridSize,
            blockSize,
            blockSize * sizeof(double)
        >>>(
            d_current,
            d_next,
            currentSize
        );

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(
            cudaFree(d_current)
        );

        d_current = d_next;

        currentSize =
            nextGridSize;
    }

    // Copy ONE scalar from final reduction buffer
    CUDA_CHECK(
        cudaMemcpy(
            d_result,
            d_current,
            sizeof(double),
            cudaMemcpyDeviceToDevice
        )
    );

    CUDA_CHECK(
        cudaFree(d_current)
    );
}


// ============================================================
// Conjugate Gradient - GPU Resident Version
// ============================================================

std::vector<double> conjugateGradientCUDA(
    const CSRMatrix& A,
    const std::vector<double>& b,
    int maxIterations,
    double tolerance
)
{
    int n = A.rows;

    const int blockSize = 256;

    int gridSize =
        (n + blockSize - 1)
        / blockSize;


    // --------------------------------------------------------
    // Host result
    // --------------------------------------------------------

    std::vector<double> x(n);


    // --------------------------------------------------------
    // Device vectors
    // --------------------------------------------------------

    double* d_x = nullptr;
    double* d_b = nullptr;
    double* d_r = nullptr;
    double* d_p = nullptr;
    double* d_Ap = nullptr;


    CUDA_CHECK(
        cudaMalloc(
            &d_x,
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


    // --------------------------------------------------------
    // Scalar values stored on GPU
    // --------------------------------------------------------

    double* d_rsold = nullptr;
    double* d_rsnew = nullptr;
    double* d_pAp   = nullptr;
    double* d_alpha = nullptr;
    double* d_beta  = nullptr;

    CUDA_CHECK(
        cudaMalloc(
            &d_rsold,
            sizeof(double)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_rsnew,
            sizeof(double)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_pAp,
            sizeof(double)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_alpha,
            sizeof(double)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_beta,
            sizeof(double)
        )
    );


    // --------------------------------------------------------
    // CSR data
    // --------------------------------------------------------

    double* d_values = nullptr;
    int* d_colIndex = nullptr;
    int* d_rowPtr = nullptr;


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


    // --------------------------------------------------------
    // Copy matrix to GPU
    // --------------------------------------------------------

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


    // --------------------------------------------------------
    // Copy b
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaMemcpy(
            d_b,
            b.data(),
            n * sizeof(double),
            cudaMemcpyHostToDevice
        )
    );


    // --------------------------------------------------------
    // x = 0
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaMemset(
            d_x,
            0,
            n * sizeof(double)
        )
    );


    // --------------------------------------------------------
    // r = b
    // p = r
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaMemcpy(
            d_r,
            d_b,
            n * sizeof(double),
            cudaMemcpyDeviceToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_p,
            d_r,
            n * sizeof(double),
            cudaMemcpyDeviceToDevice
        )
    );


    // --------------------------------------------------------
    // rsold = r^T r
    // --------------------------------------------------------

    dotGPU(
        d_r,
        d_r,
        d_rsold,
        n,
        blockSize
    );


    // --------------------------------------------------------
    // CG iteration
    // --------------------------------------------------------

    int iterations = 0;

    for (
        int iter = 0;
        iter < maxIterations;
        iter++
    )
    {
        iterations++;


        // ----------------------------------------------------
        // Ap = A*p
        // ----------------------------------------------------

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

        CUDA_CHECK(cudaGetLastError());


        // ----------------------------------------------------
        // pAp = p^T Ap
        // ----------------------------------------------------

        dotGPU(
            d_p,
            d_Ap,
            d_pAp,
            n,
            blockSize
        );


        // ----------------------------------------------------
        // alpha = rsold / pAp
        // ----------------------------------------------------

        scalarDivideKernel<<<1, 1>>>(
            d_rsold,
            d_pAp,
            d_alpha
        );

        CUDA_CHECK(cudaGetLastError());


        // ----------------------------------------------------
        // x = x + alpha*p
        // ----------------------------------------------------

        axpyScalarKernel<<<
            gridSize,
            blockSize
        >>>(
            d_x,
            d_alpha,
            d_p,
            n
        );

        CUDA_CHECK(cudaGetLastError());


        // ----------------------------------------------------
        // r = r - alpha*Ap
        // ----------------------------------------------------

        axpyScalarKernel<<<
            gridSize,
            blockSize
        >>>(
            d_r,
            d_alpha,
            d_Ap,
            n
        );

        CUDA_CHECK(cudaGetLastError());


        // ----------------------------------------------------
        // rsnew = r^T r
        // ----------------------------------------------------

        dotGPU(
            d_r,
            d_r,
            d_rsnew,
            n,
            blockSize
        );


        // ----------------------------------------------------
        // Copy only ONE scalar for convergence check
        //
        // This is temporary.
        // Full GPU-resident convergence will come later.
        // ----------------------------------------------------

        double rsnewHost = 0.0;

        CUDA_CHECK(
            cudaMemcpy(
                &rsnewHost,
                d_rsnew,
                sizeof(double),
                cudaMemcpyDeviceToHost
            )
        );


        double residual =
            std::sqrt(rsnewHost);


        if (residual < tolerance)
        {
            break;
        }


        // ----------------------------------------------------
        // beta = rsnew / rsold
        // ----------------------------------------------------

        scalarDivideKernel<<<1, 1>>>(
            d_rsnew,
            d_rsold,
            d_beta
        );

        CUDA_CHECK(cudaGetLastError());


        // ----------------------------------------------------
        // p = r + beta*p
        // ----------------------------------------------------

        combineScalarKernel<<<
            gridSize,
            blockSize
        >>>(
            d_p,
            d_r,
            d_beta,
            n
        );

        CUDA_CHECK(cudaGetLastError());


        // ----------------------------------------------------
        // rsold = rsnew
        // ----------------------------------------------------

        CUDA_CHECK(
            cudaMemcpy(
                d_rsold,
                d_rsnew,
                sizeof(double),
                cudaMemcpyDeviceToDevice
            )
        );
    }


    // --------------------------------------------------------
    // Copy final x back to CPU
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaMemcpy(
            x.data(),
            d_x,
            n * sizeof(double),
            cudaMemcpyDeviceToHost
        )
    );


    // --------------------------------------------------------
    // Free device memory
    // --------------------------------------------------------

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_Ap));

    CUDA_CHECK(cudaFree(d_rsold));
    CUDA_CHECK(cudaFree(d_rsnew));
    CUDA_CHECK(cudaFree(d_pAp));
    CUDA_CHECK(cudaFree(d_alpha));
    CUDA_CHECK(cudaFree(d_beta));

    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_colIndex));
    CUDA_CHECK(cudaFree(d_rowPtr));


    return x;
}


// ============================================================
// Main
// ============================================================

int main()
{
    const int n = 1'000'000;

    const int maxIterations = 100;

    const double tolerance = 1e-10;


    // --------------------------------------------------------
    // Create tridiagonal matrix
    //
    // A:
    //
    //  4 -1  0  0 ...
    // -1  4 -1  0 ...
    //  0 -1  4 -1 ...
    //  ...
    //
    // --------------------------------------------------------

    CSRMatrix A;

    A.rows = n;
    A.cols = n;

    A.values.reserve(
        3 * n - 2
    );

    A.colIndex.reserve(
        3 * n - 2
    );

    A.rowPtr.resize(n + 1);


    int nnz = 0;

    for (int row = 0; row < n; row++)
    {
        A.rowPtr[row] = nnz;


        // Left neighbor
        if (row > 0)
        {
            A.values.push_back(-1.0);
            A.colIndex.push_back(row - 1);
            nnz++;
        }


        // Diagonal
        A.values.push_back(4.0);
        A.colIndex.push_back(row);
        nnz++;


        // Right neighbor
        if (row < n - 1)
        {
            A.values.push_back(-1.0);
            A.colIndex.push_back(row + 1);
            nnz++;
        }
    }

    A.rowPtr[n] = nnz;


    // --------------------------------------------------------
    // xTrue = 1
    // --------------------------------------------------------

    std::vector<double> xTrue(
        n,
        1.0
    );


    // --------------------------------------------------------
    // b = A*xTrue
    // --------------------------------------------------------

    std::vector<double> b(n);

    b[0] = 3.0;

    for (int i = 1; i < n - 1; i++)
    {
        b[i] = 2.0;
    }

    b[n - 1] = 3.0;


    // --------------------------------------------------------
    // Solve
    // --------------------------------------------------------

    std::vector<double> x =
        conjugateGradientCUDA(
            A,
            b,
            maxIterations,
            tolerance
        );


    // --------------------------------------------------------
    // Calculate error
    // --------------------------------------------------------

    double maxError = 0.0;

    for (int i = 0; i < n; i++)
    {
        maxError =
            std::max(
                maxError,
                std::abs(
                    x[i] - xTrue[i]
                )
            );
    }


    // --------------------------------------------------------
    // Residual
    // --------------------------------------------------------

    double residualSquared = 0.0;

    for (int i = 0; i < n; i++)
    {
        double sum = 0.0;

        for (
            int k = A.rowPtr[i];
            k < A.rowPtr[i + 1];
            k++
        )
        {
            sum +=
                A.values[k] *
                x[A.colIndex[k]];
        }

        double r =
            b[i] - sum;

        residualSquared +=
            r * r;
    }

    double residual =
        std::sqrt(
            residualSquared
        );


    // --------------------------------------------------------
    // Output
    // --------------------------------------------------------

    std::cout
        << "================================\n";

    std::cout
        << "CUDA CG V2 - GPU Scalar\n";

    std::cout
        << "================================\n";

    std::cout
        << "Matrix size : "
        << n
        << " x "
        << n
        << "\n";

    std::cout
        << "NNZ         : "
        << nnz
        << "\n\n";

    std::cout
        << "Residual   : "
        << residual
        << "\n";

    std::cout
        << "x[0]       : "
        << x[0]
        << "\n";

    std::cout
        << "x[last]    : "
        << x[n - 1]
        << "\n";

    std::cout
        << "Max error  : "
        << maxError
        << "\n";

    std::cout
        << "Correctness : "
        << (
            maxError < 1e-8
            ? "PASS"
            : "FAIL"
        )
        << "\n";

    std::cout
        << "================================\n";


    return 0;
}
