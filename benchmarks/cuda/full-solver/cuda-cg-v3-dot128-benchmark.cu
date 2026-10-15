
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <iomanip>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        std::cerr << "CUDA Error: "                            \
                  << cudaGetErrorString(err) << std::endl;    \
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
// AXPY
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
// Negative AXPY
// y = y - alpha * x
// ============================================================

__global__ void axpyNegativeScalarKernel(
    double* y,
    const double* alpha,
    const double* x,
    int n
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        y[i] -= (*alpha) * x[i];
    }
}


// ============================================================
// Combine
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
// First-stage Dot Product
//
// partialSums[blockIdx.x] = block reduction
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
// Generic Reduction
//
// input[n] → output[numberOfBlocks]
//
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
        shared[tid] =
            input[i];
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
// ============================================================

__global__ void scalarDivideKernel(
    const double* numerator,
    const double* denominator,
    double* result
)
{
    if (threadIdx.x == 0 &&
        blockIdx.x == 0)
    {
        *result =
            (*numerator) /
            (*denominator);
    }
}


// ============================================================
// dotGPU V3
//
// IMPORTANT:
// No cudaMalloc / cudaFree here.
//
// Buffers are allocated once outside
// and reused for every dot operation.
// ============================================================

void dotGPUReuse(
    const double* d_a,
    const double* d_b,
    double* d_result,
    double* d_bufferA,
    double* d_bufferB,
    int n,
    int blockSize
)
{
    int gridSize =
        (n + blockSize - 1) /
        blockSize;


    // --------------------------------------------------------
    // Stage 1
    // a[i] * b[i]
    // --------------------------------------------------------

    dotKernel<<<
        gridSize,
        blockSize,
        blockSize * sizeof(double)
    >>>(
        d_a,
        d_b,
        d_bufferA,
        n
    );

    CUDA_CHECK(cudaGetLastError());


    int currentSize = gridSize;


    // --------------------------------------------------------
    // Stage 2+
    // Reduce partial sums
    // --------------------------------------------------------

    double* current = d_bufferA;
    double* next = d_bufferB;


    while (currentSize > 1)
    {
        int nextSize =
            (currentSize + blockSize - 1) /
            blockSize;


        reduceKernel<<<
            nextSize,
            blockSize,
            blockSize * sizeof(double)
        >>>(
            current,
            next,
            currentSize
        );

        CUDA_CHECK(cudaGetLastError());


        currentSize = nextSize;


        std::swap(
            current,
            next
        );
    }


    // --------------------------------------------------------
    // Copy final scalar
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaMemcpy(
            d_result,
            current,
            sizeof(double),
            cudaMemcpyDeviceToDevice
        )
    );
}


// ============================================================
// CUDA Conjugate Gradient V3
//
// Fixed iteration count for performance experiment.
// ============================================================

void conjugateGradientCUDA(
    const CSRMatrix& A,
    const std::vector<double>& b,
    std::vector<double>& x,
    int maxIterations,
    int blockSize,
    int dotBlockSize
)
{
    int n = A.rows;

    int gridSize =
        (n + blockSize - 1) /
        blockSize;

    // Dot Product uses its own block size; other CG kernels keep blockSize.
    int dotGridSize =
        (n + dotBlockSize - 1) / dotBlockSize;


    // ========================================================
    // Device vectors
    // ========================================================

    double* d_x;
    double* d_b;
    double* d_r;
    double* d_p;
    double* d_Ap;

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


    // ========================================================
    // CSR arrays
    // ========================================================

    double* d_values;
    int* d_colIndex;
    int* d_rowPtr;

    CUDA_CHECK(
        cudaMalloc(
            &d_values,
            A.values.size() * sizeof(double)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_colIndex,
            A.colIndex.size() * sizeof(int)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_rowPtr,
            A.rowPtr.size() * sizeof(int)
        )
    );


    CUDA_CHECK(
        cudaMemcpy(
            d_values,
            A.values.data(),
            A.values.size() * sizeof(double),
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_colIndex,
            A.colIndex.data(),
            A.colIndex.size() * sizeof(int),
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_rowPtr,
            A.rowPtr.data(),
            A.rowPtr.size() * sizeof(int),
            cudaMemcpyHostToDevice
        )
    );


    // ========================================================
    // Copy b
    // ========================================================

    CUDA_CHECK(
        cudaMemcpy(
            d_b,
            b.data(),
            n * sizeof(double),
            cudaMemcpyHostToDevice
        )
    );


    // ========================================================
    // x0 = 0
    // ========================================================

    CUDA_CHECK(
        cudaMemset(
            d_x,
            0,
            n * sizeof(double)
        )
    );


    // ========================================================
    // r0 = b
    // p0 = r0
    // ========================================================

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


    // ========================================================
    // Scalar buffers
    // ========================================================

    double* d_rsold;
    double* d_rsnew;
    double* d_pAp;
    double* d_alpha;
    double* d_beta;

    CUDA_CHECK(cudaMalloc(&d_rsold, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rsnew, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pAp, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_alpha, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_beta, sizeof(double)));


    // ========================================================
    // Reduction buffers
    //
    // Allocate ONCE.
    //
    // Largest reduction size = dotGridSize
    // ========================================================

    double* d_reduceA;
    double* d_reduceB;

    CUDA_CHECK(
        cudaMalloc(
            &d_reduceA,
            dotGridSize * sizeof(double)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_reduceB,
            dotGridSize * sizeof(double)
        )
    );


    // ========================================================
    // Initial:
    //
    // rsold = r^T r
    // ========================================================

    dotGPUReuse(
        d_r,
        d_r,
        d_rsold,
        d_reduceA,
        d_reduceB,
        n,
        dotBlockSize
    );


    // ========================================================
    // CG iterations
    // ========================================================

    for (int iter = 0;
         iter < maxIterations;
         iter++)
    {
        // ----------------------------------------------------
        // Ap = A * p
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

        dotGPUReuse(
            d_p,
            d_Ap,
            d_pAp,
            d_reduceA,
            d_reduceB,
            n,
            dotBlockSize
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

        axpyNegativeScalarKernel<<<
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

        dotGPUReuse(
            d_r,
            d_r,
            d_rsnew,
            d_reduceA,
            d_reduceB,
            n,
            dotBlockSize
        );


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


    // ========================================================
    // Copy final x
    // ========================================================

    CUDA_CHECK(
        cudaMemcpy(
            x.data(),
            d_x,
            n * sizeof(double),
            cudaMemcpyDeviceToHost
        )
    );


    // ========================================================
    // Free
    // ========================================================

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_Ap));

    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_colIndex));
    CUDA_CHECK(cudaFree(d_rowPtr));

    CUDA_CHECK(cudaFree(d_rsold));
    CUDA_CHECK(cudaFree(d_rsnew));
    CUDA_CHECK(cudaFree(d_pAp));
    CUDA_CHECK(cudaFree(d_alpha));
    CUDA_CHECK(cudaFree(d_beta));

    CUDA_CHECK(cudaFree(d_reduceA));
    CUDA_CHECK(cudaFree(d_reduceB));
}


// ============================================================
// Main
// ============================================================

int main()
{
    const int n = 1'000'000;

    const int maxIterations = 20;

    const int blockSize = 256;     // SpMV, AXPY and Combine remain unchanged
    const int dotBlockSize = 128;  // Experiment: optimize Dot Product only


    // ========================================================
    // Build tridiagonal matrix
    //
    //      4  -1
    //     -1   4  -1
    //          ...
    //             -1  4
    //
    // ========================================================

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
            static_cast<int>(
                A.values.size()
            );
    }


    // ========================================================
    // True solution
    // x = [1,1,1,...]
    // ========================================================

    std::vector<double> xTrue(
        n,
        1.0
    );


    // ========================================================
    // b = A*xTrue
    //
    // [3, 2, 2, ..., 2, 3]
    // ========================================================

    std::vector<double> b(n);

    b[0] = 3.0;

    for (int i = 1; i < n - 1; i++)
    {
        b[i] = 2.0;
    }

    b[n - 1] = 3.0;


    // ========================================================
    // Initial x
    // ========================================================

    std::vector<double> x(
        n,
        0.0
    );


    // ========================================================
    // Benchmark: warm-up + 10 measured full CG runs
    // Uses wall-clock timing with device synchronization.
    // ========================================================

    const int warmupRuns = 1;
    const int benchmarkRuns = 10;
    std::vector<double> runTimesMs;
    runTimesMs.reserve(benchmarkRuns);

    for (int i = 0; i < warmupRuns; ++i)
    {
        conjugateGradientCUDA(A, b, x, maxIterations, blockSize, dotBlockSize);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (int i = 0; i < benchmarkRuns; ++i)
    {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::steady_clock::now();
        conjugateGradientCUDA(A, b, x, maxIterations, blockSize, dotBlockSize);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto stop = std::chrono::steady_clock::now();
        double elapsedMs = std::chrono::duration<double, std::milli>(stop - start).count();
        runTimesMs.push_back(elapsedMs);
    }

    double sumMs = 0.0;
    for (double t : runTimesMs) sumMs += t;
    double averageMs = sumMs / benchmarkRuns;
    std::sort(runTimesMs.begin(), runTimesMs.end());
    double medianMs = (runTimesMs[4] + runTimesMs[5]) / 2.0;


    // ========================================================
    // Check solution
    // ========================================================

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


    // ========================================================
    // Calculate residual on CPU
    //
    // r = b - A*x
    // ========================================================

    double residualSquared = 0.0;


    for (int row = 0; row < n; row++)
    {
        double Ax = 0.0;

        for (
            int k = A.rowPtr[row];
            k < A.rowPtr[row + 1];
            k++
        )
        {
            Ax +=
                A.values[k] *
                x[A.colIndex[k]];
        }

        double r =
            b[row] - Ax;

        residualSquared +=
            r * r;
    }


    double residual =
        std::sqrt(
            residualSquared
        );


    // ========================================================
    // Output
    // ========================================================

    std::cout
        << "================================\n";

    std::cout
        << "CUDA CG V3 - Reused Reduction Buffers\n";

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
        << A.values.size()
        << "\n";

    std::cout
        << "Iterations  : "
        << maxIterations
        << "\n";

    std::cout
        << "Other kernel block size : " << blockSize << "\n";
    std::cout
        << "Dot block size          : " << dotBlockSize << "\n";
    std::cout
        << "Benchmark runs          : " << benchmarkRuns << " (after " << warmupRuns << " warm-up)\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "CG average time         : " << averageMs << " ms\n";
    std::cout << "CG median time          : " << medianMs << " ms\n";
    std::cout << "CG min time             : " << runTimesMs.front() << " ms\n";
    std::cout << "CG max time             : " << runTimesMs.back() << " ms\n\n";

    std::cout
        << "Residual    : "
        << residual
        << "\n";

    std::cout
        << "x[0]        : "
        << x[0]
        << "\n";

    std::cout
        << "x[last]     : "
        << x[n - 1]
        << "\n";

    std::cout
        << "Max error   : "
        << maxError
        << "\n";


    if (maxError < 1e-8 &&
        residual < 1e-8)
    {
        std::cout
            << "Correctness : PASS\n";
    }
    else
    {
        std::cout
            << "Correctness : FAIL\n";
    }


    std::cout
        << "================================\n";


    return 0;
}
