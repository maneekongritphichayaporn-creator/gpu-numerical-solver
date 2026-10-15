
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <chrono>

#include <cuda_runtime.h>


// ============================================================
// CUDA Error Checking
// ============================================================

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
// Timing Breakdown
// ============================================================

struct TimingBreakdown
{
double spmv = 0.0;

double dotKernel = 0.0;
double dotD2H = 0.0;
double dotCPU = 0.0;

double axpy = 0.0;
double combine = 0.0;

double fullCG = 0.0;
};


// ============================================================
// Create Tridiagonal Matrix
// ============================================================

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


// ============================================================
// AXPY Kernel
//
// y = y + alpha*x
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
// Combine Kernel
//
// p = r + beta*p
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
        r[i] + beta * p[i];
}
}


// ============================================================
// CSR SpMV Kernel
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
// Dot Kernel
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

__global__ void reduceKernel(
    const double* input,
    double* output,
    int n
)
{
    extern __shared__ double shared[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
        shared[tid] = input[i];
    else
        shared[tid] = 0.0;

    __syncthreads();

    for (int stride = blockDim.x / 2;
         stride > 0;
         stride /= 2)
    {
        if (tid < stride)
            shared[tid] += shared[tid + stride];

        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = shared[0];
}


// ============================================================
// Dot Result
// ============================================================

struct DotTiming
{
double result;

double kernel;
double d2h;
double cpu;
};


// ============================================================
// Profiled Dot Product
// ============================================================

DotTiming dotGPUProfiled(
const double* d_a,
const double* d_b,
double* d_partialSums,
int n,
int blockSize
)
{
DotTiming timing{};

int gridSize =
    (n + blockSize - 1)
    / blockSize;


// --------------------------------------------------------
// CUDA Events
// --------------------------------------------------------

cudaEvent_t startKernel;
cudaEvent_t stopKernel;

cudaEvent_t startD2H;
cudaEvent_t stopD2H;

CUDA_CHECK(
    cudaEventCreate(&startKernel)
);

CUDA_CHECK(
    cudaEventCreate(&stopKernel)
);

CUDA_CHECK(
    cudaEventCreate(&startD2H)
);

CUDA_CHECK(
    cudaEventCreate(&stopD2H)
);


// --------------------------------------------------------
// Dot kernel
// --------------------------------------------------------


float kernelMilliseconds = 0.0f;

CUDA_CHECK(
    cudaEventRecord(startKernel)
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
    cudaDeviceSynchronize()
);

CUDA_CHECK(
    cudaEventRecord(stopKernel)
);

CUDA_CHECK(
    cudaEventSynchronize(stopKernel)
);

CUDA_CHECK(
    cudaEventElapsedTime(
        &kernelMilliseconds,
        startKernel,
        stopKernel
    )
);

timing.kernel =
    static_cast<double>(kernelMilliseconds);

// --------------------------------------------------------
// GPU → CPU
// --------------------------------------------------------

std::vector<double> partialSums(
    gridSize
);

CUDA_CHECK(
    cudaEventRecord(startD2H)
);

CUDA_CHECK(
    cudaMemcpy(
        partialSums.data(),
        d_partialSums,
        gridSize * sizeof(double),
        cudaMemcpyDeviceToHost
    )
);

CUDA_CHECK(
    cudaEventRecord(stopD2H)
);

CUDA_CHECK(
    cudaEventSynchronize(stopD2H)
);


float d2hMilliseconds = 0.0f;

CUDA_CHECK(
cudaEventElapsedTime(
    &d2hMilliseconds,
    startD2H,
    stopD2H
)
);

timing.d2h = d2hMilliseconds;


// --------------------------------------------------------
// CPU final reduction
// --------------------------------------------------------

auto cpuStart =
    std::chrono::high_resolution_clock::now();

double result = 0.0;

for (double value : partialSums)
{
    result += value;
}

auto cpuStop =
    std::chrono::high_resolution_clock::now();


timing.result = result;

timing.cpu =
    std::chrono::duration<double, std::milli>(
        cpuStop - cpuStart
    ).count();


// --------------------------------------------------------
// Cleanup events
// --------------------------------------------------------

CUDA_CHECK(
    cudaEventDestroy(startKernel)
);

CUDA_CHECK(
    cudaEventDestroy(stopKernel)
);

CUDA_CHECK(
    cudaEventDestroy(startD2H)
);

CUDA_CHECK(
    cudaEventDestroy(stopD2H)
);


return timing;
}


// ============================================================
// Main
// ============================================================

int main()
{
const int n =
    1'000'000;

const int maxIterations =
    100;

const double tolerance =
    1e-10;

const int blockSize =
    256;

const int gridSize =
    (n + blockSize - 1)
    / blockSize;


// ========================================================
// Create Matrix
// ========================================================

CSRMatrix A =
    createTridiagonalMatrix(n);


std::vector<double> xTrue(
    n,
    1.0
);


std::vector<double> b(n);

b[0] = 3.0;

for (
    int i = 1;
    i < n - 1;
    i++
)
{
    b[i] = 2.0;
}

b[n - 1] = 3.0;


// ========================================================
// Device Pointers
// ========================================================

double* d_values = nullptr;
int* d_colIndex = nullptr;
int* d_rowPtr = nullptr;

double* d_b = nullptr;
double* d_x = nullptr;
double* d_r = nullptr;
double* d_p = nullptr;
double* d_Ap = nullptr;

double* d_partialSums = nullptr;


// ========================================================
// Allocate
// ========================================================

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


// ========================================================
// Copy Data
// ========================================================

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
        d_b,
        b.data(),
        n * sizeof(double),
        cudaMemcpyHostToDevice
    )
);


CUDA_CHECK(
    cudaMemset(
        d_x,
        0,
        n * sizeof(double)
    )
);


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
// Timing
// ========================================================

TimingBreakdown timing;


cudaEvent_t fullStart;
cudaEvent_t fullStop;

CUDA_CHECK(
    cudaEventCreate(&fullStart)
);

CUDA_CHECK(
    cudaEventCreate(&fullStop)
);


// ========================================================
// Start Full CG Timer
// ========================================================

CUDA_CHECK(
    cudaEventRecord(fullStart)
);


// ========================================================
// Initial Dot
// ========================================================

DotTiming initialDot =
    dotGPUProfiled(
        d_r,
        d_r,
        d_partialSums,
        n,
        blockSize
    );


timing.dotKernel +=
    initialDot.kernel;

timing.dotD2H +=
    initialDot.d2h;

timing.dotCPU +=
    initialDot.cpu;


double rsold =
    initialDot.result;


double residual =
    std::sqrt(rsold);


int iterations = 0;

auto fullStartCPU =
    std::chrono::steady_clock::now();


// ========================================================
// CG Loop
// ========================================================

for (
    int iter = 0;
    iter < maxIterations;
    iter++
)
{
    // ----------------------------------------------------
    // SpMV
    // ----------------------------------------------------

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

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

    timing.spmv +=
        milliseconds;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));


    // ----------------------------------------------------
    // pAp = p^T Ap
    // ----------------------------------------------------

    DotTiming pAp =
        dotGPUProfiled(
            d_p,
            d_Ap,
            d_partialSums,
            n,
            blockSize
        );

    timing.dotKernel +=
        pAp.kernel;

    timing.dotD2H +=
        pAp.d2h;

    timing.dotCPU +=
        pAp.cpu;


    double alpha =
        rsold / pAp.result;


    // ----------------------------------------------------
    // x = x + alpha*p
    // ----------------------------------------------------

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(
        cudaEventRecord(start)
    );

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

    CUDA_CHECK(
        cudaEventRecord(stop)
    );

    CUDA_CHECK(
        cudaEventSynchronize(stop)
    );

    CUDA_CHECK(
        cudaEventElapsedTime(
            &milliseconds,
            start,
            stop
        )
    );

    timing.axpy +=
        milliseconds;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));


    // ----------------------------------------------------
    // r = r - alpha*Ap
    // ----------------------------------------------------

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(
        cudaEventRecord(start)
    );

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

    CUDA_CHECK(
        cudaEventRecord(stop)
    );

    CUDA_CHECK(
        cudaEventSynchronize(stop)
    );

    CUDA_CHECK(
        cudaEventElapsedTime(
            &milliseconds,
            start,
            stop
        )
    );

    timing.axpy +=
        milliseconds;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));


    // ----------------------------------------------------
    // rsnew = r^T r
    // ----------------------------------------------------

    DotTiming rsnew =
        dotGPUProfiled(
            d_r,
            d_r,
            d_partialSums,
            n,
            blockSize
        );

    timing.dotKernel +=
        rsnew.kernel;

    timing.dotD2H +=
        rsnew.d2h;

    timing.dotCPU +=
        rsnew.cpu;


    residual =
        std::sqrt(rsnew.result);

    iterations =
        iter + 1;


    if (residual < tolerance)
    {
        break;
    }


    double beta =
        rsnew.result / rsold;


    // ----------------------------------------------------
    // Combine
    // ----------------------------------------------------

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(
        cudaEventRecord(start)
    );

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

    CUDA_CHECK(
        cudaEventRecord(stop)
    );

    CUDA_CHECK(
        cudaEventSynchronize(stop)
    );

    CUDA_CHECK(
        cudaEventElapsedTime(
            &milliseconds,
            start,
            stop
        )
    );

    timing.combine +=
        milliseconds;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));


    rsold =
        rsnew.result;
}

auto fullStopCPU =
    std::chrono::steady_clock::now();

double fullWallMilliseconds =
    std::chrono::duration<double, std::milli>(
        fullStopCPU - fullStartCPU
    ).count();

// ========================================================
// Stop Full Timer
// ========================================================

CUDA_CHECK(
    cudaEventRecord(fullStop)
);

CUDA_CHECK(
    cudaEventSynchronize(fullStop)
);

float fullMilliseconds = 0.0f;

CUDA_CHECK(
    cudaEventElapsedTime(
        &fullMilliseconds,
        fullStart,
        fullStop
    )
);

timing.fullCG =
    fullMilliseconds;


// ========================================================
// Copy Result
// ========================================================

std::vector<double> x(n);

CUDA_CHECK(
    cudaMemcpy(
        x.data(),
        d_x,
        n * sizeof(double),
        cudaMemcpyDeviceToHost
    )
);


// ========================================================
// Correctness
// ========================================================

double maxError = 0.0;

for (int i = 0; i < n; i++)
{
    double error =
        std::abs(
            x[i] - xTrue[i]
        );

    maxError =
        std::max(
            maxError,
            error
        );
}


// ========================================================
// Calculate totals
// ========================================================

double measuredTotal =
    timing.spmv
    + timing.dotKernel
    + timing.dotD2H
    + timing.dotCPU
    + timing.axpy
    + timing.combine;


double unaccounted =
    timing.fullCG
    - measuredTotal;


// ========================================================
// Output
// ========================================================

std::cout
    << "================================"
    << std::endl;

std::cout
    << "CUDA CG Bottleneck Breakdown"
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
    << "Iterations  : "
    << iterations
    << std::endl;


std::cout
    << "\n--------------------------------"
    << std::endl;

std::cout
    << "Timing Breakdown"
    << std::endl;

std::cout
    << "--------------------------------"
    << std::endl;

std::cout
    << "SpMV total          : "
    << timing.spmv
    << " ms"
    << std::endl;

std::cout
    << "Dot kernel total    : "
    << timing.dotKernel
    << " ms"
    << std::endl;

std::cout
    << "Dot D2H total       : "
    << timing.dotD2H
    << " ms"
    << std::endl;

std::cout
    << "Dot CPU reduction   : "
    << timing.dotCPU
    << " ms"
    << std::endl;

std::cout
    << "AXPY total          : "
    << timing.axpy
    << " ms"
    << std::endl;

std::cout
    << "Combine total       : "
    << timing.combine
    << " ms"
    << std::endl;

std::cout
    << "--------------------------------"
    << std::endl;

std::cout
    << "Measured total      : "
    << measuredTotal
    << " ms"
    << std::endl;

std::cout
    << "Full CG             : "
    << timing.fullCG
    << " ms"
    << std::endl;

std::cout
    << "Full CG wall time  : "
    << fullWallMilliseconds
    << " ms"
    << std::endl;

std::cout
    << "Unaccounted         : "
    << unaccounted
    << " ms"
    << std::endl;


std::cout
    << "\n--------------------------------"
    << std::endl;

std::cout
    << "Solver Result"
    << std::endl;

std::cout
    << "--------------------------------"
    << std::endl;

std::cout
    << "Residual            : "
    << residual
    << std::endl;

std::cout
    << "Max error           : "
    << maxError
    << std::endl;

std::cout
    << "Correctness         : "
    << (
        maxError < 1e-8
        ? "PASS"
        : "FAIL"
    )
    << std::endl;

std::cout
    << "================================"
    << std::endl;


// ========================================================
// Cleanup
// ========================================================

CUDA_CHECK(cudaEventDestroy(fullStart));
CUDA_CHECK(cudaEventDestroy(fullStop));

CUDA_CHECK(cudaFree(d_values));
CUDA_CHECK(cudaFree(d_colIndex));
CUDA_CHECK(cudaFree(d_rowPtr));

CUDA_CHECK(cudaFree(d_b));
CUDA_CHECK(cudaFree(d_x));
CUDA_CHECK(cudaFree(d_r));
CUDA_CHECK(cudaFree(d_p));
CUDA_CHECK(cudaFree(d_Ap));

CUDA_CHECK(cudaFree(d_partialSums));


return 0;
}
