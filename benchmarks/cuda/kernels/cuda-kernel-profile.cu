
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>

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
// AXPY
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
    y[i] += alpha * x[i];
}
}


// ============================================================
// Main
// ============================================================

int main()
{
const int n = 1000000;

const int blockSize = 256;

const int gridSize =
    (n + blockSize - 1)
    / blockSize;


// --------------------------------------------------------
// Host
// --------------------------------------------------------

std::vector<double> h_x(
    n,
    1.0
);

std::vector<double> h_y(
    n,
    2.0
);


// --------------------------------------------------------
// Device
// --------------------------------------------------------

double* d_x = nullptr;
double* d_y = nullptr;


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


CUDA_CHECK(
    cudaMemcpy(
        d_x,
        h_x.data(),
        n * sizeof(double),
        cudaMemcpyHostToDevice
    )
);


CUDA_CHECK(
    cudaMemcpy(
        d_y,
        h_y.data(),
        n * sizeof(double),
        cudaMemcpyHostToDevice
    )
);


// --------------------------------------------------------
// Warm-up
// --------------------------------------------------------

axpyKernel<<<
    gridSize,
    blockSize
>>>(
    d_y,
    2.0,
    d_x,
    n
);

CUDA_CHECK(
    cudaDeviceSynchronize()
);


// --------------------------------------------------------
// Reset data after warm-up
// --------------------------------------------------------

CUDA_CHECK(
    cudaMemcpy(
        d_y,
        h_y.data(),
        n * sizeof(double),
        cudaMemcpyHostToDevice
    )
    );

// --------------------------------------------------------
// CUDA Events
// --------------------------------------------------------

cudaEvent_t start;
cudaEvent_t stop;

CUDA_CHECK(
cudaEventCreate(&start)
);

CUDA_CHECK(
cudaEventCreate(&stop)
);


// --------------------------------------------------------
// Benchmark configuration
// --------------------------------------------------------

const int runs = 100;

std::vector<float> times;
times.reserve(runs);


// --------------------------------------------------------
// Benchmark
// --------------------------------------------------------

for (int run = 0; run < runs; run++)
{
CUDA_CHECK(
    cudaEventRecord(start)
);


axpyKernel<<<
    gridSize,
    blockSize
>>>(
    d_y,
    2.0,
    d_x,
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


times.push_back(
    milliseconds
);
}


// --------------------------------------------------------
// Statistics
// --------------------------------------------------------

float sum = 0.0f;
float minTime = times[0];
float maxTime = times[0];


for (float t : times)
{
    sum += t;

    if (t < minTime)
        minTime = t;

    if (t > maxTime)
        maxTime = t;
}


float average =
    sum / runs;


// --------------------------------------------------------
// Copy result back
// --------------------------------------------------------

CUDA_CHECK(
    cudaMemcpy(
        h_y.data(),
        d_y,
        n * sizeof(double),
        cudaMemcpyDeviceToHost
    )
);


// --------------------------------------------------------
// Correctness
// --------------------------------------------------------

double expected =
    2.0 + 2.0 * runs;


double maxError = 0.0;


for (int i = 0; i < n; i++)
{
    double error =
        std::abs(
            h_y[i]
            - expected
        );

    if (error > maxError)
        maxError = error;
}


// --------------------------------------------------------
// Output
// --------------------------------------------------------

std::cout
    << "================================"
    << std::endl;

std::cout
    << "CUDA AXPY Kernel Profile"
    << std::endl;

std::cout
    << "================================"
    << std::endl;

std::cout
    << "N          : "
    << n
    << std::endl;

std::cout
    << "Block size : "
    << blockSize
    << std::endl;

std::cout
    << "Grid size  : "
    << gridSize
    << std::endl;

std::cout
    << "Runs       : "
    << runs
    << std::endl;

std::cout
    << std::endl;

std::cout
    << "Average    : "
    << average
    << " ms"
    << std::endl;

std::cout
    << "Min        : "
    << minTime
    << " ms"
    << std::endl;

std::cout
    << "Max        : "
    << maxTime
    << " ms"
    << std::endl;

std::cout
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


// --------------------------------------------------------
// Cleanup
// --------------------------------------------------------

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
