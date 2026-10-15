
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>

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
// Create Tridiagonal Matrix
//
// A =
// [ 4 -1  0  0 ...]
// [-1  4 -1  0 ...]
// [ 0 -1  4 -1 ...]
// [...           ]
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
            static_cast<int>(
                A.values.size()
            );
    }


    return A;
}


// ============================================================
// CSR SpMV Kernel
//
// y = A * x
//
// One thread processes one row.
//
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
// Main
// ============================================================

int main()
{
    // --------------------------------------------------------
    // Configuration
    // --------------------------------------------------------

    const int n =
        1000000;

    const int blockSize =
        256;

    const int runs =
        100;


    // --------------------------------------------------------
    // Create matrix
    // --------------------------------------------------------

    CSRMatrix A =
        createTridiagonalMatrix(n);


    std::cout
        << "================================"
        << std::endl;

    std::cout
        << "CUDA CSR SpMV Profile"
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


    // --------------------------------------------------------
    // Grid size
    // --------------------------------------------------------

    const int gridSize =
        (n + blockSize - 1)
        / blockSize;


    std::cout
        << "Block size  : "
        << blockSize
        << std::endl;

    std::cout
        << "Grid size   : "
        << gridSize
        << std::endl;

    std::cout
        << "Runs        : "
        << runs
        << std::endl;


    // ========================================================
    // Host vectors
    // ========================================================

    std::vector<double> h_x(
        n,
        1.0
    );


    std::vector<double> h_y(
        n,
        0.0
    );


    std::vector<double> h_expected(
        n
    );


    // --------------------------------------------------------
    // Expected result
    //
    // x = [1,1,1,...,1]
    //
    // First row  = 4 - 1 = 3
    // Middle     = -1 + 4 - 1 = 2
    // Last row   = -1 + 4 = 3
    // --------------------------------------------------------

    h_expected[0] =
        3.0;


    for (
        int i = 1;
        i < n - 1;
        i++
    )
    {
        h_expected[i] =
            2.0;
    }


    h_expected[n - 1] =
        3.0;


    // ========================================================
    // Device memory
    // ========================================================

    double* d_values = nullptr;
    int* d_colIndex = nullptr;
    int* d_rowPtr = nullptr;

    double* d_x = nullptr;
    double* d_y = nullptr;


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


    // ========================================================
    // Copy CPU → GPU
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
            d_x,
            h_x.data(),
            n * sizeof(double),
            cudaMemcpyHostToDevice
        )
    );


    // ========================================================
    // Warm-up
    // ========================================================

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
        cudaGetLastError()
    );


    CUDA_CHECK(
        cudaDeviceSynchronize()
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
    // Benchmark
    // ========================================================

    std::vector<float> times;

    times.reserve(runs);


    for (
        int run = 0;
        run < runs;
        run++
    )
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


        times.push_back(
            milliseconds
        );
    }


    // ========================================================
    // Statistics
    // ========================================================

    float sum = 0.0f;

    float minTime =
        times[0];

    float maxTime =
        times[0];


    for (float t : times)
    {
        sum += t;


        if (t < minTime)
        {
            minTime = t;
        }


        if (t > maxTime)
        {
            maxTime = t;
        }
    }


    float average =
        sum / runs;


    // ========================================================
    // Copy result GPU → CPU
    // ========================================================

    CUDA_CHECK(
        cudaMemcpy(
            h_y.data(),
            d_y,
            n * sizeof(double),
            cudaMemcpyDeviceToHost
        )
    );


    // ========================================================
    // Correctness
    // ========================================================

    double maxError =
        0.0;


    for (int i = 0; i < n; i++)
    {
        double error =
            std::abs(
                h_y[i]
                - h_expected[i]
            );


        if (error > maxError)
        {
            maxError =
                error;
        }
    }


    // ========================================================
    // Output
    // ========================================================

    std::cout
        << std::endl;

    std::cout
        << "================================"
        << std::endl;

    std::cout
        << "Benchmark Result"
        << std::endl;

    std::cout
        << "================================"
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
        << "y[0]       : "
        << h_y[0]
        << std::endl;


    std::cout
        << "y[n/2]     : "
        << h_y[n / 2]
        << std::endl;


    std::cout
        << "y[last]    : "
        << h_y[n - 1]
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
        cudaFree(d_values)
    );


    CUDA_CHECK(
        cudaFree(d_colIndex)
    );


    CUDA_CHECK(
        cudaFree(d_rowPtr)
    );


    CUDA_CHECK(
        cudaFree(d_x)
    );


    CUDA_CHECK(
        cudaFree(d_y)
    );


    return 0;
}
