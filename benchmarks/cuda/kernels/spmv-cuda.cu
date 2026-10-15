#include <iostream>
#include <vector>
#include <cmath>
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
// Create Test Matrix
// ========================================

CSRMatrix createMatrix()
{
    CSRMatrix A;

    A.rows = 3;
    A.cols = 3;

    A.values = {
         4, -1,
        -1,  4, -1,
            -1,  4
    };

    A.colIndex = {
        0, 1,
        0, 1, 2,
           1, 2
    };

    A.rowPtr = {
        0, 2, 5, 7
    };

    return A;
}


// ========================================
// CPU SpMV
// ========================================

std::vector<double> spmvCPU(
    const CSRMatrix& A,
    const std::vector<double>& x
)
{
    std::vector<double> y(A.rows, 0.0);

    for (int row = 0; row < A.rows; row++)
    {
        double sum = 0.0;

        for (int k = A.rowPtr[row];
             k < A.rowPtr[row + 1];
             k++)
        {
            sum +=
                A.values[k]
                * x[A.colIndex[k]];
        }

        y[row] = sum;
    }

    return y;
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
    // ------------------------------------
    // Global thread index
    // ------------------------------------

    int row =
        blockIdx.x * blockDim.x
        + threadIdx.x;


    // ------------------------------------
    // Make sure row is valid
    // ------------------------------------

    if (row < rows)
    {
        double sum = 0.0;


        // --------------------------------
        // Traverse this row
        // --------------------------------

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


        // --------------------------------
        // Store result
        // --------------------------------

        y[row] = sum;
    }
}


// ========================================
// Main
// ========================================

int main()
{
    // ====================================
    // 1. Create Matrix
    // ====================================

    CSRMatrix A = createMatrix();


    // ====================================
    // 2. Create Vector x
    // ====================================

    std::vector<double> x = {
        1.0,
        2.0,
        3.0
    };


    // ====================================
    // 3. CPU SpMV
    // ====================================

    std::vector<double> yCPU =
        spmvCPU(A, x);


    std::cout << "CPU result:" << std::endl;

    for (double value : yCPU)
    {
        std::cout << value << " ";
    }

    std::cout << std::endl;


    // ====================================
    // 4. Allocate GPU memory
    // ====================================

    double* d_values;
    int* d_colIndex;
    int* d_rowPtr;
    double* d_x;
    double* d_y;


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
        cudaMalloc(
            &d_x,
            x.size() * sizeof(double)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_y,
            A.rows * sizeof(double)
        )
    );


    // ====================================
    // 5. Copy CPU → GPU
    // ====================================

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

    CUDA_CHECK(
        cudaMemcpy(
            d_x,
            x.data(),
            x.size() * sizeof(double),
            cudaMemcpyHostToDevice
        )
    );


    // ====================================
    // 6. CUDA Launch Configuration
    // ====================================

    int blockSize = 256;

    int gridSize =
        (A.rows + blockSize - 1)
        / blockSize;


    std::cout << std::endl;

    std::cout << "Block size: "
              << blockSize
              << std::endl;

    std::cout << "Grid size: "
              << gridSize
              << std::endl;


    // ====================================
    // 7. Launch CUDA SpMV
    // ====================================

    spmvKernel<<<gridSize, blockSize>>>(
        d_values,
        d_colIndex,
        d_rowPtr,
        d_x,
        d_y,
        A.rows
    );


    // Check kernel launch
    CUDA_CHECK(cudaGetLastError());

    // Wait for GPU
    CUDA_CHECK(cudaDeviceSynchronize());


    // ====================================
    // 8. Copy GPU → CPU
    // ====================================

    std::vector<double> yGPU(A.rows);

    CUDA_CHECK(
        cudaMemcpy(
            yGPU.data(),
            d_y,
            A.rows * sizeof(double),
            cudaMemcpyDeviceToHost
        )
    );


    // ====================================
    // 9. Print GPU result
    // ====================================

    std::cout << std::endl;

    std::cout << "GPU result:" << std::endl;

    for (double value : yGPU)
    {
        std::cout << value << " ";
    }

    std::cout << std::endl;


    // ====================================
    // 10. Check Correctness
    // ====================================

    bool correct = true;

    for (int i = 0; i < A.rows; i++)
    {
        if (std::abs(yCPU[i] - yGPU[i]) > 1e-10)
        {
            correct = false;
            break;
        }
    }


    std::cout << std::endl;

    if (correct)
    {
        std::cout << "Result: CORRECT"
                  << std::endl;
    }
    else
    {
        std::cout << "Result: WRONG"
                  << std::endl;
    }


    // ====================================
    // 11. Free GPU memory
    // ====================================

    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_colIndex));
    CUDA_CHECK(cudaFree(d_rowPtr));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));


    return 0;
}