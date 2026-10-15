#include <iostream>
#include <vector>
#include <cuda_runtime.h>

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

int main()
{
    int n = 1000000;

    double alpha = 2.0;

    std::vector<double> x(n, 1.0);
    std::vector<double> y(n, 3.0);

    double* d_x;
    double* d_y;

    cudaMalloc(
        &d_x,
        n * sizeof(double)
    );

    cudaMalloc(
        &d_y,
        n * sizeof(double)
    );

    cudaMemcpy(
        d_x,
        x.data(),
        n * sizeof(double),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        d_y,
        y.data(),
        n * sizeof(double),
        cudaMemcpyHostToDevice
    );

    int blockSize = 256;

    int gridSize =
        (n + blockSize - 1)
        / blockSize;

    axpyKernel<<<gridSize, blockSize>>>(
        d_y,
        alpha,
        d_x,
        n
    );

    cudaMemcpy(
        y.data(),
        d_y,
        n * sizeof(double),
        cudaMemcpyDeviceToHost
    );

    std::cout << "y[0] = "
              << y[0]
              << std::endl;

    std::cout << "y[last] = "
              << y[n - 1]
              << std::endl;

    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}