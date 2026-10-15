
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
        std::cerr << "CUDA Error: "                              \
                  << cudaGetErrorString(err) << std::endl;    \
        std::exit(EXIT_FAILURE);                              \
    }                                                         \
} while (0)


__global__ void combineKernel(
    double* p,
    const double* r,
    double beta,
    int n
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        p[i] = r[i] + beta * p[i];
    }
}


int main()
{
    const int n = 1'000'000;
    const int blockSize = 256;
    const int gridSize =
        (n + blockSize - 1) / blockSize;

    const int runs = 100;

    const double beta = 0.5;

    std::cout << "================================\n";
    std::cout << "CUDA Combine Kernel Profile\n";
    std::cout << "================================\n";

    std::cout << "N          : " << n << "\n";
    std::cout << "Block size : " << blockSize << "\n";
    std::cout << "Grid size  : " << gridSize << "\n";
    std::cout << "Runs       : " << runs << "\n";
    std::cout << "Beta       : " << beta << "\n";

    // --------------------------------
    // Host data
    // --------------------------------

    std::vector<double> h_p(n, 2.0);
    std::vector<double> h_r(n, 1.0);

    // Expected:
    //
    // p[i] = r[i] + beta * p[i]
    //      = 1 + 0.5 * 2
    //      = 2
    //
    const double expectedValue = 2.0;

    // --------------------------------
    // Device memory
    // --------------------------------

    double* d_p = nullptr;
    double* d_r = nullptr;

    CUDA_CHECK(cudaMalloc(
        &d_p,
        n * sizeof(double)
    ));

    CUDA_CHECK(cudaMalloc(
        &d_r,
        n * sizeof(double)
    ));

    CUDA_CHECK(cudaMemcpy(
        d_p,
        h_p.data(),
        n * sizeof(double),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_r,
        h_r.data(),
        n * sizeof(double),
        cudaMemcpyHostToDevice
    ));

    // --------------------------------
    // Warm-up
    // --------------------------------

    combineKernel<<<gridSize, blockSize>>>(
        d_p,
        d_r,
        beta,
        n
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Reset p after warm-up
    CUDA_CHECK(cudaMemcpy(
        d_p,
        h_p.data(),
        n * sizeof(double),
        cudaMemcpyHostToDevice
    ));

    // --------------------------------
    // Benchmark
    // --------------------------------

    std::vector<float> times;
    times.reserve(runs);

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int run = 0; run < runs; run++)
    {
        CUDA_CHECK(cudaEventRecord(start));

        combineKernel<<<gridSize, blockSize>>>(
            d_p,
            d_r,
            beta,
            n
        );

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &milliseconds,
            start,
            stop
        ));

        times.push_back(milliseconds);

        // Reset p for the next run
        CUDA_CHECK(cudaMemcpy(
            d_p,
            h_p.data(),
            n * sizeof(double),
            cudaMemcpyHostToDevice
        ));
    }

    // --------------------------------
    // Statistics
    // --------------------------------

    double sum = 0.0;

    float minTime = *std::min_element(
        times.begin(),
        times.end()
    );

    float maxTime = *std::max_element(
        times.begin(),
        times.end()
    );

    for (float t : times)
        sum += t;

    double average =
        sum / times.size();

    // --------------------------------
    // Correctness
    // --------------------------------

    std::vector<double> h_result(n);

    CUDA_CHECK(cudaMemcpy(
        h_result.data(),
        d_p,
        n * sizeof(double),
        cudaMemcpyDeviceToHost
    ));

    // Because the final benchmark run was followed by
    // a reset, perform one final kernel execution.
    CUDA_CHECK(cudaMemcpy(
        d_p,
        h_p.data(),
        n * sizeof(double),
        cudaMemcpyHostToDevice
    ));

    combineKernel<<<gridSize, blockSize>>>(
        d_p,
        d_r,
        beta,
        n
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        h_result.data(),
        d_p,
        n * sizeof(double),
        cudaMemcpyDeviceToHost
    ));

    double maxError = 0.0;

    for (int i = 0; i < n; i++)
    {
        double error =
            std::abs(h_result[i] - expectedValue);

        maxError =
            std::max(maxError, error);
    }

    // --------------------------------
    // Output
    // --------------------------------

    std::cout << "\n================================\n";
    std::cout << "Benchmark Result\n";
    std::cout << "================================\n";

    std::cout << "Average    : "
              << average << " ms\n";

    std::cout << "Min        : "
              << minTime << " ms\n";

    std::cout << "Max        : "
              << maxTime << " ms\n";

    std::cout << "\n";
    std::cout << "p[0]       : "
              << h_result[0] << "\n";

    std::cout << "p[n/2]     : "
              << h_result[n / 2] << "\n";

    std::cout << "p[last]    : "
              << h_result[n - 1] << "\n";

    std::cout << "Max error  : "
              << maxError << "\n";

    std::cout << "Correctness : "
              << (maxError < 1e-12 ? "PASS" : "FAIL")
              << "\n";

    std::cout << "================================\n";

    // --------------------------------
    // Cleanup
    // --------------------------------

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_r));

    return 0;
}
