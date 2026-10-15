#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void spmvCSR(const double* values, const int* colIndex,
                        const int* rowPtr, const double* x,
                        double* y, int rows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        double sum = 0.0;
        for (int k = rowPtr[row]; k < rowPtr[row + 1]; ++k)
            sum += values[k] * x[colIndex[k]];
        y[row] = sum;
    }
}

struct Result { int block; double avgMs, medianMs, minMs, maxMs, bandwidthGBs; bool pass; };

int main() {
    const int n = 1000000;
    const int nnz = 3 * n - 2;
    const int warmup = 20;
    const int launchesPerBatch = 200;
    const int batches = 9;
    const int blocks[] = {128, 256, 512};

    std::vector<double> h_values(nnz);
    std::vector<int> h_col(nnz), h_rowPtr(n + 1);
    std::vector<double> h_x(n), h_y(n, 0.0), h_ref(n, 0.0);

    int p = 0;
    h_rowPtr[0] = 0;
    for (int r = 0; r < n; ++r) {
        if (r > 0) { h_values[p] = -1.0; h_col[p++] = r - 1; }
        h_values[p] = 4.0; h_col[p++] = r;
        if (r + 1 < n) { h_values[p] = -1.0; h_col[p++] = r + 1; }
        h_rowPtr[r + 1] = p;
        h_x[r] = 1.0;
    }
    for (int r = 0; r < n; ++r) {
        for (int k = h_rowPtr[r]; k < h_rowPtr[r + 1]; ++k)
            h_ref[r] += h_values[k] * h_x[h_col[k]];
    }

    double *d_values = nullptr, *d_x = nullptr, *d_y = nullptr;
    int *d_col = nullptr, *d_rowPtr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_values, nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_col, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rowPtr, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_values, h_values.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col, h_col.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowPtr, h_rowPtr.data(), (n + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(double), cudaMemcpyHostToDevice));

    // Approximate logical traffic: values + column indices + row pointers + x reads + y writes.
    // Actual DRAM traffic can be lower due to cache reuse, especially for x and rowPtr.
    const double csrBytes = nnz * (sizeof(double) + sizeof(int)) + (n + 1) * sizeof(int);
    const double logicalBytes = csrBytes + nnz * sizeof(double) + n * sizeof(double);

    std::vector<Result> results;
    std::cout << "CUDA CSR SpMV Bandwidth Benchmark\n"
              << "N=" << n << ", NNZ=" << nnz
              << ", warmup=" << warmup
              << ", launches/batch=" << launchesPerBatch
              << ", batches=" << batches << "\n"
              << "Kernel-only timing; data stays on GPU.\n\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Block  Grid    Median ms/call   Min ms/call   Max ms/call   Approx logical GB/s   Correctness\n";
    std::cout << "--------------------------------------------------------------------------------------------\n";

    for (int block : blocks) {
        int grid = (n + block - 1) / block;
        for (int i = 0; i < warmup; ++i) {
            spmvCSR<<<grid, block>>>(d_values, d_col, d_rowPtr, d_x, d_y, n);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> times;
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        for (int b = 0; b < batches; ++b) {
            CUDA_CHECK(cudaEventRecord(start));
            for (int j = 0; j < launchesPerBatch; ++j)
                spmvCSR<<<grid, block>>>(d_values, d_col, d_rowPtr, d_x, d_y, n);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float elapsed = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
            times.push_back((double)elapsed / launchesPerBatch);
        }
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y, n * sizeof(double), cudaMemcpyDeviceToHost));

        double maxError = 0.0;
        for (int i = 0; i < n; ++i)
            maxError = std::max(maxError, std::abs(h_y[i] - h_ref[i]));
        bool pass = maxError < 1e-12;

        std::sort(times.begin(), times.end());
        double median = times[times.size() / 2];
        double avg = 0.0;
        for (double t : times) avg += t;
        avg /= times.size();
        double approxGBs = logicalBytes / (median * 1.0e6); // bytes/ms = MB/s; /1000 -> GB/s, simplifies to /1e6
        Result r{block, avg, median, times.front(), times.back(), approxGBs, pass};
        results.push_back(r);
        std::cout << std::setw(5) << block << "  " << std::setw(5) << grid << "  "
                  << std::setw(14) << median << "  " << std::setw(12) << times.front() << "  "
                  << std::setw(12) << times.back() << "  " << std::setw(19) << approxGBs << "   "
                  << (pass ? "PASS" : "FAIL") << " (max error=" << maxError << ")\n";
    }

    auto best = std::min_element(results.begin(), results.end(),
        [](const Result& a, const Result& b) { return a.medianMs < b.medianMs; });
    std::cout << "\nFastest median: block size " << best->block
              << " (" << best->medianMs << " ms/call)\n";
    std::cout << "Bandwidth is an approximate logical-byte estimate, not measured DRAM bandwidth; cache reuse can reduce actual traffic.\n";

    CUDA_CHECK(cudaFree(d_values)); CUDA_CHECK(cudaFree(d_col));
    CUDA_CHECK(cudaFree(d_rowPtr)); CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y));
    return best->pass ? 0 : 2;
}
