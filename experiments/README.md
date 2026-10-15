# Experiments and Performance Optimization

This document records the experiments conducted during the development of the **GPU-Accelerated Numerical Solver** project. The goal is to understand the performance characteristics of sparse linear algebra operations, evaluate CUDA kernel configurations, and improve the performance of a Conjugate Gradient (CG) solver while preserving numerical correctness.

## 1. Experimental Objectives

The experiments focus on four main objectives:

- Establish a sequential CPU baseline for the Conjugate Gradient solver.
- Implement and evaluate CUDA kernels for sparse matrix-vector multiplication (SpMV), dot product, AXPY, and vector combination.
- Investigate the effect of CUDA thread block size on kernel and full-solver performance.
- Compare baseline and tuned CUDA configurations using repeated measurements and correctness checks.

## 2. Problem Configuration

The main experiments use a tridiagonal sparse matrix stored in Compressed Sparse Row (CSR) format.

| Parameter                                    | Value                   |
| -------------------------------------------- | ----------------------- |
| Matrix size                                  | 1,000,000 × 1,000,000   |
| Number of non-zero elements (NNZ)            | 2,999,998               |
| Matrix structure                             | Tridiagonal             |
| Main diagonal                                | 4                       |
| Adjacent diagonals                           | -1                      |
| Solver                                       | Conjugate Gradient (CG) |
| Iterations in recorded full-solver benchmark | 20                      |
| Known solution                               | All elements equal to 1 |

The matrix is symmetric positive definite, making it suitable for the Conjugate Gradient method.

## 3. CPU Baseline

The sequential CPU implementation provides a reference point for evaluating the GPU implementation.

The recorded CPU benchmark for a matrix of size \(N=1,000,000\) produced the following results:

| Metric                 |             Result |
| ---------------------- | -----------------: |
| Average runtime        |      238.900066 ms |
| Median runtime         |      238.083072 ms |
| Minimum runtime        |      230.435981 ms |
| Maximum runtime        |      252.774779 ms |
| L2 residual            | 2.86658500 × 10⁻¹¹ |
| Maximum absolute error | 5.43121104 × 10⁻¹² |
| Correctness            |               PASS |

These results establish a CPU reference measurement. A direct CPU-versus-GPU speedup claim should only be made after confirming that both implementations use comparable timing boundaries and include equivalent setup and data-transfer costs.

## 4. CUDA Kernel Experiments

Individual CUDA kernels were benchmarked to understand the cost of common vector and sparse linear algebra operations.

| Kernel             | Average runtime | Minimum runtime | Maximum runtime |
| ------------------ | --------------: | --------------: | --------------: |
| AXPY               |     0.097452 ms |     0.094496 ms |     0.120064 ms |
| CSR SpMV           |     0.230626 ms |     0.227648 ms |     0.239680 ms |
| Dot product        |     0.213165 ms |     0.209824 ms |     0.231968 ms |
| Vector combination |     0.093446 ms |     0.089664 ms |     0.097472 ms |

These are standalone kernel measurements. They should not be added together to estimate full-solver runtime because the kernels were measured separately and may have different timing boundaries and overheads.

## 5. Block Size Experiments

CUDA thread block size can affect parallel execution, reduction overhead, occupancy, and memory behavior. Experiments were conducted to compare different configurations.

### 5.1 Dot Product Kernel

Recorded median runtimes from a separate block-size experiment:

| Block size | Median runtime |
| ---------: | -------------: |
|        128 |    0.086205 ms |
|        256 |    0.101003 ms |
|        512 |    0.123625 ms |

In this experiment, block size 128 produced the lowest median runtime among the tested configurations. This result applies to the tested dot-product implementation and environment; it does not imply that the same block size is optimal for every CUDA kernel.

### 5.2 Full-Solver Configuration

The controlled full-solver experiment compared a baseline configuration against a tuned configuration.

| Parameter                     | Baseline | Tuned |
| ----------------------------- | -------: | ----: |
| SpMV block size               |      256 |   512 |
| Dot product block size        |      256 |   128 |
| AXPY / combination block size |      256 |   256 |
| Warm-up runs                  |        1 |     1 |
| Measured runs                 |       10 |    10 |

Recorded results:

| Metric                 |           Baseline |              Tuned |
| ---------------------- | -----------------: | -----------------: |
| Average runtime        |       32.926770 ms |       31.736883 ms |
| Median runtime         |       31.977975 ms |       30.414091 ms |
| Minimum runtime        |       30.205981 ms |       29.455890 ms |
| Maximum runtime        |       39.317282 ms |       37.483524 ms |
| L2 residual            | 2.86658500 × 10⁻¹¹ | 2.86658501 × 10⁻¹¹ |
| Maximum absolute error | 5.43121104 × 10⁻¹² | 5.43121104 × 10⁻¹² |
| Correctness            |               PASS |               PASS |

The tuned configuration improved the recorded median runtime by approximately **4.89%** relative to the baseline under this test setup.

The residual and maximum absolute error remained at comparable levels, indicating that the measured performance improvement did not introduce a meaningful loss of numerical accuracy in this test.

## 6. Correctness Validation

Performance measurements are only useful when the computed solution remains numerically correct.

The main solver experiments validate the computed result against the known solution and record:

- The L2 residual.
- The maximum absolute error.
- The final correctness status.

The recorded full-solver configurations passed the correctness checks for the tested problem.

## 7. Experimental Limitations

The current results have several limitations:

- Measurements were collected in specific environments and may vary with hardware, runtime conditions, and software versions.
- Standalone kernel measurements and full-solver measurements use different test scopes and should be interpreted separately.
- Some historical kernel benchmark records do not contain the number of measured runs.
- Nsight Systems (`nsys`) was not available in the recorded Kaggle environment, so a completed Nsight Systems profile is not claimed.
- A definitive CPU-versus-GPU speedup comparison requires verification that timing boundaries and included overheads are comparable.

## 8. Reproducibility and Result Tracking

The CUDA experiments were conducted in a Kaggle environment using an NVIDIA Tesla T4 GPU. The CPU baseline was measured separately.

Detailed recorded measurements are maintained in [`../results/benchmark-results.csv`](../results/benchmark-results.csv).

The relevant source files are organized under:

- `../src/cpu/` — CPU solver implementations.
- `../src/cuda/` — main CUDA solver versions.
- `../benchmarks/cpu/` — CPU benchmark programs.
- `../benchmarks/cuda/full-solver/` — full-solver performance experiments.
- `../benchmarks/cuda/kernels/` — individual CUDA kernel experiments.

## 9. Future Work

Potential next steps include:

- Validate timing boundaries and data-transfer overheads for a fair CPU-versus-GPU comparison.
- Profile GPU execution with Nsight Systems and Nsight Compute when the tools are available.
- Investigate memory throughput, occupancy, register usage, and reduction overhead.
- Compare the custom CUDA implementation with established libraries such as cuBLAS and cuSPARSE.
- Expand experiments to larger matrices and additional sparsity patterns.

The objective is to make performance improvements measurable, reproducible, and supported by numerical correctness checks.
