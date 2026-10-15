# GPU-Accelerated Numerical Solver

A C++ and CUDA project that accelerates sparse linear algebra using the **Conjugate Gradient (CG)** method. This project compares a CPU implementation against CUDA GPU implementations, evaluates numerical correctness, and benchmarks different CUDA thread-block configurations.

## Overview

Large-scale numerical problems often require repeated sparse matrix-vector operations and vector reductions. These operations can be computationally expensive on a CPU, making them suitable candidates for GPU acceleration.

This project implements a sparse linear system solver and explores how CUDA kernel configurations affect performance.

**Key objectives:**

- Implement a CPU baseline using C++.
- Implement a GPU-accelerated solver using CUDA C++.
- Compare baseline and tuned CUDA configurations.
- Validate numerical accuracy using residual and maximum-error checks.
- Measure performance using repeated benchmark runs.
- Establish a foundation for further GPU profiling and optimization.

## Problem Description

The solver addresses the linear system:

\[
Ax=b
\]

where \(A\) is a symmetric positive-definite tridiagonal sparse matrix.

| Property                |                       Value |
| ----------------------- | --------------------------: |
| Matrix size (N)         |                   1,000,000 |
| Non-zero elements (NNZ) |                   2,999,998 |
| Main diagonal           |                           4 |
| Adjacent diagonals      |                          -1 |
| CG iterations           |                          20 |
| Matrix storage          | Compressed Sparse Row (CSR) |

The test problem has a known solution in which every element of the solution vector \(x\) equals 1. This makes it possible to validate the solver against a known answer.

## Implementation

### CPU Implementation

The CPU implementation provides a sequential baseline for solving the sparse linear system using the Conjugate Gradient method.

It is compiled with optimization enabled using `g++ -O2`.

### CUDA GPU Implementation

The CUDA implementation parallelizes the main computational operations using GPU kernels:

- **Sparse Matrix-Vector Multiplication (SpMV):** Computes the product of a sparse matrix and a vector.
- **Dot Product:** Calculates vector inner products using block-level parallel reduction.
- **AXPY:** Performs vector updates of the form \(y = \alpha x + y\).
- **CG Vector Combination:** Updates the search direction during the Conjugate Gradient iterations.

The sparse matrix is stored in CSR format to avoid storing zero-valued elements.

## CUDA Optimization Experiments

The project compares a baseline CUDA configuration against a tuned configuration.

| Configuration | SpMV block size | Dot block size | AXPY / Combine block size |
| ------------- | --------------: | -------------: | ------------------------: |
| Baseline      |             256 |            256 |                       256 |
| Tuned         |             512 |            128 |                       256 |

The tuned configuration uses a larger block size for SpMV and a smaller block size for dot-product reduction.

These settings were selected based on empirical measurements for the tested workload. Optimal settings may differ for other matrix sizes, GPU architectures, or workloads.

## Benchmark Results

**Test environment:** Kaggle GPU notebook with an NVIDIA Tesla T4.

The results below are from the latest recorded benchmark runs. CUDA configurations were tested with 2 warm-up runs and 10 measured paired runs per configuration.

### CUDA Baseline vs. Tuned Configuration

| Metric          |     Baseline |        Tuned |
| --------------- | -----------: | -----------: |
| Average runtime | 32.926770 ms | 31.736883 ms |
| Median runtime  | 31.977975 ms | 30.414091 ms |
| Minimum runtime | 30.205981 ms | 29.455890 ms |
| Maximum runtime | 39.317282 ms | 37.483524 ms |
| Correctness     |         PASS |         PASS |

**Median runtime improvement: 4.89%** in the latest recorded comparison.

The tuned configuration achieved a lower median runtime in this experiment. However, cloud GPU performance can vary because of device contention and other runtime conditions. Further repeated testing is needed to establish how consistently the improvement holds.

### CPU Benchmark

The CPU implementation was benchmarked separately on the same problem size and iteration count.

| Metric                 |       CPU Result |
| ---------------------- | ---------------: |
| Average runtime        |    238.900066 ms |
| Median runtime         |    238.083072 ms |
| Minimum runtime        |    230.435981 ms |
| Maximum runtime        |    252.774779 ms |
| Residual L2            | `2.86658500e-11` |
| Maximum absolute error | `5.43121104e-12` |
| Correctness            |             PASS |

**Note:** CPU and GPU timings should not be treated as a definitive speedup comparison until their timing boundaries are verified to include equivalent work. In particular, memory allocation, synchronization, and host-to-device data transfers must be accounted for consistently.

## Numerical Correctness

The latest recorded benchmark runs passed the numerical correctness checks.

| Implementation |      Residual L2 | Maximum absolute error | Result |
| -------------- | ---------------: | ---------------------: | ------ |
| CPU            | `2.86658500e-11` |       `5.43121104e-12` | PASS   |
| CUDA Baseline  | `2.86658500e-11` |       `5.43121104e-12` | PASS   |
| CUDA Tuned     | `2.86658501e-11` |       `5.43121104e-12` | PASS   |

The residual measures how closely the computed solution satisfies the linear system. The maximum absolute error compares the computed solution with the known solution.

Small floating-point differences can occur because CPU and GPU implementations may accumulate operations in different orders.

## How to Run

Run these commands in an environment with the required compiler and source files.

### Requirements

- A C++ compiler supporting C++17
- NVIDIA CUDA Toolkit and `nvcc`
- An NVIDIA CUDA-capable GPU for the CUDA benchmark
- A CUDA-enabled environment such as Kaggle Notebooks

### Run the CUDA Benchmark

From the directory containing `cuda-cg-v3-controlled-baseline-vs-tuned.cu`:

```bash
nvcc -O2 cuda-cg-v3-controlled-baseline-vs-tuned.cu -o cg_compare
./cg_compare
```

This compiles and runs the controlled comparison between the baseline and tuned CUDA configurations.

### Run the CPU Benchmark

From the directory containing `cpu-cg-benchmark.cpp`:

```bash
g++ -O2 -std=c++17 cpu-cg-benchmark.cpp -o cpu_cg_benchmark
./cpu_cg_benchmark
```

If your source files are stored in different directories, update the commands to match your repository structure.

## Tools and Technologies

- C++
- CUDA C++
- NVIDIA CUDA Toolkit (`nvcc`)
- NVIDIA Tesla T4 GPU
- Kaggle Notebooks
- `nvidia-smi`

## Future Work

Potential next steps include:

- Verify equivalent CPU and GPU benchmark timing boundaries.
- Measure the impact of host-to-device and device-to-host data transfers.
- Profile GPU execution with Nsight Systems and Nsight Compute when available.
- Investigate memory throughput, occupancy, register usage, and kernel launch overhead.
- Explore shared-memory and reduction optimizations where profiling indicates a bottleneck.
- Benchmark larger matrices and different sparse matrix patterns.
- Automate repeated runs and export benchmark results to CSV.
- Compare additional CUDA configurations across different GPU architectures.

## Project Status

This is an experimental numerical computing and GPU optimization project. The current results demonstrate a working CPU solver, a CUDA implementation, numerical correctness checks, and an initial controlled configuration comparison.

Performance results are specific to the tested environment and should be reproduced before drawing broader conclusions.
