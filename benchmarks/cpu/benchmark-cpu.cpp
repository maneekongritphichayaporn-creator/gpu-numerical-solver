#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>

struct CSRMatrix
{
    int rows;
    int cols;

    std::vector<double> values;
    std::vector<int> colIndex;
    std::vector<int> rowPtr;
};

// =========================================
// Create Tridiagonal Matrix
// =========================================

CSRMatrix createTridiagonalMatrix(int n)
{
    CSRMatrix A;

    A.rows = n;
    A.cols = n;

    A.rowPtr.resize(n + 1);

    for (int i = 0; i < n; i++)
    {
        // ---------------------------------
        // Left value
        // ---------------------------------

        if (i > 0)
        {
            A.values.push_back(-1.0);
            A.colIndex.push_back(i - 1);
        }

        // ---------------------------------
        // Diagonal
        // ---------------------------------

        A.values.push_back(4.0);
        A.colIndex.push_back(i);

        // ---------------------------------
        // Right value
        // ---------------------------------

        if (i < n - 1)
        {
            A.values.push_back(-1.0);
            A.colIndex.push_back(i + 1);
        }

        // Number of values stored so far
        A.rowPtr[i + 1] =
            A.values.size();
    }

    return A;
}

// =========================================
// Sparse Matrix × Vector
// =========================================

std::vector<double> spmv(
    const CSRMatrix &A,
    const std::vector<double> &x)
{
    std::vector<double> y(
        A.rows,
        0.0);

    for (int row = 0; row < A.rows; row++)
    {
        for (
            int k = A.rowPtr[row];
            k < A.rowPtr[row + 1];
            k++)
        {
            y[row] +=
                A.values[k] *
                x[A.colIndex[k]];
        }
    }

    return y;
}

// =========================================
// Dot Product
// =========================================

double dot(
    const std::vector<double> &a,
    const std::vector<double> &b)
{
    double result = 0.0;

    for (size_t i = 0; i < a.size(); i++)
    {
        result += a[i] * b[i];
    }

    return result;
}

// =========================================
// Vector Add
// result = a + b
// =========================================

std::vector<double> add(
    const std::vector<double> &a,
    const std::vector<double> &b)
{
    std::vector<double> result(a.size());

    for (size_t i = 0; i < a.size(); i++)
    {
        result[i] =
            a[i] + b[i];
    }

    return result;
}

// =========================================
// Vector Subtract
// result = a - b
// =========================================

std::vector<double> subtract(
    const std::vector<double> &a,
    const std::vector<double> &b)
{
    std::vector<double> result(a.size());

    for (size_t i = 0; i < a.size(); i++)
    {
        result[i] =
            a[i] - b[i];
    }

    return result;
}

// =========================================
// Scale
// result = alpha * a
// =========================================

std::vector<double> scale(
    double alpha,
    const std::vector<double> &a)
{
    std::vector<double> result(a.size());

    for (size_t i = 0; i < a.size(); i++)
    {
        result[i] =
            alpha * a[i];
    }

    return result;
}

void axpy(
    std::vector<double> &y,
    double alpha,
    const std::vector<double> &x)
{
    for (int i = 0; i < y.size(); i++)
    {
        y[i] += alpha * x[i];
    }
}

// =========================================
// Conjugate Gradient
// =========================================
struct CGResult
{
    std::vector<double> x;
    int iterations;
    double residual;
};

CGResult conjugateGradient(
    const CSRMatrix &A,
    const std::vector<double> &b,
    int maxIterations,
    double tolerance)
{
    int n = A.rows;

    // Initial solution
    std::vector<double> x(n, 0.0);

    // r = b - A*x
    std::vector<double> Ax = spmv(A, x);

    std::vector<double> r(n);

    for (int i = 0; i < n; i++)
    {
        r[i] = b[i] - Ax[i];
    }

    // p = r
    std::vector<double> p = r;

    double rsold = dot(r, r);

    int iterations = 0;

    for (int k = 0; k < maxIterations; k++)
    {
        iterations = k + 1;

        // Ap = A*p
        std::vector<double> Ap = spmv(A, p);

        // alpha = (r^T r) / (p^T A p)
        double denominator = dot(p, Ap);

        double alpha = rsold / denominator;

        // x = x + alpha*p
        axpy(x, alpha, p);

        // r = r - alpha*Ap
        axpy(r, -alpha, Ap);

        // Check convergence
        double rsnew = dot(r, r);

        double residual = std::sqrt(rsnew);

        if (residual < tolerance)
        {
            return {
                x,
                iterations,
                residual};
        }

        // beta = rsnew / rsold
        double beta = rsnew / rsold;

        // p = r + beta*p
        for (int i = 0; i < n; i++)
        {
            p[i] = r[i] + beta * p[i];
        }

        rsold = rsnew;
    }

    // If max iterations reached
    double residual = std::sqrt(dot(r, r));

    return {
        x,
        iterations,
        residual};
}

// =========================================
// Main
// =========================================

int main()
{
    int n = 1000000;

    // ========================================
    // 1. Create matrix
    // ========================================

    CSRMatrix A = createTridiagonalMatrix(n);

    // ========================================
    // 2. Create true solution
    // ========================================

    std::vector<double> x_true(n, 1.0);

    // ========================================
    // 3. Calculate b = A * x_true
    // ========================================

    std::vector<double> b = spmv(A, x_true);

    // ========================================
    // 4. Benchmark
    // ========================================

    const int runs = 10;

    std::vector<double> times;

    int iterations = 0;
    double finalResidual = 0.0;

    for (int run = 0; run < runs; run++)
    {
        auto start = std::chrono::high_resolution_clock::now();

        CGResult result =
            conjugateGradient(A, b, 1000, 1e-10);

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed =
            end - start;

        times.push_back(elapsed.count());

        // Save iteration count from first run
        if (run == 0)
        {
            iterations = result.iterations;
            finalResidual = result.residual;
        }
    }

    // ========================================
    // 5. Calculate statistics
    // ========================================

    double sum = 0.0;
    double minTime = times[0];
    double maxTime = times[0];

    for (double time : times)
    {
        sum += time;

        if (time < minTime)
            minTime = time;

        if (time > maxTime)
            maxTime = time;
    }

    double average = sum / runs;

    std::sort(times.begin(), times.end());

    double median;

    if (runs % 2 == 0)
    {
        median =
            (times[runs / 2 - 1] + times[runs / 2]) / 2.0;
    }
    else
    {
        median = times[runs / 2];
    }

    // ========================================
    // 6. Print result
    // ========================================

    std::cout << "================================\n";
    std::cout << "CPU Conjugate Gradient Benchmark\n";
    std::cout << "================================\n\n";

    std::cout << "Matrix size : "
              << n << " x " << n << "\n";

    std::cout << "NNZ         : "
              << A.values.size() << "\n\n";

    std::cout << "Iterations  : "
              << iterations << "\n";

    std::cout << "Residual    : "
              << finalResidual << "\n\n";

    std::cout << "Runs        : "
              << runs << "\n";

    std::cout << "Average     : "
              << average << " ms\n";

    std::cout << "Median      : "
              << median << " ms\n";

    std::cout << "Min         : "
              << minTime << " ms\n";

    std::cout << "Max         : "
              << maxTime << " ms\n";

    return 0;
}