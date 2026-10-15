#include <iostream>
#include <vector>
#include <cmath>

struct CSRMatrix
{
    int rows;
    int cols;

    std::vector<double> values;
    std::vector<int> colIndex;
    std::vector<int> rowPtr;
};


// =========================================
// Sparse Matrix × Vector
// y = A × x
// =========================================

std::vector<double> spmv(
    const CSRMatrix& A,
    const std::vector<double>& x
)
{
    std::vector<double> y(A.rows, 0.0);

    for (int row = 0; row < A.rows; row++)
    {
        for (
            int k = A.rowPtr[row];
            k < A.rowPtr[row + 1];
            k++
        )
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
    const std::vector<double>& a,
    const std::vector<double>& b
)
{
    double result = 0.0;

    for (size_t i = 0; i < a.size(); i++)
    {
        result += a[i] * b[i];
    }

    return result;
}


// =========================================
// Vector Addition
// result = a + b
// =========================================

std::vector<double> add(
    const std::vector<double>& a,
    const std::vector<double>& b
)
{
    std::vector<double> result(a.size());

    for (size_t i = 0; i < a.size(); i++)
    {
        result[i] = a[i] + b[i];
    }

    return result;
}


// =========================================
// Vector Subtraction
// result = a - b
// =========================================

std::vector<double> subtract(
    const std::vector<double>& a,
    const std::vector<double>& b
)
{
    std::vector<double> result(a.size());

    for (size_t i = 0; i < a.size(); i++)
    {
        result[i] = a[i] - b[i];
    }

    return result;
}


// =========================================
// Vector Scale
// result = alpha * a
// =========================================

std::vector<double> scale(
    double alpha,
    const std::vector<double>& a
)
{
    std::vector<double> result(a.size());

    for (size_t i = 0; i < a.size(); i++)
    {
        result[i] = alpha * a[i];
    }

    return result;
}


// =========================================
// Conjugate Gradient
// =========================================

std::vector<double> conjugateGradient(
    const CSRMatrix& A,
    const std::vector<double>& b,
    int maxIterations,
    double tolerance
)
{
    // -------------------------------------
    // Initial guess
    // x = [0, 0, 0]
    // -------------------------------------

    std::vector<double> x(
        A.rows,
        0.0
    );


    // -------------------------------------
    // r = b - A*x
    // -------------------------------------

    std::vector<double> Ax = spmv(A, x);

    std::vector<double> r =
        subtract(b, Ax);


    // -------------------------------------
    // p = r
    // -------------------------------------

    std::vector<double> p = r;


    // -------------------------------------
    // rᵀr
    // -------------------------------------

    double rsold = dot(r, r);


    // -------------------------------------
    // CG iterations
    // -------------------------------------

    for (int iteration = 0;
         iteration < maxIterations;
         iteration++)
    {
        // A*p
        std::vector<double> Ap =
            spmv(A, p);


        // pᵀAp
        double denominator =
            dot(p, Ap);


        // alpha
        double alpha =
            rsold / denominator;


        // x = x + alpha*p
        std::vector<double> alphaP =
            scale(alpha, p);

        x = add(x, alphaP);


        // r = r - alpha*Ap
        std::vector<double> alphaAp =
            scale(alpha, Ap);

        r = subtract(r, alphaAp);


        // rᵀr
        double rsnew =
            dot(r, r);


        // Print progress
        std::cout
            << "Iteration "
            << iteration + 1
            << ", residual = "
            << std::sqrt(rsnew)
            << "\n";


        // Check convergence
        if (std::sqrt(rsnew) < tolerance)
        {
            break;
        }


        // beta
        double beta =
            rsnew / rsold;


        // p = r + beta*p
        std::vector<double> betaP =
            scale(beta, p);

        p = add(r, betaP);


        // Prepare for next iteration
        rsold = rsnew;
    }


    return x;
}


// =========================================
// Main
// =========================================

int main()
{
    // -------------------------------------
    // Matrix A
    // -------------------------------------

    CSRMatrix A;

    A.rows = 3;
    A.cols = 3;

    A.values = {
        4, -1,
        -1, 4, -1,
        -1, 4
    };

    A.colIndex = {
        0, 1,
        0, 1, 2,
        1, 2
    };

    A.rowPtr = {
        0, 2, 5, 7
    };


    // -------------------------------------
    // True solution
    // -------------------------------------

    std::vector<double> x_true = {
        1,
        2,
        3
    };


    // -------------------------------------
    // Create b = A*x_true
    // -------------------------------------

    std::vector<double> b =
        spmv(A, x_true);


    // -------------------------------------
    // Solve Ax = b
    // -------------------------------------

    std::vector<double> x =
        conjugateGradient(
            A,
            b,
            100,
            1e-10
        );


    // -------------------------------------
    // Print solution
    // -------------------------------------

    std::cout << "\nSolution:\n";

    for (double value : x)
    {
        std::cout << value << " ";
    }

    std::cout << "\n";


    return 0;
}