#include <iostream>
#include <vector>

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
// Main
// =========================================

int main()
{
    // -------------------------------------
    // 1. Create Matrix A
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
    // 2. Define the true answer
    // -------------------------------------

    std::vector<double> x_true = {
        1,
        2,
        3
    };


    // -------------------------------------
    // 3. Calculate b = A × x_true
    // -------------------------------------

    std::vector<double> b = spmv(A, x_true);


    // -------------------------------------
    // 4. Print b
    // -------------------------------------

    std::cout << "b = ";

    for (double value : b)
    {
        std::cout << value << " ";
    }

    std::cout << "\n";


    // -------------------------------------
    // 5. Initial guess
    // -------------------------------------

    std::vector<double> x = {
        0,
        0,
        0
    };


    // -------------------------------------
    // 6. Calculate A × x
    // -------------------------------------

    std::vector<double> Ax = spmv(A, x);


    // -------------------------------------
    // 7. Calculate residual
    //
    // r = b - Ax
    // -------------------------------------

    std::vector<double> r(3);

    for (int i = 0; i < 3; i++)
    {
        r[i] = b[i] - Ax[i];
    }


    // -------------------------------------
    // 8. Print Ax
    // -------------------------------------

    std::cout << "Ax = ";

    for (double value : Ax)
    {
        std::cout << value << " ";
    }

    std::cout << "\n";


    // -------------------------------------
    // 9. Print residual
    // -------------------------------------

    std::cout << "r = ";

    for (double value : r)
    {
        std::cout << value << " ";
    }

    std::cout << "\n";


    return 0;
}