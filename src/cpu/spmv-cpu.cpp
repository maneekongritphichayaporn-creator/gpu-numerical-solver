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


// Sparse Matrix-Vector Multiplication
// y = A * x
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


int main()
{
    // =========================================
    // 1. Create CSR Matrix
    // =========================================

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


    // =========================================
    // 2. Create vector x
    // =========================================

    std::vector<double> x = {
        1,
        2,
        3
    };


    // =========================================
    // 3. Calculate y = A * x
    // =========================================

    std::vector<double> y = spmv(A, x);


    // =========================================
    // 4. Print result
    // =========================================

    std::cout << "y = ";

    for (double value : y)
    {
        std::cout << value << " ";
    }

    std::cout << "\n";


    return 0;
}