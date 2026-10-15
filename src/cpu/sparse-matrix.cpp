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

int main()
{
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

    std::cout << "values: ";

    for (double value : A.values)
    {
        std::cout << value << " ";
    }

    std::cout << "\n";

    std::cout << "colIndex: ";

    for (int col : A.colIndex)
    {
        std::cout << col << " ";
    }

    std::cout << "\n";

    std::cout << "rowPtr: ";

    for (int row : A.rowPtr)
    {
        std::cout << row << " ";
    }

    std::cout << "\n";

    return 0;
}