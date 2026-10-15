#include <iostream>
#include <vector>

double dot(
    const std::vector<double>& A,
    const std::vector<double>& B
)
{
    double result = 0.0;

    for (size_t i = 0; i < A.size(); i++)
    {
        result += A[i] * B[i];
    }

    return result;
}

int main()
{
    std::vector<double> A = {
        1, 2, 3
    };

    std::vector<double> B = {
        4, 5, 6
    };

    double result = dot(A, B);

    std::cout << "Dot product = "
              << result
              << std::endl;

    return 0;
}