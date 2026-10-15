#include <iostream>

int main()
{
    const int N = 3;

    double A[N][N] =
    {
        { 4, -1,  0 },
        {-1,  4, -1 },
        { 0, -1,  4 }
    };

    double x[N] = {
        1, 2, 3
    };

    double y[N] = {};

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            y[i] += A[i][j] * x[j];
        }
    }

    for (int i = 0; i < N; i++)
    {
        std::cout << y[i] << std::endl;
    }

    return 0;
}