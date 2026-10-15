#include <iostream>
#include <vector>

int main()
{
    std::vector<double> A = {
        1, 2, 3, 4
    };

    std::vector<double> B = {
        10, 20, 30, 40
    };

    std::vector<double> C(4);

    for (int i = 0; i < 4; i++)
    {
        C[i] = A[i] + B[i];
    }

    for (int i = 0; i < 4; i++)
    {
        std::cout << C[i] << " ";
    }

    std::cout << std::endl;

    return 0;
}