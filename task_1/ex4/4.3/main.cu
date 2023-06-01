#include <iostream>
#include <ctime>
#include <cublas_v2.h>
#include "cuBLAS_Op.cuh"

int     main(void)
{
    cublasHandle_t  handle;
    const int       m = 3;
    const int       n = 3;
    cuBLAS_Vec      x(n, 1, COL_MAJOR, false);
    cuBLAS_Vec      T(m, n, MATRIX, true);

    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::cout << "cuBLAS init failure" << std::endl;
        return EXIT_FAILURE;
    }
    cuBLAS_Op   c_ops(handle);
    T.tridiag_toe(-1, 2, -1);
    T.print("T:");
    return 0;
 }
