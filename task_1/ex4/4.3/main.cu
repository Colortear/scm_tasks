#include <iostream>
#include <ctime>
#include <cublas_v2.h>
#include "cuBLAS_Op.cuh"

int     main(void)
{
    cublasHandle_t  handle;
    const int       n = 3;
    float           t[2];
    cuBLAS_Vec      x(n, 1, COL_MAJOR, false), y(n, 1, COL_MAJOR, false);

    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::cout << "cuBLAS init failure" << std::endl;
        return EXIT_FAILURE;
    }
    cuBLAS_Op   c_ops(handle);
    t[0] = 2;
    t[1] = -1;
    cuBLAS_Vec  T(Banded(t, 2), n);
    T.print("T:");
    x.print("x:");
    c_ops.tri_Mx(T, x, 1, y);
    y.print("Tx=");
    return 0;
 }
