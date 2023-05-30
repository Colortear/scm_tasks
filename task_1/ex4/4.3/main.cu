#include <iostream>
#include <ctime>
#include <cublas_v2.h>
#include "cuBLAS_Vec.cuh"
#include "cuBLAS_Op.cuh"

int     main(void)
{
    cublasHandle_t  handle;
    const int       m = 3;
    const int       n = 3;
    float           alpha, beta;
    cuBLAS_Vec      x1(n, 1, COL_MAJOR), x2(n, 1, COL_MAJOR), r(m, 1, COL_MAJOR);
    cuBLAS_Vec      A(m, n, MATRIX);

    srand(std::time(0));
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::cout << "cuBLAS init failure" << std::endl;
        return EXIT_FAILURE;
    }
    cuBLAS_Op   c_ops(handle);
    alpha = 3.f;
    beta = 2.f;
    x1.print("x1:");
    x2.print("x2:");
    A.print("A:");
    c_ops.Mx(A, x1, r);
    r.print("A*x1:");
    c_ops.Mtx(A, x2, r);
    r.print("A*x2:");
    return 0;
}
