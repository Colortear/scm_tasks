#include <iostream>
#include <ctime>
#include <cublas_v2.h>
#include "cuBLAS_Op.cuh"

int     main(void)
{
    cublasHandle_t  handle;
    const int       n = 5;
    cuBLAS_Vec      M(n, n), A(n, n), x(n), y(n), z(n);

    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::cout << "cuBLAS init failure" << std::endl;
        return EXIT_FAILURE;
    }
    cuBLAS_Op   c_ops(handle);
    M.print("M:");
    c_ops.MMt(M, A);
    A.print("A:");
    x.print("x:");
    c_ops.Mx(A, x, z);
    z.print("z=Ax:");
    c_ops.Mtx(M, x, y);
    c_ops.Mx(M, y, y);
    y.print("y=M*(M^t*x):");
    return 0;
 }
