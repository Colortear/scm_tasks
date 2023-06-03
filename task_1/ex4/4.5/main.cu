#include <iostream>
#include <ctime>
#include <cublas_v2.h>
#include "cuBLAS_Op.cuh"

int     main(void)
{
    cublasHandle_t  handle;
    const int       m = 6, n = 5;
    cuBLAS_Vec      M(m, n), d(n);

    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::cout << "cuBLAS init failure" << std::endl;
        return EXIT_FAILURE;
    }
    cuBLAS_Op   c_ops(handle);
    M.print("M:");
    c_ops.Diag_M(M, d);
    d.print("diag(M):");
    return 0;
 }
