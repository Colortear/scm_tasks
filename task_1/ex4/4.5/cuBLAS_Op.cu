#include <iostream>
#include <cublas_v2.h>
#include "cuBLAS_Op.cuh"

cuBLAS_Op::cuBLAS_Op(cublasHandle_t _c_handle) :
    c_handle(_c_handle)
{}

cuBLAS_Op::~cuBLAS_Op() { cublasDestroy(this->c_handle); }

void    cuBLAS_Op::saxpy(const float a, cuBLAS_Vec &x, cuBLAS_Vec &y)
{
    _cublas_wrapper(cublasSaxpy(this->c_handle, x.len, &a, x.v, 1, y.v, 1));
}

void    cuBLAS_Op::r_saxpy(const float a, cuBLAS_Vec &x, cuBLAS_Vec &y)
{
    this->saxpy(a, y, x);
}

void    cuBLAS_Op::axpby(const float a, const float b, cuBLAS_Vec &x,
        cuBLAS_Vec &y, cuBLAS_Vec &z)
{
    _cublas_wrapper(cublasScopy(this->c_handle, x.len, y.v, 1, z.v, 1));
    _cublas_wrapper(cublasSscal(this->c_handle, x.len, &b, z.v, 1));
    this->saxpy(1, x, z);
}

float   cuBLAS_Op::dot(cuBLAS_Vec &x, cuBLAS_Vec &y)
{
    float   ret;
    float   *tmp;

    cudaMallocManaged(&tmp, sizeof(float));
    _cublas_wrapper(cublasSdot(c_handle, x.len, x.v, 1, y.v, 1, tmp));
    ret = *tmp;
    cudaFree(tmp);
    return ret;
}

float   cuBLAS_Op::norm(cuBLAS_Vec &x)
{
    float   ret;
    float   *tmp;

    cudaMallocManaged(&tmp, sizeof(float));
    _cublas_wrapper(cublasSnrm2(this->c_handle, x.len, x.v, 1, tmp));
    ret = *tmp;
    cudaFree(tmp);
    return ret;
}

void    cuBLAS_Op::Mx(cuBLAS_Vec &M, cuBLAS_Vec &x, cuBLAS_Vec &r)
{
    _cublas_wrapper(cublasSgemv(this->c_handle, CUBLAS_OP_N, M.m, M.n,
                &this->_one, M.v, M.m, x.v, 1, &this->_zero, r.v, 1));
}

void    cuBLAS_Op::Mtx(cuBLAS_Vec &M, cuBLAS_Vec &x, cuBLAS_Vec &r)
{
    _cublas_wrapper(cublasSgemv(this->c_handle, CUBLAS_OP_T, M.m, M.n,
                &this->_one, M.v, M.m, x.v, 1, &this->_zero, r.v, 1));
}

void    cuBLAS_Op::tri_Mx(cuBLAS_Vec &M, cuBLAS_Vec &x, cuBLAS_Vec &r)
{
    _cublas_wrapper(cublasSsbmv(this->c_handle, CUBLAS_FILL_MODE_LOWER,
                M.n, 1, &this->_one, M.v, M.m, x.v, 1, &this->_zero, r.v, 1));
}

void    cuBLAS_Op::MMt(cuBLAS_Vec &M, cuBLAS_Vec &r)
{
    _cublas_wrapper(cublasSgemm(this->c_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                M.m, M.n, M.n, &this->_one, M.v, M.m, M.v, M.m, &this->_zero, r.v, r.m));
}

void    cuBLAS_Op::Diag_M(cuBLAS_Vec &M, cuBLAS_Vec &r)
{
    _cublas_wrapper(cublasScopy(this->c_handle, r.m, M.v, M.m+1, r.v, 1));
}

void    cuBLAS_Op::_cublas_wrapper(cublasStatus_t status)
{
    try {
        if (status)
            throw std::exception();
    }
    catch (std::exception &e) {
        std::cerr << cublasGetStatusName(status) << ": " << cublasGetStatusString(status) << std::endl;
    }
    cudaDeviceSynchronize();
}
