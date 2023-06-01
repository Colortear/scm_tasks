#ifndef CUBLAS_OP_CUH
# define CUBLAS_OP_CUH

# include "cuBLAS_Vec.cuh"

class   cuBLAS_Op {
public:
    cuBLAS_Op(cublasHandle_t _c_handle);
    ~cuBLAS_Op();

    void    saxpy(const float a, cuBLAS_Vec &x, cuBLAS_Vec &y);
    void    r_saxpy(const float a, cuBLAS_Vec &x, cuBLAS_Vec &y);
    void    axpby(const float a, const float b, cuBLAS_Vec &x, cuBLAS_Vec &y, cuBLAS_Vec &z);
    float   dot(cuBLAS_Vec &x, cuBLAS_Vec &y);
    float   norm(cuBLAS_Vec &x);
    void    Mx(cuBLAS_Vec &M, cuBLAS_Vec &x, cuBLAS_Vec &r);
    void    Mtx(cuBLAS_Vec &M, cuBLAS_Vec &x, cuBLAS_Vec &r);
    void    tri_Mx(cuBLAS_Vec &Mm, cuBLAS_Vec &x, const int band);

    cublasHandle_t  c_handle;

private:
    void    _cublas_wrapper(cublasStatus_t status);

    static constexpr float  _one = 1.f;
    static constexpr float  _zero = 0.f;
};

#endif
