#include <iostream>
#include <ctime>
#include <cublas_v2.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define COL_MAJOR 1
#define MATRIX 1
#define ROW_MAJOR 0

class   cuBLAS_Vec {
public:
    cuBLAS_Vec(int m, int n = 1, int st = ROW_MAJOR) :
        len(m*n), m(m), n(n), storetype(st)
    {
        cudaMallocManaged(&this->v, len * sizeof(float));
        this->rand_fill();
    }
    ~cuBLAS_Vec() { cudaFree(this->v); }

    void    print(std::string label = "") {
        std::cout << label << std::endl;
        if (storetype == ROW_MAJOR) {
            for (int i = 0; i < this->m; i++)
                std::cout << this->v[i] <<  " ";
            std::cout << std::endl;
            return ;
        }
        for (int i = 0; i < this->m; i++) {
            for (int j = 0; j < this->n; j++)
                std::cout << this->v[IDX2C(i, j, i+1)] << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    void    rand_fill() {
        for (int i = 0; i < len; i++)
            this->v[i] = rand() % this->_RAND_MAX;
    }

    float       *v;
    const int   len, m, n;
    const int   storetype;

private:
    const int   _RAND_MAX = 10;
};

class   CublasOp {
public:
    CublasOp(cublasHandle_t _c_handle) : c_handle(_c_handle) {}
    ~CublasOp() { cublasDestroy(c_handle); }

    cublasHandle_t  c_handle;

    void    saxpy(const float a, cuBLAS_Vec &x, cuBLAS_Vec &y) {
        _cublas_wrapper(cublasSaxpy(this->c_handle, x.len, &a, x.v, 1, y.v, 1));
    }

    void    r_saxpy(const float a, cuBLAS_Vec &x, cuBLAS_Vec &y) {
        this->saxpy(a, y, x);
    }

    void    axpby(const float a, const float b, cuBLAS_Vec &x, cuBLAS_Vec &y, cuBLAS_Vec &z) {
        _cublas_wrapper(cublasScopy(this->c_handle, x.len, y.v, 1, z.v, 1));
        _cublas_wrapper(cublasSscal(this->c_handle, x.len, &b, z.v, 1));
        this->saxpy(1, x, z);
    }

    float   dot(cuBLAS_Vec &x, cuBLAS_Vec &y) {
        float   ret;
        float   *tmp;

        cudaMallocManaged(&tmp, sizeof(float));
        _cublas_wrapper(cublasSdot(c_handle, x.len, x.v, 1, y.v, 1, tmp));
        ret = *tmp;
        cudaFree(tmp);
        return ret;
    }

    float   norm(cuBLAS_Vec &x) {
        float   ret;
        float   *tmp;

        cudaMallocManaged(&tmp, sizeof(float));
        _cublas_wrapper(cublasSnrm2(this->c_handle, x.len, x.v, 1, tmp));
        ret = *tmp;
        cudaFree(tmp);
        return ret;
    }

    void    Mx(cuBLAS_Vec &M, cuBLAS_Vec &x, cuBLAS_Vec &r) {
        float   one = 1.f;
        float   zero = 0.f;

        _cublas_wrapper(cublasSgemv(this->c_handle, CUBLAS_OP_N, M.m, M.n,
                                    &one, M.v, M.m, x.v, 1,
                                    &zero, r.v, 1));
    }

    void    Mtx(cuBLAS_Vec &M, cuBLAS_Vec &x, cuBLAS_Vec &r) {
        float   one = 1.f;
        float   zero = 0.f;
        
        _cublas_wrapper(cublasSgemv(this->c_handle, CUBLAS_OP_T, M.m, M.n,
                                    &one, M.v, M.m, x.v, 1,
                                    &zero, r.v, 1));
    }

private:
    void    _cublas_wrapper(cublasStatus_t status) {
        if (status)
            std::cout << status << std::endl;
        cudaDeviceSynchronize();
    }
};

int     main(void)
{
    cublasHandle_t  handle;
    const int       m = 2;
    const int       n = 3;
    float           alpha, beta;
    cuBLAS_Vec      x1(n, 1, COL_MAJOR), x2(n, 1, COL_MAJOR), r(m, 1, COL_MAJOR);
    cuBLAS_Vec      A(m, n, MATRIX);

    srand(std::time(0));
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::cout << "cuBLAS init failure" << std::endl;
        return EXIT_FAILURE;
    }
    CublasOp    c_ops(handle);
    alpha = 3.f;
    beta = 2.f;
    x1.print("x1:");
    x2.print("x2:");
    A.print("A:");
    c_ops.Mx(A, x1, r);
    r.print("r:");
    return 0;
}
