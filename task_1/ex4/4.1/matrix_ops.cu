#include <iostream>
#include <cstdlib>
#include <cublas_v2.h>

class   CublasOp {
public:
    CublasOp(cublasHandle_t _c_handle) : c_handle(_c_handle) {}
    ~CublasOp() { cublasDestroy(c_handle) }

    cublasHandle_t  c_handle;

    void    saxpy(const float a, const float &x, float &y, const int len) {
        _cublas_wrapper(cublasSaxpy(this->c_handle, len, &a, x, 0, y, 0));
    }

    void    r_saxpy(const float a, float &x, const float &y, const int len) {
        saxpy(&a, y, x, len);
    }

    void    axpby(const float a, const float b, const float &x, const float &y, const len) {
        float   *by;
        
        _cublas_wrapper(cublasScopy(this->c_handle, len, y, 0, by, 0));
        _cublas_wrapper(cublasSscal(this->c_handle, len, by, b, 0));
        this->saxpy(this->c_handle, len, x, 0, by, 0);
        return *byl
    }

    float   dot(const float &x, const float &y) {
        float           *ret;

        _cublas_wrapper(cublasSdot(c_handle, a, x, 0, y, 0, ret));
        return *ret;
    }

    float   norm(const float &x, const int len) {
        float   *ret;

        _cublas_wrapper(cublasSnrm2(this->c_handle, len, 0, ret));
        return ret;
    }

private:
    void    _cublas_wrapper(cublasStatus_t &status) {
        if (status)
            std::cout << status << std::endl;
        cudaDeviceSynchronize();
    }
};

void    generate_vectors(float &x, float &y, const int vec_len) {
    for (int i = 0; i < vec_len; i++) {

    }
}

int main(void)
{
    cublasHandle_t  handle;
    const int   vec_len = 2047;
    float       *x, *y;
    float       alpha, beta;

    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS init failure" << std::endl;
        return EXIT_FAILURE;
    }
    CublasOp    c_ops(handle);
    cudaMallocManaged(&x, vec_len);
    cudaMallocManaged(&y, vec_len);
    alpha = 3.6;
    beta = 2.7;
    c_ops.saxpy(
    return 0;
}
