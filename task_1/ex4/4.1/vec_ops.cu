#include <iostream>
#include <cstdlib>
#include <cublas_v2.h>

class   CublasOp {
public:
    CublasOp(cublasHandle_t _c_handle) : c_handle(_c_handle) {}
    ~CublasOp() { cublasDestroy(c_handle); }

    cublasHandle_t  c_handle;

    void    saxpy(const float a, const float *x, float *y, const int len) {
        _cublas_wrapper(cublasSaxpy(this->c_handle, len, &a, x, 1, y, 1));
    }

    void    r_saxpy(const float a, float *x, const float *y, const int len) {
        this->saxpy(a, y, x, len);
    }

    void    axpby(const float a, const float b, const float *x, const float *y, float *z, const int len) {
        _cublas_wrapper(cublasScopy(this->c_handle, len, y, 1, z, 1));
        _cublas_wrapper(cublasSscal(this->c_handle, len, &b, z, 1));
        this->saxpy(1, x, z, len);
    }

    float   dot(const float *x, const float *y, const int len) {
        float   ret;
        float   *tmp;

        cudaMallocManaged(&tmp, sizeof(float));
        _cublas_wrapper(cublasSdot(c_handle, len, x, 1, y, 1, tmp));
        ret = *tmp;
        cudaFree(tmp);
        return ret;
    }

    float   norm(const float *x, const int len) {
        float   ret;
        float   *tmp;

        cudaMallocManaged(&tmp, sizeof(float));
        _cublas_wrapper(cublasSnrm2(this->c_handle, len, x, 1, tmp));
        ret = *tmp;
        cudaFree(tmp);
        return ret;
    }

private:
    void    _cublas_wrapper(cublasStatus_t status) {
        if (status)
            std::cout << status << std::endl;
        cudaDeviceSynchronize();
    }
};

void    generate_vector(float *x, const int vec_len) {
    for (int i = 0; i < vec_len; i++)
        x[i] = rand() % 10;
}

void    print_vector(std::string name, const float *x, int len) {
    std::cout << name << ": ";
    for (int i = 0; i < len; i++)
        std::cout << x[i] << " ";
    std::cout << "..." << std::endl;
}

int main(void)
{
    cublasHandle_t  handle;
    const int   vec_len = 2047;
    const int   print_len = 10;
    float       *x, *y, *z;
    float       alpha, beta;

    srand(time(0));
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS init failure" << std::endl;
        return EXIT_FAILURE;
    }
    CublasOp    c_ops(handle);
    cudaMallocManaged(&x, vec_len * sizeof(float));
    cudaMallocManaged(&y, vec_len * sizeof(float));
    cudaMallocManaged(&z, vec_len * sizeof(float));
    generate_vector(x, vec_len);
    generate_vector(y, vec_len);
    alpha = 3.f;
    beta = 2.f;
    std::cout << "a = " << alpha << ", " << "b = " << beta << std::endl;
    print_vector("x", x, print_len);
    print_vector("y", y, print_len);
    c_ops.saxpy(alpha, x, y, vec_len);
    print_vector("y = ax + y", y, print_len);
    c_ops.r_saxpy(alpha, x, y, vec_len);
    print_vector("x = ax + y", x, print_len);
    c_ops.axpby(alpha, beta, x, y, z, vec_len);
    print_vector("z = ax + by", z, print_len);
    std::cout << "<x, y> = " << c_ops.dot(x, y, vec_len) << std::endl;
    std::cout << "||x|| = " << c_ops.norm(x, vec_len) << std::endl;
    cublasDestroy(handle);
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    return 0;
}
