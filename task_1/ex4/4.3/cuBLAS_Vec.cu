#include <iostream>
#include <ctime>
#include <cublas_v2.h>
#include "cuBLAS_Vec.cuh"

cuBLAS_Vec::cuBLAS_Vec(int m, int n = 1, int st = ROW_MAJOR) :
    len(m*n),
    m(m),
    n(n),
    storetype(st)
{
    cudaMallocManaged(&this->v, len * sizeof(float));
    this->rand_fill();
}

cuBLAS_Vec::~cuBLAS_Vec()
{
    cudaFree(this->v);
}

void    cuBLAS_Vec::print(std::string label = "")
{
    std::cout << label << std::endl;
    if (this->storetype == ROW_MAJOR) {
        for (int i = 0; i < this->len; i++)
            std::cout << this->v[i] <<  " ";
        std::cout << std::endl;
        return ;
    }
    for (int i = 0; i < this->m; i++) {
        for (int j = 0; j < this->n; j++)
            std::cout << this->v[IDX2C(i, j, this->m)] << " ";
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void    cuBLAS_Vec::rand_fill()
{
    for (int i = 0; i < len; i++)
        this->v[i] = rand() % this->_RAND_MAX;
}
