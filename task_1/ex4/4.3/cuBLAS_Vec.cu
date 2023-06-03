#include <iostream>
#include <ctime>
#include <cublas_v2.h>
#include <algorithm>
#include "cuBLAS_Vec.cuh"

Banded::Banded(const float *r, const int dim) : v(r), dim(dim) {}

cuBLAS_Vec::cuBLAS_Vec(int m, int n = 1, int st = ROW_MAJOR, bool zero_fill = false) :
    len(m*n),
    m(m),
    n(n),
    storetype(st)
{
    cudaMallocManaged(&this->v, this->len * sizeof(float));
    if (zero_fill)
        cudaMemset(this->v, 0, this->len * sizeof(float));
    else
        this->rand_fill();
}

cuBLAS_Vec::cuBLAS_Vec(Banded t, int n) :
    len(n*t.dim),
    m(t.dim),
    n(n),
    storetype(COL_MAJOR)
{
    cudaMallocManaged(&this->v, this->len * sizeof(float));
    cudaMemset(this->v, 0, this->len * sizeof(float));
    for (int i = 0; i < this->len-1; i++)
        if (i % n <= n -(i/n))
            this->v[IDX2C(i/n, i%n, this->m)] = t.v[i/n];
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
    srand(time(0));
    for (int i = 0; i < len; i++)
        this->v[i] = rand() % this->_RAND_MAX;
}
