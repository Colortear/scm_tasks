#include <iostream>
#include <ctime>
#include <cublas_v2.h>
#include <algorithm>
#include "cuBLAS_Vec.cuh"

cuBLAS_Vec::cuBLAS_Vec(int m, int n = 1, int st = ROW_MAJOR, bool zero_fill = false) :
    len(m*n),
    m(m),
    n(n),
    storetype(st)
{
    cudaMallocManaged(&this->v, len * sizeof(float));
    if (zero_fill)
        std::for_each(this->v, this->v + this->len, [=](auto x){ return 0.f; });
    else
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
    srand(time(0));
    for (int i = 0; i < len; i++)
        this->v[i] = rand() % this->_RAND_MAX;
}

void    cuBLAS_Vec::tridiag_toe(int c, int d, int e) // assumed COL_MAJOR
{
    for (int i = -1; i < this->len; i+=this->m+1) {
        if (i >= 0)
            this->v[i] = c;
        if (i+1<this->len)
            this->v[i+1] = d;
        if (i+2<this->len)
            this->v[i+2] = e;
    }
}
