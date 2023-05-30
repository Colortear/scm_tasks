#ifndef CUBLAS_VEC_CUH
# define CUBLAS_VEC_CUH

#include <iostream>
#include <ctime>
#include <cublas_v2.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))
#define COL_MAJOR 1
#define MATRIX 1
#define ROW_MAJOR 0

class   cuBLAS_Vec {
public:
    cuBLAS_Vec(int m, int n, int st);
    ~cuBLAS_Vec();

    void    print(std::string label);
    void    rand_fill();

    float       *v;
    const int   len, m, n;
    const int   storetype;

private:
    const int   _RAND_MAX = 10;
};

#endif
