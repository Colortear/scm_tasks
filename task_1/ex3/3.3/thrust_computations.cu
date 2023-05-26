#include <cassert>
#include <iostream>
#include <random>
#include <cstdio>
#include <thrust/reduce.h>

struct  exp_gpu {
    __host__ __device__ float operator()(const float &a) const {
        return expf(a);
    }
};

struct  inc_gpu {
    __host__ __device__ float operator()(const float &a) const {
        return a+1;
    }
};

struct  ln_gpu {
    __host__ __device__ float operator()(const float &a) const {
        return logf(a);
    }
};

struct  sum_gpu {
    __host__ __device__ float operator()(const float &a, const float &b) const {
        return a + b;
    }
};

using namespace std;

int     main(void)
{
    const int   N = 1000000;
    float       *v[4];
    vector<float>   t(N);

    srand(time(0));
    for (int i = 0; i < 3; i++) {
        cudaMallocManaged(&v[i], N * sizeof(float));
        for (int j = 0; j < N; j++)
            v[i][j] = static_cast<float>(rand() % 100);
    }
    cudaMallocManaged(&v[3], N * sizeof(float));
    for (int i = 0; i < N; i++)
        v[3][i] = v[2][i];
    thrust::transform(v[3], v[3]+N, v[3], exp_gpu());
    thrust::transform(v[3], v[3]+N, v[3], ln_gpu());
    //these mostly return to the same values excluding some cases.
    transform(v[0], v[0]+N, v[1], t.begin(), plus<float>());
    thrust::transform(v[0], v[0]+N, v[1], v[1], sum_gpu());
    for (int i = 0; i < N; i++)
        assert(t[i] == v[1][i]);
    cout << "sum test ok" << endl;
    for (int i = 0; i < N; i++)
        t[i] = v[2][i];
    thrust::transform(v[2], v[2]+N, v[2], inc_gpu());
    for (int i = 0; i < N; i++)
        assert(t[i]+1 == v[2][i]);
    cout << "increment test ok" << endl;
    for (int i = 0; i < 4; i++)
        cudaFree(v[i]);
    return 0;
}
