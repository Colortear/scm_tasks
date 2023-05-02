// originates from Ruetsch/Oster: Getting Started with CUDA
// more C++-style by Haase
#include <cassert>
#include <iostream>
#include <random>
#include <cstdio>

__global__ void cmp_gpu(bool *ret, float *const a, float *const b, int N)
{
    extern __shared__ bool  sdata[];

    const int   idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int   str = gridDim.x * blockDim.x;
    bool        cmp = true;

    for (int i = idx; i < N; i += str)
        if (a[i] != b[i])
            cmp = false;
    sdata[threadIdx.x] = cmp;
    if (!cmp)
        printf("yes\n");
    __syncthreads();
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x <= s && sdata[threadIdx.x] && !sdata[threadIdx.x + s])
            sdata[threadIdx.x] = sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        *ret = sdata[0];
}

using namespace std;

template <class T>
class CUDA_vec {
public:

    CUDA_vec() {};
    CUDA_vec(const int N, T *init)
    {
        cudaMallocManaged(&this->v, N * sizeof(T));
        for (int i = 0; i < N; i++)
            this->v[i] = init[i];
    }
    ~CUDA_vec() { cudaFree(v); }

    bool    cmp(const CUDA_vec &rhs, int const nb, int const bs, int const N)
    {
        bool    *result;
        bool    ret;

        cudaMallocManaged(&result, sizeof(bool));
        cmp_gpu<<<nb, bs>>>(result, this->v, rhs.v, N);
        cudaDeviceSynchronize();
        cout << cudaGetErrorName(cudaGetLastError()) << endl;
        ret = *result;
        cudaFree(result);
        return ret;
    }

    T   *v;
};

int     main(void)
{
    int const       N = 1400;
    int const       blockSize = 64;
    int const       numBlocks = (N+blockSize-1)/blockSize;
    float           init_data[N];
    CUDA_vec<float> V[3];
    int             rand_idx;


    srand(time(0));
    for (int i = 0; i < N; i++)
        init_data[i] = static_cast<float>(rand() % 100);
    for (int i = 0; i < 3; i++)
        V[i] = CUDA_vec<float>(N, init_data);
    rand_idx = rand() % N;
    V[3].v[rand_idx] = V[1].v[rand_idx];
    assert(V[1].cmp(V[2], numBlocks, blockSize, N));
    cout << "test 1 is ok." << endl;
    assert(!V[1].cmp(V[3], numBlocks, blockSize, N));
    cout << "test 2 is ok.";
    return 0;
}
