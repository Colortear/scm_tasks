// originates from Ruetsch/Oster: Getting Started with CUDA
// more C++-style by Haase
#include <cassert>
#include <iostream>
#include <random>

struct vecs {
    float   *a, *b, *c;
};

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

void    prepare_data(vecs &V, int N)
{
    int rand_idx;

    cudaMallocManaged(&V.a, N * sizeof(float));
    cudaMallocManaged(&V.b, N * sizeof(float));
    cudaMallocManaged(&V.c, N * sizeof(float));
    srand(time(0));
    for (int i = 0; i < N; i++) {
        V.a[i] = static_cast<float>(rand() % 100);
        V.b[i] = V.a[i];
        V.c[i] = V.a[i];
    }
    rand_idx = rand() % N;
    V.c[rand_idx] = V.a[rand_idx] + 1;
    cout << "V.c[rand_idx] = " << V.c[rand_idx] << endl;
    cout << "V.a[rand_idx] = " << V.a[rand_idx] << endl;
    cout << rand_idx << endl;
}

void    compare_vec_gpu(float *x, float *y, int const nb, int const bs, int N, bool desired_res)
{
    static int  test_num = 0;
    bool        *result;

    cudaMallocManaged(&result, sizeof(bool));
    cmp_gpu<<<nb, bs>>>(result, x, y, N);
    cudaDeviceSynchronize();
    cout << cudaGetErrorName(cudaGetLastError()) << endl;
    assert(*result == desired_res);
    cout << "test " << ++test_num << " is ok." << endl;
    cudaFree(result);
}

void    free_vecs(vecs &V)
{
    cudaFree(V.a);
    cudaFree(V.b);
    cudaFree(V.c);
}

int     main(void)
{
    int const   N = 1400;
    int const   blockSize = 64;
    int const   numBlocks = (N+blockSize-1)/blockSize;
    vecs        V;

    prepare_data(V, N);
    compare_vec_gpu(V.a, V.b, numBlocks, blockSize, N, true);
    compare_vec_gpu(V.a, V.c, numBlocks, blockSize, N, false);
    free_vecs(V);
    return 0;
}
