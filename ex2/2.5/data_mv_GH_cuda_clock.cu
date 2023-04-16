// originates from Ruetsch/Oster: Getting Started with CUDA
// more C++-style by Haase
#include <cassert>
#include <iostream>
#include <chrono>

__global__ void inc_gpu(float *const a, int N);

__global__ void inc_gpu(float *const a, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        a[idx] = a[idx] + 1;
}

__global__ void sum_gpu(float *const a, float *const b, float *const c, int N);

__global__ void sum_gpu(float *const a, float *const b, float *const c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        c[idx] = a[idx]+b[idx];
}

using namespace std;

int main(void)
{
    int const   N = 1400;
    int const   blockSize = 64;
    int const   numBlocks = (N+blockSize-1)/blockSize;
    float       *a, *b, *c;
    float       time_elapsed;
    cudaEvent_t start, stop;

    cudaMallocManaged(&a, N*sizeof(float));
    cudaMallocManaged(&b, N*sizeof(float));
    cudaMallocManaged(&c, N*sizeof(float));

    for (int i = 0; i < N; i++) {
        a[i] = 100.0f + static_cast<float>(i);
        b[i] = 100.0f + static_cast<float>(i);
        c[i] = 100.0f + static_cast<float>(i);
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    inc_gpu<<<numBlocks, blockSize>>>(b, N);
    cout << cudaGetErrorName(cudaGetLastError()) << endl;

    sum_gpu<<<numBlocks, blockSize>>>(a, b, c, N);
    cout << cudaGetErrorName(cudaGetLastError()) << endl;
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&time_elapsed, start, stop);
    cout << time_elapsed << " seconds since gpu start." << endl;

    for (int i = 0; i < N; i++)
        assert( a[i] == b[i] - 1.0f );
    cout << "Check 1  OK" << endl;
    for (int i = 0; i < N; i++)
        assert(c[i] == a[i] + b[i]);
    cout << "Check 2  OK" << endl;

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}
