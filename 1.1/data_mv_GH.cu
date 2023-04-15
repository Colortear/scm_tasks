// originates from Ruetsch/Oster: Getting Started with CUDA
// more C++-style by Haase
#include <cassert>
#include <iostream>

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
    int const N = 1400;
    int const nBytes = N * sizeof(float);

    int const blockSize = 64;
    int const numBlocks = (N + blockSize - 1) / blockSize;


    float *a_h = new float [nBytes];     // host data
    float *b_h = new float [nBytes];     // host data
    float *c_h = new float [nBytes];
    float *a_d, *b_d, *c_d;                    // device data
    cudaMalloc((void **) &a_d, nBytes);
    cudaMalloc((void **) &b_d, nBytes);
    cudaMalloc((void **) &c_d, nBytes);

    for (int i = 0; i < N; i++)
        a_h[i] = 100.0f + static_cast<float>(i);

    cudaMemcpy(a_d, a_h, nBytes, cudaMemcpyHostToDevice);    //  a_d <- a_h
    cudaMemcpy(b_d, a_d, nBytes, cudaMemcpyDeviceToDevice);  //  b_d <- a_d
    cudaMemcpy(c_d, a_d, nBytes, cudaMemcpyDeviceToDevice);

// ---------------------------------------------------------

// Manipulate on GPU
    inc_gpu <<< numBlocks, blockSize>>>(b_d, N);             //  b_d := b_d+1.0    
    cout << cudaGetErrorName(cudaGetLastError()) << endl;

    sum_gpu <<< numBlocks, blockSize>>>(a_d,b_d,c_d,N);
    cout << cudaGetErrorName(cudaGetLastError()) << endl;

    cudaMemcpy(b_h, b_d, nBytes, cudaMemcpyDeviceToHost);    //  b_h <- b_d
    cudaMemcpy(c_h, c_d, nBytes, cudaMemcpyDeviceToHost);

// Check on CPU
    for (int i = 0; i < N; i++)
        assert( a_h[i] == b_h[i] - 1.0f );
    cout << "Check 1  OK" << endl;
    for (int i = 0; i < N; i++)
        assert(c_h[i] == a_h[i] + b_h[i]);
    cout << "Check 2  OK" << endl;
// ---------------------------------------------------------

    delete [] b_h;
    delete [] a_h;
    cudaFree(a_d);
    cudaFree(b_d);

    return 0;
}
