#include <stdio.h>
#include <cuda_runtime.h>

#define N 16

// Kernel definition
__global__ void matAdd(const float A[N][N], const float B[N][N], float C[N][N]) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}

// Host code
int main() {
    float A[N][N], B[N][N], C[N][N];
    float *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(float);

    // Initialize matrices A and B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = static_cast<float>(i + j);
            B[i][j] = static_cast<float>(i - j);
        }
    }

    // Allocate device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy matrices A and B to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 threadsPerBlock(N, N);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    // Launch kernel
    matAdd<<<numBlocks, threadsPerBlock>>>( (float (*)[N])d_A, (float (*)[N])d_B, (float (*)[N])d_C );

    // Copy result matrix C back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Print result matrix C
    printf("Result matrix C:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%4.1f ", C[i][j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}