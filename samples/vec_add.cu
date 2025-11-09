#include <stdio.h>
#include <cuda_runtime.h>

// Kernel definition
__global__ void vecAdd(const float *A, const float *B, float *C) {
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

// Host code
int main() {
    constexpr int N = 5;

    // --- 1. Define Host Data ---
    float h_A[N] = {1, 2, 3, 4, 5};
    float h_B[N] = {10, 20, 30, 40, 50};
    float h_C[N] = {0};

    // --- 2. Allocate Device Memory ---
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));

    // --- 3. Copy Data from Host to Device ---
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // --- 4. Launch Kernel ---
    int threadsPerBlock = N;
    int blocksPerGrid = 1;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);

    // --- 5. Copy Result from Device to Host ---
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // --- 6. Copy Result from Device to Host ---
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    for (int i = 0; i < N; i++) {
        printf("%f ", h_C[i]);
    }
    printf("\n");

    // --- 7. Free Device Memory ---
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
