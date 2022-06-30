#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define LOG_INPUT if (0)
#define LOG_OUTPUT if (0)
#define LOG if (0)

__global__ void hadamard(float* A, float* B, float* C, int M, int N)
{
    // Complete the kernel code snippet
    int row = (blockIdx.y * blockDim.y) + threadIdx.y;
    int col = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (row < M && col < N)
    {
        int globalThreadId = col + (row * (gridDim.x * blockDim.x));
        C[globalThreadId] = A[globalThreadId] * B[globalThreadId];
    }
}

/**
 * Host main routine
 */
void print_matrix(float* A, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
            printf("%.2f ", A[i * n + j]);
        printf("\n");
    }
}

int main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int t; // number of test cases
    scanf("%d", &t);
    while (t--)
    {
        int m, n;
        scanf("%d %d", &m, &n);
        size_t size = m * n * sizeof(float);
        LOG printf("[Hadamard product of two matrices ]\n");

        // Allocate the host input vector A
        float* h_A = (float*)malloc(size);

        // Allocate the host input vector B
        float* h_B = (float*)malloc(size);

        // Allocate the host output vector C
        float* h_C = (float*)calloc(size, sizeof(float));

        // Verify that allocations succeeded
        if (h_A == NULL || h_B == NULL || h_C == NULL)
        {
            fprintf(stderr, "Failed to allocate host vectors!\n");
            exit(EXIT_FAILURE);
        }

        // Initialize the host input vectors
        for (int i = 0; i < n * m; ++i)
            scanf("%f", &h_A[i]);
        for (int i = 0; i < n * m; ++i)
            scanf("%f", &h_B[i]);

        // Allocate the device input vector A
        float* d_A = NULL;
        cudaMalloc(&d_A, size);

        // Allocate the device input vector B
        float* d_B = NULL;
        cudaMalloc(&d_B, size);

        // Allocate the device output vector C
        float* d_C = NULL;
        cudaMalloc(&d_C, size);

        // Copy the host input vectors A and B in host memory to the device input vectors in
        // device memory
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

        // initialize blocksPerGrid and threads Per Block
        dim3 blocksPerGrid(ceil((float)m / 32), ceil((float)n / 32), 1);
        dim3 threadsPerBlock(32, 32, 1);
        hadamard<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, m, n);
        err = cudaGetLastError();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Copy the device result vector in device memory to the host result vector
        // in host memory.
        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
        LOG printf("\nRESULT:\n");
        LOG print_matrix(h_C, m, n);

        // Verify that the result vector is correct
        for (int i = 0; i < n * m; ++i)
        {
            if (fabs(h_A[i] * h_B[i] - h_C[i]) > 1e-5)
            {
                LOG fprintf(stderr, "Result verification failed at element %d!\n", i);
                fprintf(stderr, "Expected: %f, Got: %f!\n", h_A[i] * h_B[i], h_C[i]);
                exit(EXIT_FAILURE);
            }
        }

        LOG printf("Test PASSED\n");

        // Free device global memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        // Free host memory
        free(h_A);
        free(h_B);

        err = cudaDeviceReset();

        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        print_matrix(h_C, m, n);
        free(h_C);

        LOG printf("Done\n");
    }
    return 0;
}
