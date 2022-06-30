#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Mask param not needed as it is fixed for all inputs.
__global__ void
row_convolution(float* mat_input1, float* mat_output1, int mat_datasize, int mat_dim)
{
    int blockNum = blockIdx.x;
    int threadNum = threadIdx.x;
    int globalThreadId = blockNum * (blockDim.x) + threadNum;

    int row_index = globalThreadId / mat_dim;
    int col_index = globalThreadId % mat_dim;

    if (row_index < mat_dim && col_index < mat_dim)
    {
        float leftVal = col_index - 1 < 0 ? 0.0f : mat_input1[globalThreadId - 1]; // padding
        float rightVal = col_index + 1 >= mat_dim ? 0.0f : mat_input1[globalThreadId + 1];
        mat_output1[globalThreadId] = (leftVal * -1) + (mat_input1[globalThreadId] * 2.5) + (rightVal * -1);
    }
}

__global__ void
col_convolution(float* mat_input1, float* mat_output1, int mat_datasize, int mat_dim)
{
    int blockNum = blockIdx.x;
    int threadNum = threadIdx.x;
    int globalThreadId = blockNum * (blockDim.x) + threadNum;

    int row_index = globalThreadId / mat_dim;
    int col_index = globalThreadId % mat_dim;

    if (row_index < mat_dim && col_index < mat_dim)
    {
        float topVal = row_index - 1 < 0 ? 0.0f : mat_input1[(mat_dim * (row_index - 1)) + col_index];
        float bottomVal = row_index + 1 >= mat_dim ? 0.0f : mat_input1[(mat_dim * (row_index + 1)) + col_index];
        mat_output1[globalThreadId] += (topVal * -1) + mat_input1[globalThreadId] * 2.5 + (bottomVal * -1);
    }
}

void print_matrix(float* A, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
            printf("%.1f ", A[i * n + j]); // single decimal precision as specified
        printf("\n");
    }

}

int main(void)
{
    cudaError_t err = cudaSuccess;
    
    int i, j, k;
    int testCases;
    scanf("%d", &testCases);
    int testCasesOriginal = testCases; // used at end to print results

    // using two primitive arrays instead of one struct array to store results
    float **results = new float *[testCases];
    int *dims = new int[testCases];

    while (testCases--)
    {
        int mat_dim;
        scanf("%d", &mat_dim);
        int mat_num_eles = mat_dim * mat_dim;
        size_t mat_size = mat_num_eles * sizeof(float);

        /* populate code for allocating host memory */
        float *h_mat_input1 = (float*)malloc(mat_size);
        float *h_mat_output1 = (float*)malloc(mat_size);

        int mat_conv_dim = 3;
        int mat_conv_num_eles = mat_conv_dim*mat_conv_dim;
        size_t mat_conv_size = mat_conv_num_eles*sizeof(float);

        if (h_mat_input1 == NULL || h_mat_output1 == NULL)
        {
            fprintf(stderr, "Failed to allocate host vectors!\n");
            exit(EXIT_FAILURE);
        }

        for (i = 0; i < mat_num_eles; i++)
        {
            j = i / mat_dim;
            k = i % mat_dim;

            scanf("%f", &h_mat_input1[mat_dim * j + k]);
        }

        float *d_mat_input1 = NULL;
        err = cudaMalloc((void **)&d_mat_input1, mat_size);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector d_mat_input1 (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        float *d_mat_output1 = NULL;
        err = cudaMalloc((void **)&d_mat_output1, mat_size);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to allocate device vector d_mat_output1 (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }


        err = cudaMemcpy(d_mat_input1, h_mat_input1, mat_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector h_mat_input1 from host to device (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        int mat_block_dim = 16;
        int mat_grid_dim = mat_num_eles / mat_block_dim > 0 ? ceilf((float)mat_num_eles / mat_block_dim) : 1;
        row_convolution<<<mat_grid_dim, mat_block_dim>>>(d_mat_input1, d_mat_output1, mat_size, mat_dim);
        col_convolution<<<mat_grid_dim, mat_block_dim>>>(d_mat_input1, d_mat_output1, mat_size, mat_dim);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch process_kernel2 kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }


        err = cudaMemcpy(h_mat_output1, d_mat_output1, mat_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to copy vector d_mat_output1 from device to host (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }


        err = cudaFree(d_mat_input1);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector d_mat_input1 (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        err = cudaFree(d_mat_output1);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to free device vector d_mat_output1 (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        free(h_mat_input1);
        results[testCases] = h_mat_output1;
        dims[testCases] = mat_dim;

        err = cudaDeviceReset();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

    }

    // print the results and then free them
    for (int i = testCasesOriginal-1; i >= 0; --i)
    {
        print_matrix(results[i], dims[i], dims[i]);
        free(results[i]);
    }

    return 0;
}
