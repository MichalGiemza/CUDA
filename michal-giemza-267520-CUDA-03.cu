// CUDA-01.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <cuda_runtime.h>
#include <stdlib.h>

#define N 1024

__global__ void VecScalarProduct_SyncSum(float *A, float *B, float *SP) {
    //int x = blockIdx.x * blockDim.x + threadIdx.x;
    //int y = blockIdx.y * blockDim.y + threadIdx.y;
	register int i = threadIdx.x;

	__shared__ float mult[N];
	mult[i] = A[i] * B[i];

	__syncthreads();
	
	if (i == 0) {
		register float sum = 0;
		for (int j = 0; j < N; j++)
			sum += mult[j];

		*SP = sum;
	}
}

__global__ void VecScalarProduct_SyncSumNoReg(float *A, float *B, float *SP) {
    //int x = blockIdx.x * blockDim.x + threadIdx.x;
    //int y = blockIdx.y * blockDim.y + threadIdx.y;
	register int i = threadIdx.x;

	__shared__ float mult[N];
	mult[i] = A[i] * B[i];

	__syncthreads();
	
	if (i == 0) {
		*SP = 0;
		for (int j = 0; j < N; j++)
			*SP += mult[j];
	}
}

__global__ void VecScalarProduct(float *A, float *B, float *SP) {
    //int x = blockIdx.x * blockDim.x + threadIdx.x;
    //int y = blockIdx.y * blockDim.y + threadIdx.y;
	register int i = threadIdx.x;

	__shared__ float mult[N];
	mult[i] = A[i] * B[i];

	__syncthreads();
	
	for (int k = 2; k <= N; k *= 2)
	{   // iteracje
		__syncthreads();
		if (i % k == 0)
		{   // po tablicy
			mult[i] += mult[i + k/2];
		}
	}
	__syncthreads();
	*SP = mult[0];
}

void VecPrint(float *V, int n) {
	printf("[ ");
	for (int i = 0; i < n; i++) {
		printf("%2.0f ", V[i]);
	}
	printf("]\n");
}

int _tmain(int argc, _TCHAR* argv[]) {
    size_t size = N * sizeof(float);

	float *h_A, *h_B, h_SP;
	h_A = (float *)malloc(size);
	h_B = (float *)malloc(size);
	for (int i = 0; i < N; i++) {
		h_A[i] = 1;
		h_B[i] = 1;
	}

	float *d_A, *d_B, *d_SP;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_SP, sizeof(float));

	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	
	dim3 block(N, 1, 1);
	dim3 grid(1, 1, 1);

    VecScalarProduct<<<grid, block>>>(d_A, d_B, d_SP);

    cudaMemcpy(&h_SP, d_SP, sizeof(float), cudaMemcpyDeviceToHost);
	
	printf("A = [1, 1, ..., 1], B = [1, 1, ..., 1]\n");
	printf("Iloczyn skalarny: %f", h_SP);
	
	cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_SP);

	free(h_A);
	free(h_B);

	getchar();
	return 0;
}

