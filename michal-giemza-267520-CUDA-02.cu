// CUDA-01.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <cuda_runtime.h>
#include <stdlib.h>

#define N 32

__global__ void VecAdd(float *A, float *B, float *C) {
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
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

	float *h_A, *h_B, *h_C;
	h_A = (float *)malloc(size);
	h_B = (float *)malloc(size);
	h_C = (float *)malloc(size);
	for (int i = 0; i < N; i++) {
		h_A[i] = i;
		h_B[i] = N - i;
	}

	float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	
    VecAdd<<<1, N>>>(d_A, d_B, d_C);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	
	printf("A: ");
	VecPrint(h_A, N);
	printf("\nB: ");
	VecPrint(h_B, N);
	printf("\nC: ");
	VecPrint(h_C, N);
	
	cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

	free(h_A);
	free(h_B);
	free(h_C);

	getchar();
	return 0;
}

