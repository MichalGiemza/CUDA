// CUDA-01.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <windows.h>

#define N 40

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

static void HandleError(cudaError_t err, const char *file, int line) {
	//cudaError_t cudastatus = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		getchar();
		exit(EXIT_FAILURE);
	}
}

void MatPrint(float *V, int x, int y) {
	
	for (int i = 0; i < x * y; i++) {
		if (i % x == 0)
			putchar('\n');
		printf(" %6.1f", V[i]);
	}
	putchar('\n');
}

double getTime(LARGE_INTEGER start, LARGE_INTEGER end, LARGE_INTEGER countPerSec) {
	return (double)(end.QuadPart - start.QuadPart) / countPerSec.QuadPart * 1000;
}

__global__ void Tabliczka(float *A, float *B, float *M) {
	// x = threadIdx.x; y = blockIdx.x

	// Wektory do shared
	__shared__ float a[N], b[N];
	a[threadIdx.x] = A[threadIdx.x];
	b[threadIdx.x] = B[threadIdx.x];
	
	// Macierz
	M[threadIdx.x + blockIdx.x*N] = a[threadIdx.x] * b[blockIdx.x];
}

int _tmain(int argc, _TCHAR* argv[]) {
	size_t vs = N * sizeof(float);
	size_t ms = N * N * sizeof(float);
	cudaError_t e;

	e = cudaSetDevice(0);
	HANDLE_ERROR(e);

	float *h_A, *h_B, *h_M;

	h_A = (float *)malloc(vs);
	h_B = (float *)malloc(vs);
	h_M = (float *)malloc(ms);

	for (int i = 0; i < N; i++) {
		h_A[i] = i;
		h_B[i] = i;// * 2;
	}

	float *d_A, *d_B, *d_M;
    e = cudaMalloc(&d_A, vs);
	HANDLE_ERROR(e);
    e = cudaMalloc(&d_B, vs);
	HANDLE_ERROR(e);
    e = cudaMalloc(&d_M, ms);
	HANDLE_ERROR(e);

	e = cudaMemcpy(d_A, h_A, vs, cudaMemcpyHostToDevice);
	HANDLE_ERROR(e);
	e = cudaMemcpy(d_B, h_B, vs, cudaMemcpyHostToDevice);
	HANDLE_ERROR(e);
		
	dim3 block(N, 1, 1);
	dim3 grid(N, 1, 1);
	Tabliczka<<<grid, block>>>(d_A, d_B, d_M);
		
    e = cudaMemcpy(h_M, d_M, ms, cudaMemcpyDeviceToHost);
	HANDLE_ERROR(e);
	
	printf("Macierz wyjsciowa:");
	MatPrint(h_M, N, N);

	cudaFree(d_A);
    cudaFree(d_M);

	free(h_A);
	free(h_M);

	getchar();
	return 0;
}

