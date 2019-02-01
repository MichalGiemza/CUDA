// CUDA-01.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <windows.h>
#include "cublas_v2.h"

#define N 256

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

static void HandleError(cudaError_t err, const char *file, int line) {
	//cudaError_t cudastatus = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		getchar();
		exit(EXIT_FAILURE);
	}
}

void MatPrint(float *V, int n) {
	
	printf("\n(tylko pierwszy wiersz)");
	
	for (int i = 0; i < n; i++) {
		if (i % n == 0)
			putchar('\n');
		printf(" %5.1f", V[i]);
	}
	putchar('\n');
}

double getTime(LARGE_INTEGER start, LARGE_INTEGER end, LARGE_INTEGER countPerSec) {
	return (double)(end.QuadPart - start.QuadPart) / countPerSec.QuadPart * 1000;
}

int _tmain(int argc, _TCHAR* argv[]) {
	size_t size = N * N * sizeof(float);
	cudaError_t e;

	float *h_A, *h_B, *h_M;

	h_A = (float *)malloc(size);
	h_B = (float *)malloc(size);
	h_M = (float *)malloc(size);

	for (int i = 0; i < N * N; i++) {
		h_A[i] = 1;
		h_B[i] = 1;
	}

	float *d_A, *d_B, *d_M;
    e = cudaMalloc(&d_A, size);
    e = cudaMalloc(&d_B, size);
    e = cudaMalloc(&d_M, size);
	HANDLE_ERROR(e);

	e = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    e = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	HANDLE_ERROR(e);
		
	LARGE_INTEGER countPerSec, timeA, timeB;
	QueryPerformanceFrequency(&countPerSec);

	printf("\nMnozenie macierzy %dx%d przez cuBLAS:", N, N);

	cublasHandle_t handle;
	cublasCreate(&handle);

	float alpha = 1.0f;
	float beta = 1.0f;

	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_M, N);

    e = cudaMemcpy(h_M, d_M, size, cudaMemcpyDeviceToHost);
	HANDLE_ERROR(e);

	MatPrint(h_M, N); // Wypisanie

	cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_M);

	free(h_A);
	free(h_B);
	free(h_M);

	getchar();
	return 0;
}

