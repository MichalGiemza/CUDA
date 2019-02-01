// CUDA-01.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <windows.h>

#define N 2048
#define WS 5
#define MAX_T 1024

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
	
	for (int i = 0; i < n; i++) {
		if (i % n == 0)
			putchar('\n');
		printf(" %2.1f", V[i]);
	}
	putchar('\n');
}

double getTime(LARGE_INTEGER start, LARGE_INTEGER end, LARGE_INTEGER countPerSec) {
	return (double)(end.QuadPart - start.QuadPart) / countPerSec.QuadPart * 1000;
}

__global__ void FiltrMediana(float *A, float *M) {
	/*__shared__ float a[MAX_T];
	for (int i = WS / 2 + 1 + MAX_T * blockIdx.x; i < MAX_T * block; i++)
	{

	}*/
	
	//  Wartoœci skrajne
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		for (int i = 0; i < WS / 2; i++)
			M[i] = A[i];
	}
	if (threadIdx.x == 1 && blockIdx.x == 0) {
		for (int i = N - WS / 2; i < N; i++)
			M[i] = A[i];
	}
	//  Filtr mediana

	// wczytanie okna
	register float w[WS];
	for (int i = 0; i < WS; i++)
	{
		w[i] = A[i + threadIdx.x + blockIdx.x * MAX_T];
	}
	// sortowanie w oknie
	float tmp;
	for (int j = 0; j < WS; j++)
	{
		for (int i = 0; i < WS-1; i++)
		{
			if (w[i] > w[i+1])
			{
				tmp = w[i];
				w[i] = w[i+1];
				w[i+1] = tmp;
			}
		}
	}
	// wpisanie wyniku (mediany z okna)
	M[WS / 2 + threadIdx.x + blockIdx.x * MAX_T] = w[WS / 2];
}

int _tmain(int argc, _TCHAR* argv[]) {
	size_t size = N * sizeof(float);
	cudaError_t e;

	float *h_A, *h_M;

	h_A = (float *)malloc(size);
	h_M = (float *)malloc(size);

	for (int i = 0; i < N; i++) {
		h_A[i] = i % 9;
	}

	float *d_A, *d_M;
    e = cudaMalloc(&d_A, size);
    e = cudaMalloc(&d_M, size);
	HANDLE_ERROR(e);

	e = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	HANDLE_ERROR(e);
	
	printf("Sygnal wejsciowy:");
	dim3 block(MAX_T - WS / 2, 1, 1);
	dim3 grid(2, 1, 1);
	FiltrMediana<<<grid, block>>>(d_A, d_M);
		
    e = cudaMemcpy(h_M, d_M, size, cudaMemcpyDeviceToHost);
	HANDLE_ERROR(e);
	
	printf("Sygnal po filtracji medianowej:");
	MatPrint(h_M, N);

	cudaFree(d_A);
    cudaFree(d_M);

	free(h_A);
	free(h_M);

	getchar();
	return 0;
}

