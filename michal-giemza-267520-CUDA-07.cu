// CUDA-01.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <windows.h>

#define N 64

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

static void HandleError(cudaError_t err, const char *file, int line) {
	//cudaError_t cudastatus = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		getchar();
		exit(EXIT_FAILURE);
	}
}

static void MultMatCPU(float *A, float *B, float *M) {
	float sum, a, b;

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			sum = 0;
			for (int k = 0; k < N; k++) {
				a = A[i * N + k];
				b = B[k * N + j];
				sum += a * b;
			}
			M[i * N + j] = sum;
		}
	}
}

__global__ void MultMat(float *A, float *B, float *M) {
	__shared__ float Nds[32][32], Mds[32][32];//, Ods[32][32];
	register float sum = 0;

	//Ods[threadIdx.x][threadIdx.y] = 0;
	for (int k = 0; k < N / 32; k += 1) {
		__syncthreads();

		Mds[threadIdx.x][threadIdx.y] = A[(blockIdx.x + k) * 32 + threadIdx.x];
		Nds[threadIdx.x][threadIdx.y] = B[(blockIdx.y + k) * 32 + threadIdx.y];

		for (int p = 0; p < 32; p++) {
			//Ods[threadIdx.x][threadIdx.y] += Mds[p][threadIdx.y] * Nds[threadIdx.x][p];
			sum += Mds[p][threadIdx.y] * Nds[threadIdx.x][p];
		}
	}
	__syncthreads();
	M[(blockIdx.y * 32 + threadIdx.y) * N + blockIdx.x * 32 + threadIdx.x] = sum;
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
		
	printf("\nMnozenie macierzy %dx%d przez CUDA (Rozwiazanie z kafelkami):", N, N);

	dim3 block(32, 32, 1);
	dim3 grid(N / 32, N / 32, 1);
	
    MultMat<<<grid, block>>>(d_A, d_B, d_M);
	cudaDeviceSynchronize();

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

