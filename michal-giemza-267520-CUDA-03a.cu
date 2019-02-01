// CUDA-01.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <windows.h>

#define N 1024

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

static void HandleError(cudaError_t err, const char *file, int line) {
	//cudaError_t cudastatus = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		getchar();
		exit(EXIT_FAILURE);
	}
}

__global__ void VecScalarProduct_0(float *A, float *B, float *SP) {
    //int x = blockIdx.x * blockDim.x + threadIdx.x;
    //int y = blockIdx.y * blockDim.y + threadIdx.y;
	register int i = threadIdx.x;

	float mult[N]; //?
	mult[i] = A[i] * B[i];

	__syncthreads();
	
	if (i == 0) {
		*SP = 0;
		for (int j = 0; j < N; j++)
			*SP += mult[j];
	}
}

__global__ void VecScalarProduct_SyncSum(float *A, float *B, float *SP) {
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

__global__ void VecScalarProduct2(float *A, float *B, float *SP) {
	register int i = threadIdx.x;

	__shared__ float mult[N];
	mult[i] = A[i] * B[i];

	__syncthreads();
	
	for (int k = N / 2; k >= 1; k /= 2)
	{   // iteracje
		__syncthreads();
		if (i < k)
		{   // po tablicy
			mult[i] += mult[i + k];
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

double getTime(LARGE_INTEGER start, LARGE_INTEGER end, LARGE_INTEGER countPerSec) {
	return (double)(end.QuadPart - start.QuadPart) / countPerSec.QuadPart * 1000;
}

int _tmain(int argc, _TCHAR* argv[]) {
    size_t size = N * sizeof(float);
	cudaError_t e;

	float *h_A, *h_B, h_SP;
	h_A = (float *)malloc(size);
	h_B = (float *)malloc(size);
	for (int i = 0; i < N; i++) {
		h_A[i] = 1;
		h_B[i] = 1;
	}

	float *d_A, *d_B, *d_SP;
    e = cudaMalloc(&d_A, size);
    e = cudaMalloc(&d_B, size);
    e = cudaMalloc(&d_SP, sizeof(float));
	HANDLE_ERROR(e);

	e = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    e = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	HANDLE_ERROR(e);
	
	dim3 block(N, 1, 1);
	dim3 grid(1, 1, 1);
	
	//LARGE_INTEGER countPerSec, timeA, timeB, timeC, timeD, timeE, timeF;
	//QueryPerformanceFrequency(&countPerSec);

	//QueryPerformanceCounter(&timeA);
    //VecScalarProduct_0<<<grid, block>>>(d_A, d_B, d_SP);
	//cudaDeviceSynchronize();

	//QueryPerformanceCounter(&timeB);
    //VecScalarProduct_SyncSumNoReg<<<grid, block>>>(d_A, d_B, d_SP);
	//cudaDeviceSynchronize();

	//QueryPerformanceCounter(&timeC);
    //VecScalarProduct_SyncSum<<<grid, block>>>(d_A, d_B, d_SP);
	//cudaDeviceSynchronize();

	//QueryPerformanceCounter(&timeD);
    //VecScalarProduct<<<grid, block>>>(d_A, d_B, d_SP);
	//cudaDeviceSynchronize();

	//QueryPerformanceCounter(&timeE);
    VecScalarProduct2<<<grid, block>>>(d_A, d_B, d_SP);
	//cudaDeviceSynchronize();

	//QueryPerformanceCounter(&timeF);


	//printf("%fms - Czas dla wer. 0\n", getTime(timeA, timeB, countPerSec));
	//printf("%fms - Czas dla wer. bez rejestru\n", getTime(timeB, timeC, countPerSec));
	//printf("%fms - Czas dla wer. z suma synchr.\n", getTime(timeC, timeD, countPerSec));
	//printf("%fms - Czas dla wer. rownoleglej\n", getTime(timeD, timeE, countPerSec));
	//printf("%fms - Czas dla wer. zsunietej\n", getTime(timeE, timeF, countPerSec));

    e = cudaMemcpy(&h_SP, d_SP, sizeof(float), cudaMemcpyDeviceToHost);
	HANDLE_ERROR(e);

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

