// CUDA-01.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <windows.h>
#include <math.h>
#include <cufft.h>

#define SIZE_ 1024 * 1024
#define N 75

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

static void HandleError(cudaError_t err, const char *file, int line) {
	//cudaError_t cudastatus = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		getchar();
		exit(EXIT_FAILURE);
	}
}

__global__ void kernel(float *a, float *b, float *c) {
	c[threadIdx.x] =  a[threadIdx.x] + b[threadIdx.x];
	c[threadIdx.x] =  a[threadIdx.x] + b[threadIdx.x];
	c[threadIdx.x] =  a[threadIdx.x] + b[threadIdx.x];
	c[threadIdx.x] =  a[threadIdx.x] + b[threadIdx.x];
	c[threadIdx.x] =  a[threadIdx.x] + b[threadIdx.x];
	c[threadIdx.x] =  a[threadIdx.x] + b[threadIdx.x];
	c[threadIdx.x] =  a[threadIdx.x] + b[threadIdx.x];
	c[threadIdx.x] =  a[threadIdx.x] + b[threadIdx.x];
	c[threadIdx.x] =  a[threadIdx.x] + b[threadIdx.x];
	c[threadIdx.x] =  a[threadIdx.x] + b[threadIdx.x];
	c[threadIdx.x] =  a[threadIdx.x] + b[threadIdx.x];
	c[threadIdx.x] =  a[threadIdx.x] + b[threadIdx.x];
}

int _tmain(int argc, _TCHAR* argv[]) {
	size_t size = SIZE_ * sizeof(float);
	cudaError_t e;

	e = cudaSetDevice(0);
	HANDLE_ERROR(e);

	float *h, *hb, *hc;
	
	float *d, *db, *dc;
	
    e = cudaMalloc(&d, size * N);
	HANDLE_ERROR(e);
    e = cudaMalloc(&db, size * N);
	HANDLE_ERROR(e);
    e = cudaMalloc(&dc, size * N);
	HANDLE_ERROR(e);

	//{
	//	h = (float *)malloc(size);
	//
	//	cudaEvent_t start, stop;
	//
	//	cudaEventCreate(&start);
	//	cudaEventCreate(&stop);
	//
	//	cudaEventRecord(start, 0);
	//	e = cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);
	//	cudaEventRecord(stop, 0);
	//	cudaEventSynchronize(stop);
	//	HANDLE_ERROR(e);
	//
	//	float time;
	//	cudaEventElapsedTime(&time, start, stop);
	//	printf("Czas dzialania ze str : %f\n", time);
	//
	//	free(h);
	//}

	{
		cudaMallocHost(&h, size * N);
		cudaMallocHost(&hb, size * N);
		cudaMallocHost(&hc, size * N);
		for (int i = 0; i < SIZE_ * N; i++)
			h[i] = 2;
		for (int i = 0; i < SIZE_ * N; i++)
			hb[i] = 3;

		cudaEvent_t start, stop;
		cudaStream_t stream1, stream2;
	
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaStreamCreate(&stream1);
		cudaStreamCreate(&stream2);
		
		cudaEventRecord(start, 0);
		for (int i = 0; i < SIZE_ * N; i += 2 * SIZE_)
		{
			e = cudaMemcpyAsync(d, h + i, size, cudaMemcpyHostToDevice, stream1);
			e = cudaMemcpyAsync(d, h + i + SIZE_, size, cudaMemcpyHostToDevice, stream2);
			
			e = cudaMemcpyAsync(d, hb + i, size, cudaMemcpyHostToDevice, stream1);
			e = cudaMemcpyAsync(d, hb + i+ SIZE_, size, cudaMemcpyHostToDevice, stream2);

			kernel<<< SIZE_ / 512, 512, 0, stream1 >>>(h, hb, hc);
			kernel<<< SIZE_ / 512, 512, 0, stream2 >>>(h, hb, hc);

			e = cudaMemcpyAsync(h + i, d, size, cudaMemcpyDeviceToHost, stream1);
			e = cudaMemcpyAsync(h + i + SIZE_, d, size, cudaMemcpyDeviceToHost, stream2);
		}
		cudaStreamSynchronize(stream1);
		cudaStreamSynchronize(stream2);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		HANDLE_ERROR(e);

		float time;
		cudaEventElapsedTime(&time, start, stop);
		printf("Czas dzialania bez str: %f\n", time);

		cudaFreeHost(h);
		cudaStreamDestroy(stream1);
		cudaStreamDestroy(stream2);
	}

	cudaFree(d);

	printf("Nacisnij Enter, aby zakonczyc.");
	getchar();
	return 0;
}

