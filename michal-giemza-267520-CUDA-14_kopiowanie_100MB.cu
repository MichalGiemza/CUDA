// CUDA-01.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <windows.h>
#include <math.h>
#include <cufft.h>

#define INPUT_FILE "indeksyMedic.bin"
#define OUTPUT_FILE "wykresy_Z14.txt"
#define X 1024
#define Y 1024
#define N 100

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

static void HandleError(cudaError_t err, const char *file, int line) {
	//cudaError_t cudastatus = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		getchar();
		exit(EXIT_FAILURE);
	}
}

int _tmain(int argc, _TCHAR* argv[]) {
	size_t size = X * Y * sizeof(float);
	cudaError_t e;

	e = cudaSetDevice(0);
	HANDLE_ERROR(e);

	float *h;
	
	float *d;
    e = cudaMalloc(&d, size);
	HANDLE_ERROR(e);

	{
		h = (float *)malloc(size);

		cudaEvent_t start, stop;
	
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	
		cudaEventRecord(start, 0);
		e = cudaMemcpy(d, h, size, cudaMemcpyHostToDevice);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		HANDLE_ERROR(e);

		float time;
		cudaEventElapsedTime(&time, start, stop);
		printf("Czas dzialania ze str : %f\n", time);

		free(h);
	}

	{
		cudaMallocHost(&h, size);

		cudaEvent_t start, stop;
		cudaStream_t stream;
	
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaStreamCreate(&stream);
	
		cudaEventRecord(start, 0);
		cudaMemcpyAsync(d, h, size, cudaMemcpyHostToDevice, stream);
		cudaStreamSynchronize(stream);
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		HANDLE_ERROR(e);

		float time;
		cudaEventElapsedTime(&time, start, stop);
		printf("Czas dzialania bez str: %f\n", time);

		cudaFreeHost(h);
		cudaStreamDestroy(stream);
	}

	cudaFree(d);

	printf("Nacisnij Enter, aby zakonczyc.");
	getchar();
	return 0;
}

