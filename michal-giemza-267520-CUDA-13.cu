// CUDA-01.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <windows.h>
#include <math.h>

#define INPUT_FILE "indeksyMedic.bin"
#define OUTPUT_FILE "wykresy.txt"
#define N  2048
#define T2 1024
#define T4 512
#define T8 256
#define PI 3.14159
#define B  10000

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
		printf(" %4.1f", V[i]);
	}
	putchar('\n');
}

double getTime(LARGE_INTEGER start, LARGE_INTEGER end, LARGE_INTEGER countPerSec) {
	return (double)(end.QuadPart - start.QuadPart) / countPerSec.QuadPart * 1000;
}

__global__ void Przeprobkowanie(float *I, float *Sin, float *O, int _t) {
	register int l, h, t = _t;
	register float lw, hw, inp;
	__shared__ float sin[N];
	
	for (int i = 0; i < N / t; i++) // i=0: 0-1023; i=1: 1024-2047;
		sin[threadIdx.x + i * t] = Sin[threadIdx.x + i * t];

	for (int i = 0; i < N / t; i++) {
		inp = I[threadIdx.x + i * t];

		l = (int)inp;
		h = ((float)l == inp ? l : l + 1);
		hw = inp - (float)l;
		lw = 1.0 - hw;

		l = (l >= N ? N-1 : l);
		h = (h >= N ? N-1 : h);
		l = (l < 0 ? 0 : l);
		h = (h < 0 ? 0 : h);

		O[threadIdx.x + i * t + blockIdx.x * N] = sin[l] * lw + sin[h] * hw;
	}
}

void InitSignalSin(float *s, float len) {
	float fb = 4 * (PI / 1800);
	for (int i = 0; i < len; i++)
	{
		s[i] = sin(i * fb) * 10;
	}
}

int _tmain(int argc, _TCHAR* argv[]) {
	size_t size = N * sizeof(float);
	cudaError_t e;
	FILE *iF, *oF;

	e = cudaSetDevice(0);
	HANDLE_ERROR(e);

	float *h_I, *h_S, *h_O;

	h_I = (float *)malloc(size);
	h_S = (float *)malloc(size);
	h_O = (float *)malloc(size * B);

	{
		iF = fopen(INPUT_FILE, "rb");
		if (!iF) {
			return 1;
		}
		fread(h_I, sizeof(float), 2048, iF);
		fclose(iF);
	}
	InitSignalSin(h_S, N);

	float *d_I, *d_S, *d_O;
    e = cudaMalloc(&d_I, size);
	HANDLE_ERROR(e);
    e = cudaMalloc(&d_S, size);
	HANDLE_ERROR(e);
    e = cudaMalloc(&d_O, size * B);
	HANDLE_ERROR(e);
	
	e = cudaMemcpy(d_I, h_I, size, cudaMemcpyHostToDevice);
	HANDLE_ERROR(e);
	e = cudaMemcpy(d_S, h_S, size, cudaMemcpyHostToDevice);
	HANDLE_ERROR(e);
	
	LARGE_INTEGER countPerSec, timeA, timeB;
	QueryPerformanceFrequency(&countPerSec);
	dim3 grid(B, 1, 1);

	{
		dim3 block(T2, 1, 1);
		QueryPerformanceCounter(&timeA);
		Przeprobkowanie<<<grid, block>>>(d_I, d_S, d_O, T2);
		cudaDeviceSynchronize();
		QueryPerformanceCounter(&timeB);
		printf("Czas dla %4d watkow: %f\n", T2, getTime(timeA, timeB, countPerSec));
		e = cudaGetLastError();
		HANDLE_ERROR(e);
	}

	{
		dim3 block(T4, 1, 1);
		QueryPerformanceCounter(&timeA);
		Przeprobkowanie<<<grid, block>>>(d_I, d_S, d_O, T4);
		cudaDeviceSynchronize();
		QueryPerformanceCounter(&timeB);
		printf("Czas dla %4d watkow: %f\n", T4, getTime(timeA, timeB, countPerSec));
		e = cudaGetLastError();
		HANDLE_ERROR(e);
	}
		
	{
		dim3 block(T8, 1, 1);
		QueryPerformanceCounter(&timeA);
		Przeprobkowanie<<<grid, block>>>(d_I, d_S, d_O, T8);
		cudaDeviceSynchronize();
		QueryPerformanceCounter(&timeB);
		printf("Czas dla %4d watkow: %f\n", T8, getTime(timeA, timeB, countPerSec));
		e = cudaGetLastError();
		HANDLE_ERROR(e);
	}

    e = cudaMemcpy(h_O, d_O, size * B, cudaMemcpyDeviceToHost);
	HANDLE_ERROR(e);
	
	{
		oF = fopen(OUTPUT_FILE, "wt");

		for (int i = 0; i < N; i++)
		{
			fprintf(oF, "%d\t%.4f\t%.4f\n",i , h_S[i % N], h_O[i]);
		}

		fclose(oF);
	}
	printf("\nWyniki zapisano do pliku %s.", OUTPUT_FILE);
	
	cudaFree(d_I);
    cudaFree(d_O);

	free(h_I);
	free(h_O);

	getchar();
	return 0;
}

