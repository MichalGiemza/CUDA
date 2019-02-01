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

LARGE_INTEGER countPerSec, timeA, timeB;

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

// - - - - - - - - - -

__global__ void kernelTestRemapping2(float* spectra_dev, float* indeksy)
{
    __shared__ float spectra_shared[2048];
    register int index = blockIdx.x * blockDim.x + threadIdx.x; //liniowy numer w¹tku
    register int indx4 = index << 1;

    register int ind_m_x4 = threadIdx.x << 1;

    spectra_shared[ind_m_x4] = spectra_dev[indx4];
    spectra_shared[ind_m_x4 + 1] = spectra_dev[indx4 + 1];


    register float in1 = indeksy[ind_m_x4];
    register float in2 = indeksy[ind_m_x4 + 1];

    register float temp1 = in1 - floorf(in1);
    register float temp2 = in2 - floorf(in2);

    __syncthreads();

    spectra_dev[indx4] = (1 - temp1)*spectra_shared[(int)floorf(in1)] + temp1*spectra_shared[(int)ceilf(in1)];
    spectra_dev[indx4 + 1] = (1 - temp2)*spectra_shared[(int)floorf(in2)] + temp2*spectra_shared[(int)ceilf(in2)];
}

void TestRemapping2(float *dane_dev, float *indeksy, int allLines)
{
    // execute the kernel
    int nThreads = 1024;                            //liczba w¹tków w bloku
    int totalThreads = allLines*nThreads;
    int nBlocks = allLines;

	QueryPerformanceCounter(&timeA);
    kernelTestRemapping2 << < nBlocks, nThreads >> >(dane_dev, indeksy);

    cudaDeviceSynchronize();
	QueryPerformanceCounter(&timeB);
}

__global__ void kernelTestRemapping4(float* spectra_dev, float* indeksy)
{
    __shared__ float spectra_shared[2048];
    register int index = blockIdx.x * blockDim.x + threadIdx.x; //liniowy numer w¹tku
    register int indx4 = index << 2;

    register int ind_m_x4 = threadIdx.x << 2; //modulo 512, indeks_m tymczasowo pod nazw¹ "ind_m_x4"

    spectra_shared[ind_m_x4] = spectra_dev[indx4];
    spectra_shared[ind_m_x4 + 1] = spectra_dev[indx4 + 1];
    spectra_shared[ind_m_x4 + 2] = spectra_dev[indx4 + 2];
    spectra_shared[ind_m_x4 + 3] = spectra_dev[indx4 + 3];


    register float in1 = indeksy[ind_m_x4];
    register float in2 = indeksy[ind_m_x4 + 1];
    register float in3 = indeksy[ind_m_x4 + 2];
    register float in4 = indeksy[ind_m_x4 + 3];

    register float temp1 = in1 - floorf(in1);
    register float temp2 = in2 - floorf(in2);
    register float temp3 = in3 - floorf(in3);
    register float temp4 = in4 - floorf(in4);

    __syncthreads();

    spectra_dev[indx4] = (1 - temp1)*spectra_shared[(int)floorf(in1)] + temp1*spectra_shared[(int)ceilf(in1)];
    spectra_dev[indx4 + 1] = (1 - temp2)*spectra_shared[(int)floorf(in2)] + temp2*spectra_shared[(int)ceilf(in2)];
    spectra_dev[indx4 + 2] = (1 - temp3)*spectra_shared[(int)floorf(in3)] + temp3*spectra_shared[(int)ceilf(in3)];
    spectra_dev[indx4 + 3] = (1 - temp4)*spectra_shared[(int)floorf(in4)] + temp4*spectra_shared[(int)ceilf(in4)];

}

void TestRemapping4(float *dane_dev, float *indeksy, int allLines)
{
    // execute the kernel
    int nThreads = 512;                            //liczba w¹tków w bloku
    int totalThreads = allLines*nThreads;
    int nBlocks = allLines;
	
	QueryPerformanceCounter(&timeA);
    kernelTestRemapping4 << < nBlocks, nThreads >> >(dane_dev, indeksy);

    cudaDeviceSynchronize();
	QueryPerformanceCounter(&timeB);
}

__global__ void kernelTestRemapping8(float* spectra_dev, float* indeksy)
{
    __shared__ float spectra_shared[2048];
    register int index = blockIdx.x * blockDim.x + threadIdx.x; //liniowy numer w¹tku
    register int indx4 = index << 3;

    register int ind_m_x4 = threadIdx.x << 3; //modulo 512, indeks_m tymczasowo pod nazw¹ "ind_m_x4"

    spectra_shared[ind_m_x4] = spectra_dev[indx4];
    spectra_shared[ind_m_x4 + 1] = spectra_dev[indx4 + 1];
    spectra_shared[ind_m_x4 + 2] = spectra_dev[indx4 + 2];
    spectra_shared[ind_m_x4 + 3] = spectra_dev[indx4 + 3];
    spectra_shared[ind_m_x4 + 4] = spectra_dev[indx4 + 4];
    spectra_shared[ind_m_x4 + 5] = spectra_dev[indx4 + 5];
    spectra_shared[ind_m_x4 + 6] = spectra_dev[indx4 + 6];
    spectra_shared[ind_m_x4 + 7] = spectra_dev[indx4 + 7];


    register float in1 = indeksy[ind_m_x4];
    register float in2 = indeksy[ind_m_x4 + 1];
    register float in3 = indeksy[ind_m_x4 + 2];
    register float in4 = indeksy[ind_m_x4 + 3];
    register float in5 = indeksy[ind_m_x4 + 4];
    register float in6 = indeksy[ind_m_x4 + 5];
    register float in7 = indeksy[ind_m_x4 + 6];
    register float in8 = indeksy[ind_m_x4 + 7];

    register float temp1 = in1 - floorf(in1);
    register float temp2 = in2 - floorf(in2);
    register float temp3 = in3 - floorf(in3);
    register float temp4 = in4 - floorf(in4);
    register float temp5 = in4 - floorf(in5);
    register float temp6 = in4 - floorf(in6);
    register float temp7 = in4 - floorf(in7);
    register float temp8 = in4 - floorf(in8);

    __syncthreads();

    spectra_dev[indx4] = (1 - temp1)*spectra_shared[(int)floorf(in1)] + temp1*spectra_shared[(int)ceilf(in1)];
    spectra_dev[indx4 + 1] = (1 - temp2)*spectra_shared[(int)floorf(in2)] + temp2*spectra_shared[(int)ceilf(in2)];
    spectra_dev[indx4 + 2] = (1 - temp3)*spectra_shared[(int)floorf(in3)] + temp3*spectra_shared[(int)ceilf(in3)];
    spectra_dev[indx4 + 3] = (1 - temp4)*spectra_shared[(int)floorf(in4)] + temp4*spectra_shared[(int)ceilf(in4)];
    spectra_dev[indx4 + 4] = (1 - temp5)*spectra_shared[(int)floorf(in5)] + temp5*spectra_shared[(int)ceilf(in5)];
    spectra_dev[indx4 + 5] = (1 - temp6)*spectra_shared[(int)floorf(in6)] + temp6*spectra_shared[(int)ceilf(in6)];
    spectra_dev[indx4 + 6] = (1 - temp7)*spectra_shared[(int)floorf(in7)] + temp7*spectra_shared[(int)ceilf(in7)];
    spectra_dev[indx4 + 7] = (1 - temp8)*spectra_shared[(int)floorf(in8)] + temp8*spectra_shared[(int)ceilf(in8)];

}

void TestRemapping8(float *dane_dev, float *indeksy, int allLines)
{
    // execute the kernel
    int nThreads = 256;                            //liczba w¹tków w bloku
    int totalThreads = allLines*nThreads;
    int nBlocks = allLines;
	
	QueryPerformanceCounter(&timeA);
    kernelTestRemapping8 << < nBlocks, nThreads >> >(dane_dev, indeksy);

    cudaDeviceSynchronize();
	QueryPerformanceCounter(&timeB);
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

	TestRemapping2(d_S, d_I, B);
	printf("Czas dla TestRemapping2: %f\n", getTime(timeA, timeB, countPerSec));

	e = cudaMemcpy(d_S, h_S, size, cudaMemcpyHostToDevice);
	HANDLE_ERROR(e);
		
	TestRemapping4(d_S, d_I, B);
	printf("Czas dla TestRemapping4: %f\n", getTime(timeA, timeB, countPerSec));

	e = cudaMemcpy(d_S, h_S, size, cudaMemcpyHostToDevice);
	HANDLE_ERROR(e);
	
	TestRemapping8(d_S, d_I, B);
	printf("Czas dla TestRemapping8: %f\n", getTime(timeA, timeB, countPerSec));


    e = cudaMemcpy(h_O, d_O, size * B, cudaMemcpyDeviceToHost);
	HANDLE_ERROR(e);
	
	//{
	//	oF = fopen(OUTPUT_FILE, "wt");
	//
	//	for (int i = 0; i < N; i++)
	//	{
	//		fprintf(oF, "%d\t%.4f\t%.4f\n",i , h_S[i % N], h_O[i]);
	//	}
	//
	//	fclose(oF);
	//}
	//printf("\nWyniki zapisano do pliku %s.", OUTPUT_FILE);
	
	cudaFree(d_I);
    cudaFree(d_O);

	free(h_I);
	free(h_O);

	getchar();
	return 0;
}

