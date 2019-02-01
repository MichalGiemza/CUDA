// CUDA-01.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <cuda_runtime.h>

int _tmain(int argc, _TCHAR* argv[])
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	int device;
	for (device = 0; device < deviceCount; ++device) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		
		printf("Nazwa: %s\nPamiec urzadzenia: %dGB\nPamiec dzielona: %dKB\nRejestry na blok: %dKB\nRozmiar osnowy: %d\nParametr zlozonosci obliczeniowej: major: %d, minor: %d\n\n",
			   deviceProp.name, deviceProp.totalGlobalMem/1024/1024, deviceProp.sharedMemPerBlock/1024, deviceProp.regsPerBlock/1024, deviceProp.warpSize, deviceProp.major, deviceProp.minor
			   );
	}
	getchar();
	return 0;
}

