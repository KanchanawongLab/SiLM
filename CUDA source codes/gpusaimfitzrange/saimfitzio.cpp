//main functions 
#include <windows.h>
#pragma comment(lib, "kernel32.lib")

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <tchar.h>
#include <io.h>
#include "device_launch_parameters.h"

void cudasafe( cudaError_t err, char* str, int lineNumber,FILE *fp1);
void cudaavailable(int silent,FILE *fp1) ;

FILE *fp;



#define Nfitsmax 10e6
#define PI 3.141582f
#define BSZ 128
extern void kernel_saimfitz_wrapper(dim3 dimGrid, dim3 dimBlock,float *data, const float *absrTE, const float *anglerTE, const float *zfact,float *fit_result, float *fiterror,
	const int iterations, int sz, int fitmode, const int *zrange);


#ifndef max
//! not defined in the C standard used by visual studio
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
//! not defined in the C standard used by visual studio
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif


#define DLL_EXPORT extern "C" __declspec(dllexport)

DLL_EXPORT void addone(unsigned int *a, int b)
{
	  for (int i=0;i<b;i++)
	  {
		  a[i]++;
	  }

	  //FILE *fp;
	  fp=fopen("log.txt","a");
	  
	  fprintf(fp,"addone\n");
	 
	  fclose(fp);
}

// input
//data:  pointer of the dataset to be fitted
//absrTE, anglerTE, zfact: constant parameters in scanning angle interference illumination
//Nfitraw: number of data points to be fitted
//sz: number of illumination angles in scanning angle interference experiments
//iterations: number of interations of the optimization algorithm
//zrange: pre-defined range of axial positions to be fitted

//output
//fit_result: output fitresults 
//fit_error: fiterror and crlb
//fitmode: fitmode used, now it only use one mode

DLL_EXPORT void pysaimfitz(float *data, float *absrTE, float *anglerTE, float *zfact, int Nfitraw, int sz, int iterations,float *fit_result,float *fit_error, int fitmode, int *zrange){
    int blockx=0;
	int threadx=0;
	
	// debug codes checking whether data are passed to this code correctly
	const int NV=2;
	fp=fopen("log.txt","w");
	fprintf(fp,"sz%d", sz);
	fprintf(fp,"nfitraw%d", Nfitraw);
	fprintf(fp,"iterations%d\n",iterations);
	fprintf(fp,"fitmode%d\n",fitmode);


	float *anglerTE1=0;
	int i;
	for (i=0;i<sz;i++)
	{
		
		fprintf(fp,"%5f\t",absrTE[i]);
		fprintf(fp,"%5f\t",anglerTE[i]);
		fprintf(fp,"%5f\t",data[i]);
		fprintf(fp,"%5f\n",data[sz+i]);
	}


	// define the gpu parameters
	int silent=0;
	
	//check the GPUs
	int deviceCount=0;
	int driverVersion=0;
	int runtimeVersion=0;
	cudaDeviceProp deviceProp;
	cudaavailable(silent,fp);
	cudasafe(cudaGetDeviceProperties(&deviceProp,0),"Could not get properties for device 0.",__LINE__,fp);
	
	 
	int BlockSize=BSZ;

	printf("%d\n",deviceProp.totalGlobalMem/(1024*1024));

	const size_t availableMemory = deviceProp.totalGlobalMem/(1024*1024);
	size_t requiredMemory = 3*Nfitraw;

	if (requiredMemory > 0.95*deviceProp.totalGlobalMem)
		printf("GPUfitZ:NotEnoughMemory Trying to allocation %dMB. GPU only has %dMB. Please break your fitting into multiple smaller runs.\n",
		requiredMemory/(1024*1024),availableMemory);

	

	float *d_data, *d_absrTE,*d_anglerTE,*d_zfact;
	int *d_zrange;

	cudaDeviceReset();
	cudaSetDevice(0);


	const int sz1=sz;
	const int Nfitraw1=Nfitraw;
	const int iterations1=iterations;
	const int fitmode1=fitmode;
	const int NCRLB=3;   

	
	// passing the data to the gpu 
	cudasafe(cudaMalloc((void**)&sz,sizeof(int)),"testcupdamallc",__LINE__,fp);

	cudasafe(cudaMalloc((void**)&d_anglerTE,sz1*sizeof(float)),"failed cudamalloc on d_anglerTE",__LINE__,fp);
	cudasafe(cudaMemset(d_anglerTE, 0, sz1*sizeof(float)),"Failed cudaMemset on d_anglerTE.",__LINE__,fp);
	cudasafe(cudaMemcpy(d_anglerTE,anglerTE,sz1*sizeof(float),cudaMemcpyHostToDevice),"failed cudamemcpy on d_anglerTE",__LINE__,fp);
	
	cudasafe(cudaMalloc((void**)&d_data, sz1*Nfitraw1*sizeof(float)),"Failed cudaMalloc on d_data.",__LINE__,fp);
	cudasafe(cudaMemset(d_data, 0, sz1*Nfitraw1*sizeof(float)),"Failed cudaMemset on d_data.",__LINE__,fp);
	cudasafe(cudaMemcpy(d_data,data,sz1*Nfitraw1*sizeof(float),cudaMemcpyHostToDevice),"failed cudamemcpy on d_data",__LINE__,fp);
	
	cudasafe(cudaMalloc((void**)&d_absrTE,sz1*sizeof(float)),"failed cudamalloc on d_absrTE",__LINE__,fp);
	cudasafe(cudaMemset(d_absrTE, 0, sz1*sizeof(float)),"Failed cudaMemset on d_absrTE.",__LINE__,fp);
	cudasafe(cudaMemcpy(d_absrTE,absrTE,sz1*sizeof(float),cudaMemcpyHostToDevice),"failed cudamemcpy on d_absrTE",__LINE__,fp);
	
    
	cudasafe(cudaMalloc((void**)&d_zfact,sz1*sizeof(float)),"failed cudamalloc on d_zfact",__LINE__,fp);
	cudasafe(cudaMemset(d_zfact, 0, sz1*sizeof(float)),"Failed cudaMemset on d_zfact.",__LINE__,fp);
	cudasafe(cudaMemcpy(d_zfact,zfact,sz1*sizeof(float),cudaMemcpyHostToDevice),"failed cudamemcpy on d_zfact",__LINE__,fp);

	cudasafe(cudaMalloc((void**)&d_zrange, 2 * sizeof(int)), "failed cudamalloc on d_zrange", __LINE__, fp);
	cudasafe(cudaMemset(d_zrange, 0, 2 * sizeof(int)), "Failed cudaMemset on d_zrange", __LINE__, fp);
	cudasafe(cudaMemcpy(d_zrange, zrange, 2 * sizeof(int), cudaMemcpyHostToDevice), "failed cudamemcpy on d_zrange", __LINE__, fp);




	// define the return result 
	float *d_fit_result;
	float *d_fit_error;
	cudasafe(cudaMalloc((void**)&d_fit_result, Nfitraw1*NV*sizeof(float)),"Failed cudaMalloc on d_fitresults",__LINE__,fp);
	cudasafe(cudaMemset(d_fit_result, 0, Nfitraw1*NV*sizeof(float)),"Failed cudaMemset on d_fitresults.",__LINE__,fp);

	cudasafe(cudaMalloc((void**)&d_fit_error, NCRLB*Nfitraw1*sizeof(float)),"Failed cudaMalloc on d_fiterror",__LINE__,fp);
	cudasafe(cudaMemset(d_fit_error, 0, NCRLB*Nfitraw1*sizeof(float)),"Failed cudaMemset on d_fiterror.",__LINE__,fp);

	
	//call the gpu code
	blockx = (int) ceil( (float)Nfitraw1/(float)BlockSize);
	threadx= BlockSize;  //blocksize=64,128 or 256. suggest 128

	

	
	
	printf("%d",blockx);
	printf("%d",threadx);

	
	// calling the gpu code 
	dim3 dimBlock(threadx);
	dim3 dimGrid(blockx);
	kernel_saimfitz_wrapper(dimGrid, dimBlock,d_data,d_absrTE,d_anglerTE,d_zfact,d_fit_result,d_fit_error,iterations1,sz1,fitmode1,d_zrange);


	// return the gpu data to cpu
	cudasafe(cudaMemcpy(fit_error, d_fit_error, NCRLB*Nfitraw1*sizeof(float), cudaMemcpyDeviceToHost),
			"cudaMemcpy failed for copyfitresults.",__LINE__,fp);
	cudasafe(cudaMemcpy(fit_result, d_fit_result, Nfitraw1*NV*sizeof(float), cudaMemcpyDeviceToHost),
			"cudaMemcpy failed for fiterror.",__LINE__,fp);  
	
	cudasafe(cudaFree(d_data),"cudaFree failed on d_data.",__LINE__,fp);
	cudasafe(cudaFree(d_absrTE),"cudaFree failed on d_absrTE.",__LINE__,fp);
	cudasafe(cudaFree(d_anglerTE),"cudaFree failed on d_anglerTE.",__LINE__,fp);
	cudasafe(cudaFree(d_zfact),"cudaFree failed on d_zfact.",__LINE__,fp);
	cudasafe(cudaFree(d_fit_result),"cudaFree failed on fit_result.",__LINE__,fp);
	cudasafe(cudaFree(d_fit_error),"cudaFree failed on fit_error",__LINE__,fp); 
	cudasafe(cudaFree(d_zrange), "cudaFree failed on zrange", __LINE__, fp);

	fclose(fp);
	//exit(1);

	return;

	}

void cudasafe( cudaError_t err, char* str, int lineNumber,FILE *fp1)
{
	
	
	if (err != cudaSuccess)
	{
		//reset all cuda devices
		int deviceCount = 0;
		int ii = 0;
		cudasafe(cudaGetDeviceCount(&deviceCount),"cudaGetDeviceCount",__LINE__ ,fp); //query number of GPUs
		for (ii = 0; ii< deviceCount;ii++) {
			cudaSetDevice(ii);
			cudaDeviceReset();
		}
		fprintf(fp1,"GPUfitZ:cudaFail %s failed \n",str);
		exit(1); // might not stop matlab
	}

	if(err==cudaSuccess)
	{
		fprintf(fp1,"cudasuccessed%s\n",str);
		//exit(1);
	}
	
}

void cudaavailable(int silent,FILE *fp1) {
	int driverVersion=0, runtimeVersion=0, deviceCount=0;
	if (cudaSuccess == cudaDriverGetVersion(&driverVersion)) {
		if  (silent==0)
			fprintf(fp1,"CUDA driver version: %d\n", driverVersion);

	} else { 
		fprintf(fp1,"GPUfitZ:nodriver","Could not query CUDA driver version\n");
	}
	if (cudaSuccess == cudaRuntimeGetVersion(&runtimeVersion)) {
		if  (silent==0)
			fprintf(fp1,"CUDA driver version: %d\n", runtimeVersion);

	} else { 
		fprintf(fp1,"GPUfitZ:noruntime","Could not query CUDA runtime version\n");
	}
	if (cudaSuccess == cudaGetDeviceCount(&deviceCount)) {
		if  (silent==0)
			fprintf(fp1,"CUDA devices detected: %d\n", deviceCount);

	} else {
		fprintf(fp1,"GPUfitZ:nodevices","Could not query CUDA device count\n", runtimeVersion);
	}
	if (deviceCount < 1) {
		fprintf(fp1,"GPUfitZ:NoDevice","No CUDA capable devices were detected");
	}
}
