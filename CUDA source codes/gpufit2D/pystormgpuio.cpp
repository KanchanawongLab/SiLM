//main function for 2D Gaussian fitting 
// The code is modified from Yiming Li's code 
//Li, Yiming, et al. "Real-time 3D single-molecule localization using experimental point spread functions." Nature methods 15.5 (2018) : 367 - 369.
//and Keith Lidke's code 
//C. Simth, N. Joseph, B. Rieger & K. Lidke. "Fast, single-molecule localization that achieves theoretically minimum uncertainty." Nat. Methods, 7, 373, 2010
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

extern void kernel_gpustorm_fit_wrapper(dim3 dimGrid, dim3 dimBlock, float *d_data, float *fitresult,float *fiterror ,const int iterations, const int sz, int fitmode, const float psfSigma);



#ifndef max
//! not defined in the C standard used by visual studio
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
//! not defined in the C standard used by visual studio
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif


#define DLL_EXPORT extern "C" __declspec(dllexport)

//test function addone which is used while debuging
DLL_EXPORT void addone(unsigned int *a, int b)
{
	  for (int i=0;i<b;i++)
	  {
		  a[i]++;
	  }

	  //FILE *fp;
	  fp=fopen("log.txt","a");
	  
	  fprintf(fp,"addone\n");
	  //fprintf(fp,"")
	  //fputs("52525",fp);
	  fclose(fp);
}





//input
// data: dataset to fit
// Nfitraw: total numbers of dataset
//sz: size of the dataset, sz*sz
//interations:
//PSFSigma: initial value of psfsigma
//fitmode: switch between different modes of sigma
//1: fit with least squre error and two sigma are output
//2: fit with maximum likelihood error and two sigma are output
//3: fit with least squre error and one sigma are output
//4: fit with maximum likelihood error and one sigma are output

//output:
//fitresults: fitresults with  dimension of NV*Nfitraw
//fiterror: crlb for each fits


DLL_EXPORT void pystorm(float *data, int Nfitraw, int sz, int iterations, float PSFSigma, float *fit_result,float *fit_error, int fitmode){
    int blockx=0;
	int threadx=0;
	int NV1;
	int NCRLB1;

	
	// sx,sy,sz, data, fittype, psfsigma, iterations,

	if (fitmode < 3) {
		NV1 = 6; NCRLB1 = 7;
	}
	else if (fitmode > 2) {
		NV1 = 5; NCRLB1 = 6;
	}
	
	fp=fopen("log.txt","w");
 

	fprintf(fp,"sz%d", sz);
	fprintf(fp,"nfitraw%d", Nfitraw);
	fprintf(fp,"iterations%d\n",iterations);
	fprintf(fp,"fitmode%d\n",fitmode);
	fprintf(fp,"psfsigma%5f\n",PSFSigma);

	//debug codes testing whether the dataset are passed to the c code
	int i;
	for (i=0;i<sz*sz;i++)
	{
		fprintf(fp,"%5f\t",data[i]);
		fprintf(fp,"%5f\n",data[sz*sz+i]);
	}



	// define the gpu parameters, and test calling the gpu
	int silent=0;
	
	int deviceCount=0;
	int driverVersion=0;
	int runtimeVersion=0;
	cudaDeviceProp deviceProp;
	cudaavailable(silent,fp);
	cudasafe(cudaGetDeviceProperties(&deviceProp,0),"Could not get properties for device 0.",__LINE__,fp);
	
	 
	int BlockSize=BSZ;
	printf("%d\n",deviceProp.totalGlobalMem/(1024*1024));

	const size_t availableMemory = deviceProp.totalGlobalMem/(1024*1024);
	size_t requiredMemory = 30*Nfitraw;

	if (requiredMemory > 0.95*deviceProp.totalGlobalMem)
		printf("GPU2dFit:NotEnoughMemory Trying to allocation %dMB. GPU only has %dMB. Please break your fitting into multiple smaller runs.\n",
		requiredMemory/(1024*1024),availableMemory);

	

	float *d_data;

	cudaDeviceReset();
	cudaSetDevice(0);


	const int sz1=sz;
	const int Nfitraw1=Nfitraw;
	const int iterations1=iterations;
	const int fitmode1=fitmode;
	const float psfSigma1=PSFSigma;
	const int NCRLB=NCRLB1;

	
	//passing the dataset to gpu
	cudasafe(cudaMalloc((void**)&sz,sizeof(int)),"testcupdamallc",__LINE__,fp);
	cudasafe(cudaMalloc((void**)&d_data, sz1*sz1*Nfitraw1*sizeof(float)),"Failed cudaMalloc on d_data.",__LINE__,fp);
	cudasafe(cudaMemset(d_data, 0, sz1*sz1*Nfitraw1*sizeof(float)),"Failed cudaMemset on d_data.",__LINE__,fp);
	cudasafe(cudaMemcpy(d_data,data,sz1*sz1*Nfitraw1*sizeof(float),cudaMemcpyHostToDevice),"failed cudamemcpy on d_data",__LINE__,fp);
	
	

	// define the return result 
	float *d_fit_result;
	float *d_fit_error;
	cudasafe(cudaMalloc((void**)&d_fit_result, Nfitraw1*NV1*sizeof(float)),"Failed cudaMalloc on d_fitresults",__LINE__,fp);
	cudasafe(cudaMemset(d_fit_result, 0, Nfitraw1*NV1*sizeof(float)),"Failed cudaMemset on d_fitresults.",__LINE__,fp);

	cudasafe(cudaMalloc((void**)&d_fit_error, NCRLB*Nfitraw1*sizeof(float)),"Failed cudaMalloc on d_fiterror",__LINE__,fp);
	cudasafe(cudaMemset(d_fit_error, 0, NCRLB*Nfitraw1*sizeof(float)),"Failed cudaMemset on d_fiterror.",__LINE__,fp);

	
	//call the gpu code
	blockx = (int) ceil( (float)Nfitraw1/(float)BlockSize);
	threadx= BlockSize;  //blocksize=64,128 or 256. suggest 128
	printf("%d",blockx);
	printf("%d",threadx);
	dim3 dimBlock(threadx);
	dim3 dimGrid(blockx);
	kernel_gpustorm_fit_wrapper(dimGrid, dimBlock, d_data, d_fit_result,d_fit_error ,iterations1, sz1,fitmode1, psfSigma1);


	// return the fit results from the gpu to cpu
	cudasafe(cudaMemcpy(fit_error, d_fit_error, NCRLB*Nfitraw1*sizeof(float), cudaMemcpyDeviceToHost),
			"cudaMemcpy failed for copyfitresults.",__LINE__,fp);
	cudasafe(cudaMemcpy(fit_result, d_fit_result, Nfitraw1*NV1*sizeof(float), cudaMemcpyDeviceToHost),
			"cudaMemcpy failed for fiterror.",__LINE__,fp);  
	
	cudasafe(cudaFree(d_data),"cudaFree failed on d_data.",__LINE__,fp);
	cudasafe(cudaFree(d_fit_result),"cudaFree failed on fit_result.",__LINE__,fp);
	cudasafe(cudaFree(d_fit_error),"cudaFree failed on fit_error",__LINE__,fp); 
	
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
		fprintf(fp1,"GPU2dFit:cudaFail %s failed \n",str);
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
		fprintf(fp1,"GPU2dFit:nodriver","Could not query CUDA driver version\n");
	}
	if (cudaSuccess == cudaRuntimeGetVersion(&runtimeVersion)) {
		if  (silent==0)
			fprintf(fp1,"CUDA driver version: %d\n", runtimeVersion);

	} else { 
		fprintf(fp1,"GPU2dFit:noruntime","Could not query CUDA runtime version\n");
	}
	if (cudaSuccess == cudaGetDeviceCount(&deviceCount)) {
		if  (silent==0)
			fprintf(fp1,"CUDA devices detected: %d\n", deviceCount);

	} else {
		fprintf(fp1,"GPU2dFit:nodevices","Could not query CUDA device count\n", runtimeVersion);
	}
	if (deviceCount < 1) {
		fprintf(fp1,"GPU2dFit:NoDevice","No CUDA capable devices were detected");
	}
}
