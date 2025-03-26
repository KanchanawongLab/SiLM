//main functions to call cuda codes
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

extern void kernel_gpustorm_repeatfit_wrapper(dim3 dimGrid, dim3 dimBlock, float *d_data, float *fitresult, float *fiterror, const int iterations, const int sz, int fitmode, float *pold);

//void cpu_MatInvN(float * M, float * Minv, float * DiagMinv, int szz) ;

#ifndef max
//! not defined in the C standard used by visual studio
#define max(a,b) (((a) > (b)) ? (a) : (b))
#endif
#ifndef min
//! not defined in the C standard used by visual studio
#define min(a,b) (((a) < (b)) ? (a) : (b))
#endif


#define DLL_EXPORT extern "C" __declspec(dllexport)
//DLL_EXPORT void main(unsigned int *a, int b);
//DLL_EXPORT void pygpu(float *data, float *absrTE, float *anglerTE, float *zfact, int Nfitraw, int sz, int iterations,float *fit_result,float *fit_error);

//test dlls addone which is used while debuging
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
//fitmode 1,psf is fit to have to sigma sx and sy, fitmode 2, psf is fit to have only one sigma sx
//pold: earlier 2D fit results contains the sigma, and 2D center 

//output
//fit_result: the fitresult after repeated fit where the intensity and background are extracted
//fit_error: output lse errors inside each iterations

	DLL_EXPORT void pystorm_repeat(float *data, int Nfitraw, int sz, int iterations, float PSFSigma, float *fit_result,float *fit_error, int fitmode, float* pold){
    int blockx=0;
	int threadx=0;
	
	int NV;  int NP;
	if (fitmode==1) {
		 NV=6; NP=2;}
	else if (fitmode==2) {
	      NV=5; NP=2; 
	}
	
	//debuging codes,checking whether the input dataset are correct. 
	fp=fopen("log1.txt","w");
	fprintf(fp,"sz%d", sz);
	fprintf(fp,"nfitraw%d", Nfitraw);
	fprintf(fp,"iterations%d\n",iterations);
	fprintf(fp,"fitmode%d\n",fitmode);
	fprintf(fp,"psfsigma%5f\n",PSFSigma);
	

	int i;
	for (i=0;i<sz*sz;i++)
	{
	
		fprintf(fp,"%5f\t",data[i]);
		fprintf(fp,"%5f\n",data[sz*sz+i]);
	}

	
		for (i=0;i<NV;i++)
	{
		
		
		fprintf(fp,"%5f\t",pold[i]);
		fprintf(fp,"%5f\n",pold[NV+i]);
		fprintf(fp,"%5f\n",pold[NV*2+i]);
		fprintf(fp,"%5f\n",pold[NV*3+i]);
		}  


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
		printf("GPUmleFit_LM:NotEnoughMemory Trying to allocation %dMB. GPU only has %dMB. Please break your fitting into multiple smaller runs.\n",
		requiredMemory/(1024*1024),availableMemory);

	

	float *d_data;
	cudaDeviceReset();
	cudaSetDevice(0);


	const int sz1=sz;
	const int Nfitraw1=Nfitraw;
	const int iterations1=iterations;
	const int fitmode1=fitmode;
	const float psfSigma1=PSFSigma;
	const int NCRLB=50;

	
	// define the data passing to the GPU
	cudasafe(cudaMalloc((void**)&sz,sizeof(int)),"testcupdamallc",__LINE__,fp);
	cudasafe(cudaMalloc((void**)&d_data, sz1*sz1*Nfitraw1*sizeof(float)),"Failed cudaMalloc on d_data.",__LINE__,fp);
	cudasafe(cudaMemset(d_data, 0, sz1*sz1*Nfitraw1*sizeof(float)),"Failed cudaMemset on d_data.",__LINE__,fp);
	cudasafe(cudaMemcpy(d_data,data,sz1*sz1*Nfitraw1*sizeof(float),cudaMemcpyHostToDevice),"failed cudamemcpy on d_data",__LINE__,fp);
	
	

	// define the return result 
	float *d_fit_result;
	float *d_fit_error;
	cudasafe(cudaMalloc((void**)&d_fit_result, Nfitraw1*NP*sizeof(float)),"Failed cudaMalloc on d_fitresults",__LINE__,fp);
	cudasafe(cudaMemset(d_fit_result, 0, Nfitraw1*NP*sizeof(float)),"Failed cudaMemset on d_fitresults.",__LINE__,fp);

	cudasafe(cudaMalloc((void**)&d_fit_error, NCRLB*Nfitraw1*sizeof(float)),"Failed cudaMalloc on d_fiterror",__LINE__,fp);
	cudasafe(cudaMemset(d_fit_error, 0, NCRLB*Nfitraw1*sizeof(float)),"Failed cudaMemset on d_fiterror.",__LINE__,fp);

	
	//call the gpu code
	blockx = (int) ceil( (float)Nfitraw1/(float)BlockSize);
	threadx= BlockSize;  //blocksize=64,128 or 256. suggest 128

	float *d_pold;
	
	cudasafe(cudaMalloc((void**)&d_pold, Nfitraw1*NV*sizeof(float)),"Failed cudaMalloc on d_pold",__LINE__,fp);
	cudasafe(cudaMemset(d_pold, 0, NV*Nfitraw1*sizeof(float)),"Failed cudaMemset on d_pold.",__LINE__,fp); 
	cudasafe(cudaMemcpy(d_pold,pold,NV*Nfitraw1*sizeof(float),cudaMemcpyHostToDevice),"failed cudamemcpy on d_data",__LINE__,fp);

	
	
	printf("%d",blockx);
	printf("%d",threadx);

	
	dim3 dimBlock(threadx);
	dim3 dimGrid(blockx);

	//call gpu to do the fit
	kernel_gpustorm_repeatfit_wrapper(dimGrid, dimBlock, d_data, d_fit_result,d_fit_error ,iterations1, sz1,fitmode1, d_pold);

	// return the results from gpu
	cudasafe(cudaMemcpy(fit_error, d_fit_error, NCRLB*Nfitraw1*sizeof(float), cudaMemcpyDeviceToHost),
			"cudaMemcpy failed for copyfiterror.",__LINE__,fp);
	cudasafe(cudaMemcpy(fit_result, d_fit_result, Nfitraw1*NP*sizeof(float), cudaMemcpyDeviceToHost),
			"cudaMemcpy failed for fitresults.",__LINE__,fp);  
	
	cudasafe(cudaFree(d_data),"cudaFree failed on d_data.",__LINE__,fp);
	cudasafe(cudaFree(d_fit_result),"cudaFree failed on fit_result.",__LINE__,fp);
	cudasafe(cudaFree(d_fit_error),"cudaFree failed on fit_error",__LINE__,fp); 
	cudasafe(cudaFree(d_pold),"cudaFree failed on d_pold.",__LINE__,fp);

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
		fprintf(fp1,"GPUrepeatfit:cudaFail %s failed \n",str);
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
		fprintf(fp1,"GPUrepeatfit:nodriver","Could not query CUDA driver version\n");
	}
	if (cudaSuccess == cudaRuntimeGetVersion(&runtimeVersion)) {
		if  (silent==0)
			fprintf(fp1,"CUDA driver version: %d\n", runtimeVersion);

	} else { 
		fprintf(fp1,"GPUrepeatfit:noruntime","Could not query CUDA runtime version\n");
	}
	if (cudaSuccess == cudaGetDeviceCount(&deviceCount)) {
		if  (silent==0)
			fprintf(fp1,"CUDA devices detected: %d\n", deviceCount);

	} else {
		fprintf(fp1,"GPUrepeatfit:nodevices","Could not query CUDA device count\n", runtimeVersion);
	}
	if (deviceCount < 1) {
		fprintf(fp1,"GPUrepeatfit:NoDevice","No CUDA capable devices were detected");
	}
}
