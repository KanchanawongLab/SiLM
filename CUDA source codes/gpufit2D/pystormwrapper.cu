

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//switch different modes of psf fitting
//1: fit with least squre error and two sigma are output
//2: fit with maximum likelihood error and two sigma are output
//3: fit with least squre error and one sigma are output
//4: fit with maximum likelihood error and one sigma are output


__global__ void kernel_gpustorm_fitsigmaxy( float *d_data,  float *fitresult, float *fiterror,const int iterations, const int sz, const float psfSigma);
__global__ void kernel_gpustorm_fitsigma( float *d_data,    float *fitresult, float *fiterror,const int iterations, const int sz, const float psfSigma);
__global__ void kernel_gpustorm_fitsigmaxymle( float *d_data,  float *fitresult, float *fiterror,const int iterations, const int sz, const float psfSigma);
__global__ void kernel_gpustorm_fitsigmamle( float *d_data,    float *fitresult, float *fiterror,const int iterations, const int sz, const float psfSigma);

extern void kernel_gpustorm_fit_wrapper(dim3 dimGrid, dim3 dimBlock,  float *d_data, float *fitresult,float *fiterror , const int iterations,  const int sz,  int fitmode, const float psfSigma)
{
if(fitmode==1){
		kernel_gpustorm_fitsigmaxy <<<dimGrid, dimBlock >>> (d_data, fitresult, fiterror,iterations, sz,psfSigma);}

if(fitmode==2)
	{
	
		kernel_gpustorm_fitsigmaxymle <<<dimGrid, dimBlock >>> (d_data, fitresult, fiterror,iterations, sz,psfSigma);}
	
if(fitmode==3)
	{
	
		kernel_gpustorm_fitsigma <<<dimGrid, dimBlock >>> (d_data, fitresult, fiterror,iterations, sz,psfSigma);}

if(fitmode==4)
	{
	
		kernel_gpustorm_fitsigmamle <<<dimGrid, dimBlock >>> (d_data, fitresult, fiterror,iterations, sz,psfSigma);}


}

