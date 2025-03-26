

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



__global__ void kernel_gpustorm_fitnbg(float *data,    float *fitresult, float *fiterror,const int iterations, const int sz,  float *pold);
__global__ void kernel_gpustorm_fitxynbg(float *data,    float *fitresult, float *fiterror,const int iterations, const int sz,  float *pold);


//gpufit wrapper, call different gpu code for different fit mode, fitmode 1, the psf is fit with two sigma, while others are fitted with one sigma
extern void kernel_gpustorm_repeatfit_wrapper(dim3 dimGrid, dim3 dimBlock, float *data, float *fitresult,float *fiterror ,const int iterations, const int sz, int fitmode,float *pold)
{
	//gpu refit routine when psf is fit with two sigma sx and sy
	if(fitmode==1){
		kernel_gpustorm_fitxynbg <<<dimGrid, dimBlock >>> (data, fitresult, fiterror,iterations, sz, pold);}
	//gpu refit routine when psf is fit with one sigma sxy
	else
	{
		kernel_gpustorm_fitnbg <<<dimGrid, dimBlock >>> (data, fitresult, fiterror,iterations, sz,pold);
	}


}