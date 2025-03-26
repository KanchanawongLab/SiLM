

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


__global__ void kernel_saimfitz_lsezrange(float *data, const float *absrTE, const float *anglerTE, const float *zfact, float *fitresult, float *fiterror,const int iterations, int sz, const int *zrange);

//wrapper function for calling cuda code
extern void kernel_saimfitz_wrapper(dim3 dimGrid, dim3 dimBlock, float *data, const float *absrTE, const float *anglerTE, const float *zfact, float *fitresult,float *fiterror ,const int iterations, int sz, int fitmode, const int *zrange)
{

	kernel_saimfitz_lsezrange <<<dimGrid, dimBlock >>> (data, absrTE, anglerTE, zfact, fitresult, fiterror,iterations, sz, zrange);

}
