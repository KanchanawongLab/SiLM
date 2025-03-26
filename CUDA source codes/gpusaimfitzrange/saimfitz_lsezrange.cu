//CUDA codes for fitting the z position in SiLM
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gpusaimlib.cuh"



// input
//data:  pointer of the dataset to be fitted
//absrTE, anglerTE, zfact: constant parameters in scanning angle interference illumination
//Nfitraw: number of data points to be fitted
//sz: number of illumination angles in scanning angle interference experiments
//iterations: number of interations of the optimization algorithm
//zrange: the position is set to zrange(0)*40nm and zrange(1)*40nm. For zrange [0:5], the fit position is between 0 and 200nm

//output
//fit_result: output fitresults 
//fit_error: crlb 


__global__ void kernel_saimfitz_lsezrange(float *data,const float *absrTE,const float *anglerTE,const float *zfact,float *fitresult,float *fiterror,const int iterations, int sz, const int *zrange){
	
	const int NV=2;
	float dudt[NV];
	//printf("%d\n",sz);

	int tx=threadIdx.x;
	int bx=blockIdx.x;
	int BlockSize=blockDim.x;

	int tid=blockDim.x*blockIdx.x+threadIdx.x;

	//float NR_Numerator[NV], NR_Denominator[NV];
	float theta[NV];
	float maxjump[NV]={20.0f,100.0f};
	float gamma[NV]={1.0f,1.0f};
	//float *sdata=data+sz*(ii-1); this one is for cpu

	const float *sdata=data+sz*(bx*BlockSize+tx);
	const int sz1=sz;
	

	//set the maximum to intensity, and minimum to the background
	float Nmin;
	kernel_saimMaxMin(sdata, &theta[1], &Nmin,sz1);

	// initial value of the fitting, z, I ,bg
	//theta[0]=100.0f;
	//theta[1]=theta[1]/2;  // here 2 means that the model maximum value around 2
	//theta[2]=*minda;

	float model, modela,datai;
	float zfact1;
	float zerr=1e10; float oldzerr=1e10;
	float zguess;
	float zval;
	//same definition of initial value as numbafit
	float amptemp;
	amptemp = (theta[1] -Nmin)/ 2;
	theta[1]=amptemp;
	int iii=0; int jjj=0;
	float thetaGlob[NV]; float errGlob=1e15; 

	int m=0;
	float new_err=1e12; 
	float old_err=1e15;
	float lambda=0.1, scale_up=10,scale_down=0.1,acceptance=1.1;
	float new_lambda=lambda;
	float old_lambda;
	float tolerance=1e-6;
	float new_update[NV]={1e13f,1e13f};
	float old_update[NV];
	float old_theta[NV];

	float M_jacob[NV]={0};
	float M_hession[NV][NV]={1,0,0,2};
	float Minv_hession[NV][NV];
	float diagMinv_hession[NV];
	int p=0; int q=0;
	//float *merr;
	kernel_MatInvN(*M_hession, *Minv_hession, diagMinv_hession, NV) ;



	for (iii = 0; iii < zrange[1]-zrange[0]; iii++) {

		theta[0] = 40.0*(iii+zrange[0]);
		theta[1] = amptemp;
		new_err = 1e12;
		old_err = 1e15;
		lambda = 0.1; scale_up = 10; scale_down = 0.1; acceptance = 1.1;
		new_lambda = lambda;


		int p = 0; int q = 0;
		float t1 = 0.0f, t2 = 0.0f;
		float mu = 1 + new_lambda;
		int kk1 = 0;

		for (kk1 = 0; kk1 < iterations; kk1++)
		{

			memset(M_jacob, 0, NV * sizeof(float));
			memset(M_hession, 0, NV*NV * sizeof(float));
			for (int jj = 0; jj < NV; jj++)
			{
				old_update[jj] = new_update[jj];
				old_theta[jj] = theta[jj];
			}
			old_lambda = new_lambda;
			old_err = new_err;

			for (m = 0; m < sz; m++) {
				//zfact1=(1e-3)*(zfact[m]);
				zfact1 = zfact[m];
				modela = (1 + pow(absrTE[m], 2) + 2 * absrTE[m] * cos(zfact1*theta[0] + anglerTE[m]));
				model = theta[1] * modela;
				datai = sdata[m];



				dudt[1] = modela;
				dudt[0] = theta[1] * (-2 * zfact1*absrTE[m] * sin(zfact1*theta[0] + anglerTE[m]));

				if (model > 10e-3) {
					//t1=1-datai/model;
					//t2=datai/pow(model,2);
					t1 = model - datai;
					t2 = 1.0;
				}

				for (p = 0; p < NV; p++)
				{
					M_jacob[p] = M_jacob[p] + t1 * dudt[p];
					for (q = 0; q < NV; q++)
					{
						M_hession[p][q] = M_hession[p][q] + t2 * dudt[p] * dudt[q];
					}
				}

			}

			for (p = 0; p < NV; p++) {
				M_hession[p][p] = M_hession[p][p] + (mu - 1);
			}

			kernel_MatInvN(*M_hession, *Minv_hession, diagMinv_hession, NV);

			for (p = 0; p < NV; p++)
			{
				new_update[p] = 0;
				for (q = 0; q < NV; q++)
				{
					new_update[p] = new_update[p] + Minv_hession[p][q] * M_jacob[q];

				}
			}

			for (p = 0; p < NV; p++) {
				//new_update[p]=M_jacob[p]/diagMinv_hession[p];
				if (new_update[p] / old_update[p] < -0.5)
					maxjump[p] = 0.5*maxjump[p];

				new_update[p] = new_update[p] / (1 + abs(new_update[p] / maxjump[p]));
				theta[p] = theta[p] - new_update[p];
			}



			theta[0] = kernel_cumax(theta[0], zrange[0]*40);
			theta[0] = kernel_cumin(theta[0], zrange[1]*40);


			theta[1] = kernel_cumax(theta[1], 1);



			new_err = 0;
			for (m = 0; m < sz; m++) {
				//zfact1=(1e-3)*(zfact[m]);
				zfact1 = zfact[m];
				modela = (1 + pow(absrTE[m], 2) + 2 * absrTE[m] * cos(zfact1*theta[0] + anglerTE[m]));
				model = theta[1] * modela;
				datai = sdata[m];
				new_err = new_err + pow((model - datai), 2);
			}


			//break if it converged
			if (abs((new_err - old_err) / new_err) < 1e-10) {

				break;
			}


			if (new_err > acceptance*old_err) {
				for (p = 0; p < NV; p++) {
					theta[p] = old_theta[p];
					new_update[p] = old_update[p];
				}
				new_lambda = old_lambda;
				new_err = old_err;
				mu = kernel_cumax((1 + new_lambda * scale_up) / (1 + new_lambda), 1.3);
				new_lambda = scale_up * new_lambda;
			}

			
		}

		if (new_err < errGlob)
		{
			errGlob = new_err;
			for (m = 0; m < NV; m++) {
				thetaGlob[m] = theta[m];
			}

		}

	}

	for (m=0;m<NV;m++){
		theta[m]=thetaGlob[m];
		}


	
    
	float Mcrlb[NV][NV]={0}; float Mcrlb_inv[NV][NV];
	float diagMcrlb_inv[NV];
	//calculate CRLB model and error 
	new_err=0;
	for (m = 0; m < sz; m++) {
		//zfact1=(1e-3)*(zfact[m]);
		zfact1 = zfact[m];
		modela = (1 + pow(absrTE[m], 2) + 2 * absrTE[m] * cos(zfact1*theta[0] + anglerTE[m]));
		model = theta[1] * modela;
		datai = sdata[m];



		dudt[1] = modela;
		dudt[0] = theta[1] * (-2 * zfact1*absrTE[m] * sin(zfact1*theta[0] + anglerTE[m]));


		new_err = new_err + pow((model - datai), 2);

		for (p = 0; p < NV; p++)
			for (q = 0; q < NV; q++)
			{
				{
					Mcrlb[p][q] = Mcrlb[p][q] + dudt[p] * dudt[q] / (model);
				}
			}


	}

	kernel_MatInvN(*Mcrlb, *Mcrlb_inv, diagMcrlb_inv, NV) ;

    fiterror[(tid)*3+0]=new_err;
	for(m=0;m<NV;m++)
	{
		fiterror[(tid)*3+m+1]=diagMcrlb_inv[m];
	}

	


	for (m = 0; m < NV; m++) {
		fitresult[(tid)*NV + m] = theta[m];
	}
	
}
       
