
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "GPUgaussLib.cuh"

//gpu refit routine when psf is fit with two sigma sx and sy
//l-m algorithms are used to solve the optimization problem
__global__ void kernel_gpustorm_fitxynbg(float *data,  float *fitresult, float *fiterror,const int iterations, const int sz, float *pold){

	const int NV=2;
	float dudt[NV];
	//printf("%d\n",sz);

	int tx=threadIdx.x;
	int bx=blockIdx.x;
	int BlockSize=blockDim.x;

	int tid=blockDim.x*blockIdx.x+threadIdx.x;
	
	//pold  x y N bg sigma


	float NR_Numerator[NV], NR_Denominator[NV];
	float theta[NV];
	float maxjump[NV]={100.0f, 2.0f};
	float gamma[NV]={ 0.5f, 1.0f};
	

	const float *sdata=data+sz*sz*(bx*BlockSize+tx);
    float *spold=pold+6*(bx*BlockSize+tx);
	const int sz1=sz;
	//initial value of fitting I, bg,
	float psfSigma=spold[4];
	float Nmax=0;
	kernel_GaussFMaxMin2D(sz, psfSigma, sdata, &Nmax, &theta[1]);
	theta[0]=kernel_cumax(0.0, (Nmax-theta[1])*2*pi*psfSigma*psfSigma);
	theta[1] = kernel_cumax(theta[1],0.01);
	
	
	maxjump[0]=kernel_cumax(maxjump[0],theta[0]);
    maxjump[1]=kernel_cumax(maxjump[1],theta[1]);


	float new_err=1e13; 
	float old_err=1e15;
	float lambda=0.1, scale_up=10,scale_down=0.1,acceptance=1.1;
	float new_lambda=lambda;
	float old_lambda;
	float tolerance=1e-6;
	float new_update[NV]={1e13f,1e13f};
	float old_update[NV];
	float old_theta[NV];

	float M_jacob[NV]={0};
	float M_hession[NV][NV]={1};
	float Minv_hession[NV][NV];
	float diagMinv_hession[NV];
	int p=0; int q=0;
	
	


	float model, modela,datai;
	float zfact1;
	int m=0; float t1=0.0f,t2=0.0f; int mm=0;
	

	float mu=1+new_lambda;
	int kk1=0;

	

	for (kk1=0;kk1<iterations;kk1++)
	{
		
	  memset(M_jacob,0,NV*sizeof(float));
	  memset(M_hession,0,NV*NV*sizeof(float));
		for (int jj=0;jj<NV; jj++)
		{
			old_update[jj]=new_update[jj];
			old_theta[jj]=theta[jj];
		}
		old_lambda=new_lambda;
		old_err=new_err;
		
		for (m=0; m<sz;m++) for (mm=0;mm<sz;mm++) {

			kernel_DerivativeGauss2D_nbg(m,mm,theta,spold,dudt,&model);

			datai=sdata[sz*mm+m];
			
			
			if (model>10e-3){
				t1=1-datai/model;
				t2=datai/pow(model,2);}

			for (p=0; p<NV;p++)
			{
				M_jacob[p]=M_jacob[p]+t1*dudt[p];
				for(q=0; q<NV;q++)
				{
					M_hession[p][q]=M_hession[p][q]+t2*dudt[p]*dudt[q];
				}
			}

			}
		
		for (p=0;p<NV;p++){
			M_hession[p][p]=M_hession[p][p]+(mu-1);}

		kernel_MatInvN(*M_hession, *Minv_hession, diagMinv_hession, NV) ;

       for (p=0; p<NV;p++)
		{
				new_update[p]=0;
				for(q=0; q<NV;q++)
				{
					new_update[p]=new_update[p]+Minv_hession[p][q]*M_jacob[q];

				}
		}
		
		for (p=0;p<NV;p++){
			//new_update[p]=M_jacob[p]/diagMinv_hession[p];
			if (new_update[p]/old_update[p]<-0.5)
				maxjump[p]=0.5*maxjump[p];

			new_update[p]=new_update[p]/(1+abs(new_update[p]/maxjump[p]));
			theta[p]=theta[p]-new_update[p];
		}

	   
		
		theta[0] = kernel_cumax(theta[0],1.0);
		theta[1] = kernel_cumax(theta[1],0.01);
		

		

	
		new_err=0;
		for(m=0; m<sz;m++) for(mm=0;mm<sz;mm++){
			
			kernel_DerivativeGauss2D_nbg(m,mm,theta,spold, dudt,&model);
			datai=sdata[sz*mm+m];
			if (datai>0)
				new_err=new_err+2*((model-datai)-datai*log(model/datai));
			else
				new_err=new_err+2*model;
		}

	

		if (new_err>acceptance*old_err){
			for (p=0;p<NV;p++){
				theta[p]=old_theta[p];
				new_update[p]=old_update[p];
			}
			new_lambda=old_lambda;
			new_err=old_err;
			mu=kernel_cumax((1+new_lambda*scale_up)/(1+new_lambda),1.3);
			new_lambda=scale_up*new_lambda;}

		if (new_err<old_err){
			new_lambda=scale_down*new_lambda;
			mu=1+new_lambda;}

		fiterror[(tid)*iterations+kk1]=new_err;
		
	}

	
    //calculating the crlb and new_err
	float Mcrlb[NV][NV]={0}; float Mcrlb_inv[NV][NV];
	float diagMcrlb_inv[NV];
	new_err=0;
	for (m=0; m<sz;m++) for(mm=0;mm<sz;mm++){
            
			kernel_DerivativeGauss2D_nbg(m,mm,theta,spold, dudt,&model);
			datai=sdata[sz*mm+m];

			if (datai>0)
				new_err=new_err+2*((model-datai)-datai*log(model/datai));
			else
				new_err=new_err+2*model;

			for (p=0;p<NV;p++)
				for(q=0;q<NV;q++)
				{
					{
						Mcrlb[p][q]=Mcrlb[p][q]+dudt[p]*dudt[q]/(model);
					}
				}

		   
	}

	kernel_MatInvN(*Mcrlb, *Mcrlb_inv, diagMcrlb_inv, NV) ; 


	//output fit result
	for (m = 0; m < NV; m++) {
		fitresult[(tid)*NV + m] = theta[m];
	}
	


	//return;  
}
       
