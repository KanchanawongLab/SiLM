
// Derivative for Gaussian modified from Keith Lidke
//"Fast, single-molecule localization that achieves theoretically minimum uncertainty." 
//C. Simth, N. Joseph, B. Rieger & K. Lidke. Nat. Methods, 7, 373, 2010
#include "GPUgaussLib.h"
#include "definitions.h"

__device__ void kernel_MatInvN(float *M, float *Minv, float *DiagMinv, int szz);
__device__ void kernel_stormMaxMin(const float * data, float *Maxda, float *Minda,const int sz1);
__device__ float kernel_cumax(float a, float b);
__device__ float kernel_cumin(float a, float b);

//	 /f$ \frac{1}{2}(erf((ii-x+frac{1}{2})*sqrt(\frac{1}{2}\sigma^2}))-erf((ii-x-frac{1}{2})*sqrt(\frac{1}{2}\sigma^2})))) /f$

//*******************************************************************************************
__device__ inline float kernel_IntGauss1D(const int ii, const float x, const float sigma) {
	/*! 
	 * \brief /f$ \frac{1}{2} /f$
	 * \param ii ???
	 * \param x ???
	 * \param sigma sigma value of the PSF
	 * \return float
	 */
	const float norm=1.0f/2.0f/sigma/sigma;
    return 1.0f/2.0f*(erf((ii-x+0.5f)*sqrt(norm))-erf((ii-x-0.5f)*sqrt(norm)));
}

//*******************************************************************************************
__device__ inline float kernel_alpha(const float z, const float Ax, const float Bx, const float d){
	/*! 
	 * \brief compute coefficient for alpha
	 * \param z ???
	 * \param Ax ???
	 * \param Bx ???
	 * \param d ???
	 * \return float alpha value
	 */
	
	return 1.0f+pow(z/d, 2)+Ax*pow(z/d, 3)+Bx*pow(z/d, 4);
}

//*******************************************************************************************
__device__ inline float kernel_dalphadz(const float z, const float Ax, const float Bx, const float d){
	/*! 
	 * \brief compute first derivative for alpha in relation to z
	 * \param z ???
	 * \param Ax ???
	 * \param Bx ???
	 * \param d ???
	 * \return float alpha value
	 */
    return (2.0f*z/(d*d) + 3.0f*Ax*pow(z, 2)/(d*d*d)+4.0f*Bx*pow(z, 3)/pow(d, 4));
}

//*******************************************************************************************
__device__ inline float kernel_d2alphadz2(const float z, const float Ax, const float Bx, const float d){
	/*! 
	 * \brief compute second derivative for alpha in relation to z
	 * \param z ???
	 * \param Ax ???
	 * \param Bx ???
	 * \param d ???
	 * \return float alpha value
	 */
    return (2.0f/(d*d) + 6.0f*Ax*z/(d*d*d)+12.0f*Bx*pow(z, 2)/pow(d, 4));
}

//*******************************************************************************************
__device__ inline void kernel_DerivativeIntGauss1D(const int ii, const float x, const float sigma, const float N,
        const float PSFy, float *dudt, float *d2udt2) {
	/*! 
	 * \brief compute the derivative of the 1D gaussian
	 * \param ii ???
	 * \param x ???
	 * \param sigma ???
	 * \param N ???
	 * \param PSFy ???
	 * \param dudt ???
	 * \param d2udt2 ???
	 */    
    float a, b;
    a = exp(-1.0f/2.0f*pow(((ii+0.5f-x)/sigma), 2.0f));
    b = exp(-1.0f/2.0f*pow((ii-0.5f-x)/sigma, 2.0f));
    
    *dudt = -N/sqrt(2.0f*pi)/sigma*(a-b)*PSFy;
    
    if (d2udt2)
        *d2udt2 =-N/sqrt(2.0f*pi)/pow(sigma, 3)*((ii+0.5f-x)*a-(ii-0.5f-x)*b)*PSFy;
}

//*******************************************************************************************
__device__ inline void kernel_DerivativeIntGauss1DSigma(const int ii, const float x, 
        const float Sx, const float N, const float PSFy, float *dudt, float *d2udt2) {
	/*! 
	 * \brief compute the derivative of the 1D gaussian
	 * \param ii ???
	 * \param x ???
	 * \param Sx ???
	 * \param N ???
	 * \param PSFy ???
	 * \param dudt ???
	 * \param d2udt2 ???
	 */    
    
    float ax, bx;
    
    ax = exp(-1.0f/2.0f*pow(((ii+0.5f-x)/Sx), 2.0f));
    bx = exp(-1.0f/2.0f*pow((ii-0.5f-x)/Sx, 2.0f)); 
    *dudt = -N/sqrt(2.0f*pi)/Sx/Sx*(ax*(ii-x+0.5f)-bx*(ii-x-0.5f))*PSFy;
    
    if (d2udt2)
        *d2udt2 =-2.0f/Sx*dudt[0]-N/sqrt(2.0f*pi)/pow(Sx, 5)*(ax*pow((ii-x+0.5f), 3)-bx*pow((ii-x-0.5f), 3))*PSFy;
}

//*******************************************************************************************
__device__ inline void kernel_DerivativeIntGauss2DSigma(const int ii, const int jj, const float x, const float y,
        const float S, const float N, const float PSFx, const float PSFy, float *dudt, float *d2udt2) {
	/*! 
	 * \brief compute the derivative of the 2D gaussian
	 * \param ii ???
	 * \param jj ???
	 * \param x ???
	 * \param y ???
	 * \param S ???
	 * \param N ???
	 * \param PSFx ???
	 * \param PSFy ???
	 * \param dudt ???
	 * \param d2udt2 ???
	 */    
    
    float dSx, dSy, ddSx, ddSy;
    
    kernel_DerivativeIntGauss1DSigma(ii,x,S,N,PSFy,&dSx,&ddSx);
    kernel_DerivativeIntGauss1DSigma(jj,y,S,N,PSFx,&dSy,&ddSy);
   
    *dudt    = dSx+dSy;
    if (d2udt2) *d2udt2 =ddSx+ddSy;
    
}


//*******************************************************************************************
__device__ inline void kernel_CenterofMass2D(const int sz, const float *data, float *x, float *y) {
	/*!
	 * \brief compute the 2D center of mass of a subregion
	 * \param sz nxn size of the subregion
	 * \param data subregion to search
	 * \param x x coordinate to return
	 * \param y y coordinate to return
	 */
    float tmpx=0.0f;
    float tmpy=0.0f;
    float tmpsum=0.0f;
    int ii, jj;
    
    for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++) {
        tmpx+=data[sz*jj+ii]*ii;
        tmpy+=data[sz*jj+ii]*jj;
        tmpsum+=data[sz*jj+ii];
    }
    *x=tmpx/tmpsum;
    *y=tmpy/tmpsum;
}

//*******************************************************************************************
__device__ inline void kernel_GaussFMaxMin2D(const int sz, const float sigma, const float * data, float *MaxN, float *MinBG) {
    /*!
	 * \brief returns filtered min and pixels of a given subregion
	 * \param sz nxn size of the subregion
	 * \param sigma used in filter calculation
	 * \param data the subregion to search
	 * \param MaxN maximum pixel value
	 * \param MinBG minimum background value
	 */
    int ii, jj, kk, ll;
    //float filteredpixel=0, sum=0;
    *MaxN=0.0f;
    *MinBG=10e10f; //big
    
    float norm=1.0f/2.0f/sigma/sigma;
    //loop over all pixels
    for (kk=0;kk<sz;kk++) for (ll=0;ll<sz;ll++){
        //filteredpixel=0.0f;
        //sum=0.0f;
        //for (ii=0;ii<sz;ii++) for(jj=0;jj<sz;jj++){
        //    filteredpixel+=exp(-pow((float)(ii-kk), 2)*norm)*exp(-pow((float)(ll-jj), 2)*norm)*data[ii*sz+jj];
        //    sum+=exp(-pow((float)(ii-kk), 2)*norm)*exp(-pow((float)(ll-jj), 2)*norm);
        //}
        //filteredpixel/=sum;
        //
        //*MaxN=max(*MaxN, filteredpixel);
        //*MinBG=min(*MinBG, filteredpixel);
		*MaxN=max(*MaxN, data[kk*sz+ll]);
        *MinBG=min(*MinBG, data[kk*sz+ll]);
    }
}


////**********************************************************************************************************************
//
//__device__ inline void kernel_DerivativeGauss2D(int ii, int jj, float* theta, float *dudt, float *model) {
//	float PSFx, float PSFy;
//	PSFx=kernel_IntGauss1D(ii, theta[0], theta[4]);
//	PSFy=kernel_IntGauss1D(jj, theta[1], theta[4]);
//	*model=theta[3]+theta[2]*PSFx*PSFy;
//	kernel_DerivativeIntGauss1D(ii, theta[0], theta[4], theta[2], PSFy, &dudt[0],NULL);
//	kernel_DerivativeIntGauss1D(jj, theta[1], theta[4], theta[2], PSFx, &dudt[1],NULL);
//	kernel_DerivativeIntGauss2DSigma(ii, jj,  theta[0], theta[1], theta[4], theta[2], PSFx, PSFy, &dudt[4],NULL);
//	dudt[2] = PSFx*PSFy;
//	dudt[3] = 1.0f;
//
//}


//**********************************************************************************************************************

__device__ inline void kernel_DerivativeGauss2D(int ii, int jj, float PSFSigma,float* theta, float *dudt, float *model) {
	float PSFx, PSFy;
	PSFx=kernel_IntGauss1D(ii, theta[0], PSFSigma);
	PSFy=kernel_IntGauss1D(jj, theta[1], PSFSigma);

	*model=theta[3]+theta[2]*PSFx*PSFy;

	//calculating derivatives
	kernel_DerivativeIntGauss1D(ii, theta[0], PSFSigma, theta[2], PSFy, &dudt[0], NULL);
	kernel_DerivativeIntGauss1D(jj, theta[1], PSFSigma, theta[2], PSFx, &dudt[1], NULL);
	dudt[2] = PSFx*PSFy;
	dudt[3] = 1.0f;
}



//**********************************************************************************************************************

__device__ inline void kernel_DerivativeGauss2D_sigma(int ii, int jj, float* theta, float *dudt, float *model) {
	float PSFx, PSFy;
	PSFx=kernel_IntGauss1D(ii, theta[0], theta[4]);
	PSFy=kernel_IntGauss1D(jj, theta[1], theta[4]);
	*model=theta[3]+theta[2]*PSFx*PSFy;
	kernel_DerivativeIntGauss1D(ii, theta[0], theta[4], theta[2], PSFy, &dudt[0],NULL);
	kernel_DerivativeIntGauss1D(jj, theta[1], theta[4], theta[2], PSFx, &dudt[1],NULL);
	kernel_DerivativeIntGauss2DSigma(ii, jj,  theta[0], theta[1], theta[4], theta[2], PSFx, PSFy, &dudt[4],NULL);
	dudt[2] = PSFx*PSFy;
	dudt[3] = 1.0f;
}

//*******************************************************************************************
__device__ inline void kernel_DerivativeIntGauss2Dz(const int ii, const int jj, const float *theta,
	const float PSFSigma_x, const float PSFSigma_y, const float Ax, const float Ay, 
	const float Bx, const float By, const float gamma, const float d, float *pPSFx, float *pPSFy, float *dudt, float *d2udt2,float *model) {
		/*! 
		* \brief compute the derivative of the 2D gaussian
		* \param ii ???
		* \param jj ???
		* \param theta ???
		* \param PSFSigma_x ???
		* \param PSFSigma_y ???
		* \param Ax ???
		* \param Ay ???
		* \param Bx ???
		* \param By ???
		* \param gamma ???
		* \param d ???
		* \param pPSFx ???
		* \param pPSFy ???
		* \param dudt ???
		* \param d2udt2 ???
		*/    

		float Sx, Sy, dSx, dSy, ddSx, ddSy, dSdzx, dSdzy,ddSddzx,ddSddzy;
		float z, PSFx, PSFy,alphax,alphay,ddx,ddy;
		float dSdalpha_x,dSdalpha_y,d2Sdalpha2_x,d2Sdalpha2_y;
		z=theta[4];

		alphax  = kernel_alpha(z-gamma, Ax, Bx, d);
		alphay  = kernel_alpha(z+gamma, Ay, By, d);

		Sx=PSFSigma_x*sqrt(alphax);
		Sy=PSFSigma_y*sqrt(alphay);

		PSFx=kernel_IntGauss1D(ii, theta[0], Sx);
		PSFy=kernel_IntGauss1D(jj, theta[1], Sy);
		*pPSFx=PSFx;
		*pPSFy=PSFy;

		kernel_DerivativeIntGauss1D(ii, theta[0], Sx, theta[2], PSFy, &dudt[0], &ddx);
		kernel_DerivativeIntGauss1D(jj, theta[1], Sy, theta[2], PSFx, &dudt[1], &ddy);
		kernel_DerivativeIntGauss1DSigma(ii, theta[0], Sx, theta[2], PSFy, &dSx, &ddSx);
		kernel_DerivativeIntGauss1DSigma(jj, theta[1], Sy, theta[2], PSFx, &dSy, &ddSy);

		dSdalpha_x=PSFSigma_x/2.0f/sqrt(alphax);
		dSdalpha_y=PSFSigma_y/2.0f/sqrt(alphay);

		dSdzx  = dSdalpha_x*kernel_dalphadz(z-gamma, Ax, Bx, d); 
		dSdzy  = dSdalpha_y*kernel_dalphadz(z+gamma, Ay, By, d);
		dudt[4] = dSx*dSdzx+dSy*dSdzy;

		dudt[2] = PSFx*PSFy;
		dudt[3] = 1.0f;

		*model=theta[3]+theta[2]*PSFx*PSFy;
		if (d2udt2){
			d2udt2[0] =ddx;
			d2udt2[1] =ddy;

			d2Sdalpha2_x=-PSFSigma_x/4.0f/pow(alphax,1.5f);
			d2Sdalpha2_y=-PSFSigma_y/4.0f/pow(alphay,1.5f);

			ddSddzx  = d2Sdalpha2_x*pow(kernel_dalphadz(z-gamma, Ax, Bx, d),2)+dSdalpha_x*kernel_d2alphadz2(z-gamma, Ax, Bx, d); 
			ddSddzy  = d2Sdalpha2_y*pow(kernel_dalphadz(z+gamma, Ay, By, d),2)+dSdalpha_y*kernel_d2alphadz2(z+gamma, Ay, By, d); 

			d2udt2[4] =ddSx*(dSdzx*dSdzx)+dSx*ddSddzx+
				ddSy*(dSdzy*dSdzy)+dSy*ddSddzy;
			d2udt2[2] = 0.0f;
			d2udt2[3] = 0.0f;
		}
}


//**********************************************************************************************************************

__device__ inline void kernel_DerivativeGauss2D_sigmaxy(int ii, int jj,float* theta, float *dudt, float *model) {
	float PSFx, PSFy;
	PSFx=kernel_IntGauss1D(ii, theta[0], theta[4]);
	PSFy=kernel_IntGauss1D(jj, theta[1], theta[5]);
	kernel_DerivativeIntGauss1D(ii, theta[0], theta[4], theta[2], PSFy, &dudt[0], NULL);
	kernel_DerivativeIntGauss1D(jj, theta[1], theta[5], theta[2], PSFx, &dudt[1], NULL);
	kernel_DerivativeIntGauss1DSigma(ii, theta[0], theta[4], theta[2], PSFy, &dudt[4], NULL);
	kernel_DerivativeIntGauss1DSigma(jj, theta[1], theta[5], theta[2], PSFx, &dudt[5], NULL);
	dudt[2] = PSFx*PSFy;
	dudt[3] = 1.0f;

	*model=theta[3]+theta[2]*PSFx*PSFy;
}


__device__ inline void kernel_DerivativeGauss2D_nbg(int ii, int jj,float* theta, float* pold1, float *dudt, float *model) {
	float PSFx, PSFy;
	PSFx=kernel_IntGauss1D(ii, pold1[0], pold1[4]);
	PSFy=kernel_IntGauss1D(jj, pold1[1], pold1[5]);
	
	dudt[0] = PSFx*PSFy;
	dudt[1] = 1.0f;

	*model=theta[1]+theta[0]*PSFx*PSFy;

}


__device__ inline void kernel_DerivativeGauss_nbg(int ii, int jj,float* theta, float* pold1, float *dudt, float *model) {
	float PSFx, PSFy;
	PSFx=kernel_IntGauss1D(ii, pold1[0], pold1[4]);
	PSFy=kernel_IntGauss1D(jj, pold1[1], pold1[4]);
	
	dudt[0] = PSFx*PSFy;
	dudt[1] = 1.0f;

	*model=theta[1]+theta[0]*PSFx*PSFy;

}


__device__ inline void kernel_MatInvN(float *M, float *Minv, float *DiagMinv, int szz) {
	/*! 
	 * \brief nxn partial matrix inversion
	 * \param M matrix to inverted
	 * \param Minv inverted matrix result
	 * \param DiagMinv just the inverted diagonal
	 * \param sz size of the matrix
	 */
    int ii, jj, kk, num, b;
    float tmp1=0;
    float yy[100];
    
    for (jj = 0; jj < szz; jj++) {
        //calculate upper matrix
        for (ii=0;ii<=jj;ii++)
            //deal with ii-1 in the sum, set sum(kk=0->ii-1) when ii=0 to zero
            if (ii>0) {
            for (kk=0;kk<=ii-1;kk++) tmp1+=M[ii+kk*szz]*M[kk+jj*szz];
            M[ii+jj*szz]-=tmp1;
            tmp1=0;
            }
  
        for (ii=jj+1;ii<szz;ii++)
            if (jj>0) {
            for (kk=0;kk<=jj-1;kk++) tmp1+=M[ii+kk*szz]*M[kk+jj*szz];
            M[ii+jj*szz]=(1/M[jj+jj*szz])*(M[ii+jj*szz]-tmp1);
            tmp1=0;
            }
            else { M[ii+jj*szz]=(1/M[jj+jj*szz])*M[ii+jj*szz]; }
    }
 
    tmp1=0;
    
    for (num=0;num<szz;num++) {
        // calculate yy
        if (num==0) yy[0]=1;
        else yy[0]=0;
        
        for (ii=1;ii<szz;ii++) {
            if (ii==num) b=1;
            else b=0;
            for (jj=0;jj<=ii-1;jj++) tmp1+=M[ii+jj*szz]*yy[jj];
            yy[ii]=b-tmp1;
            tmp1=0;    
        }
        
        // calculate Minv
        Minv[szz-1+num*szz]=yy[szz-1]/M[(szz-1)+(szz-1)*szz];
        
        for (ii=szz-2;ii>=0;ii--) {
            for (jj=ii+1;jj<szz;jj++) tmp1+=M[ii+jj*szz]*Minv[jj+num*szz];
            Minv[ii+num*szz]=(1/M[ii+ii*szz])*(yy[ii]-tmp1);
            tmp1=0;
        }
    }

    
    //if (DiagMinv) for (ii=0;ii<szz;ii++) DiagMinv[ii]=Minv[ii*szz+ii];
    for (ii=0;ii<szz;ii++) DiagMinv[ii]=Minv[ii*szz+ii];

    return;
    
}




__device__ inline void kernel_stormMaxMin(const float * idata, float *Maxda, float *Minda,const int sz1) {
    
	 
    int ii;
   
    *Maxda=0.0f;
    *Minda=10e10f; 
   
    for (ii=0;ii<sz1*sz1;ii++) {
        
		*Maxda=kernel_cumax(*Maxda, idata[ii]);
        *Minda=kernel_cumin(*Minda, idata[ii]);
    }
	return;

	
}


__device__ inline float kernel_cumax(float a, float b) {
    
	float c;
	c= (((a) > (b)) ? (a) : (b));

	return c;
}

__device__ inline float kernel_cumin(float a, float b) {
    
	float c;
	c= (((a) < (b)) ? (a) : (b));

	return c;
}