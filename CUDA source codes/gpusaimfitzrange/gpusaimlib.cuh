
__device__ void kernel_MatInvN(float *M, float *Minv, float *DiagMinv, int szz);
__device__ void kernel_saimMaxMin(const float * data, float *Maxda, float *Minda,const int sz1);
__device__ float kernel_cumax(float a, float b);
__device__ float kernel_cumin(float a, float b);


//CUDA code for calculaing the matrix inversion

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
    float yy[25];
    
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

__device__ inline void kernel_saimMaxMin(const float * idata, float *Maxda, float *Minda,const int sz1) {
    
	 
    int ii;
   
    *Maxda=0.0f;
    *Minda=10e10f; 
   
    for (ii=0;ii<sz1;ii++) {
        
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