#include "cufft.h"
#include "GPU_devstruct.h"

__global__ void padKernel(
  const FD_TYPE* MValue, FD_TYPE* dev_Mtemp,
	const FD_TYPE* dev_Ms, const int rdimx, const int rdimxy, const int rsize,
	const int cdimx, const int cdimxy, const int csize){

		int gid = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x)*blockDim.x;

		if( gid >= rsize )
			return;

		int rx = gid % rdimx;
		int rz = gid/rdimxy;
		int ry = (gid - rz*rdimxy)/rdimx;

		int c_index = rx + ry*cdimx + rz*cdimxy;
		for( int l = 0; l < 3; l++ )
			dev_Mtemp[ c_index + l*csize ] = MValue[ gid+l*rsize ]*dev_Ms[gid];
}

#define XX 0
#define XY 1
#define XZ 2
#define YY 3
#define YZ 4
#define ZZ 5

__global__ void multiplicationKernel( FD_CPLX_TYPE* dev_Mtemp, 
  const FD_TYPE* __restrict__ GreenFunc, const int green_x, const int green_y,
  const int green_z, const int cdimx, const int cdimy, const int cdimz){
		
		const int gid = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x)*blockDim.x;
		
		const int csize = cdimx*cdimy*cdimz;
		if( gid >= csize )
			return;

		int flag_y = 1;
		int flag_z = 1;
		const int i = gid % cdimx;
		int k = gid / (cdimx*cdimy);
		int j = (gid - k*cdimx*cdimy)/cdimx;
		if( j >= green_y ){
			j = cdimy - j;
			flag_y = -1;
		}
		if( k >= green_z){
			k = cdimz - k;
			flag_z = -1;
		}
		const int green_N3 = green_x*green_y*green_z;
		const int greenid = i+j*green_x+k*green_x*green_y;

		const FD_TYPE mxreal = dev_Mtemp[ gid ].x;
		const FD_TYPE mximg  = dev_Mtemp[ gid ].y;
		const FD_TYPE myreal = dev_Mtemp[ gid+csize ].x;
		const FD_TYPE myimg  = dev_Mtemp[ gid+csize ].y;
		const FD_TYPE mzreal = dev_Mtemp[ gid+2*csize ].x;
		const FD_TYPE mzimg  = dev_Mtemp[ gid+2*csize ].y;
    
		const FD_TYPE green_xx = GreenFunc[ greenid + XX*green_N3 ];
		const FD_TYPE green_xy = flag_y*GreenFunc[ greenid + XY*green_N3 ];
		const FD_TYPE green_xz = flag_z*GreenFunc[ greenid + XZ*green_N3 ];
		const FD_TYPE green_yy = GreenFunc[ greenid + YY*green_N3 ];
		const FD_TYPE green_yz = flag_y*flag_z*GreenFunc[ greenid + YZ*green_N3 ];
		const FD_TYPE green_zz = GreenFunc[ greenid + ZZ*green_N3 ];
    
		const FD_TYPE real0	= mxreal*green_xx	+ myreal*green_xy	+ mzreal*green_xz;
    
		const FD_TYPE img0	= mximg*green_xx	+ myimg*green_xy	+ mzimg*green_xz;
    
		const FD_TYPE real1	= mxreal*green_xy	+ myreal*green_yy	+ mzreal*green_yz;
    
		const FD_TYPE img1	= mximg*green_xy	+ myimg*green_yy	+ mzimg*green_yz;
		
		const FD_TYPE real2	= mxreal*green_xz	+ myreal*green_yz	+ mzreal*green_zz;
    
		const FD_TYPE img2	= mximg*green_xz	+ myimg*green_yz	+ mzimg*green_zz;
		
		dev_Mtemp[ gid ].x = real0;
		dev_Mtemp[ gid ].y = img0;
		dev_Mtemp[ gid+csize ].x = real1;
		dev_Mtemp[ gid+csize ].y = img1;
		dev_Mtemp[ gid+2*csize ].x = real2;
		dev_Mtemp[ gid+2*csize ].y = img2;
}

#define CONSTANT -6.28318531e-7
__global__ void unpadKernel( FD_TYPE* dev_Hsta, FD_TYPE* dev_Hsta_r, 
	FD_TYPE* dev_MValue, FD_TYPE* dev_Ms, FD_TYPE* dev_Energy, FD_TYPE* dev_Torque, 
	int rdimx, int rdimxy, int rsize, int cdimx, int cdimxy, int csize, 
  const bool outputH, const bool outputE, const bool accumFlag,
  FD_TYPE* dev_energy_loc, FD_TYPE* dev_field_loc){

		int gid = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x)*blockDim.x;

		if( gid >= rsize )
			return;

		int rx = gid % rdimx;
		int rz = gid/rdimxy;
		int ry = (gid - rz*rdimxy)/rdimx;

		int c_index = rx + ry*cdimx + rz*cdimxy;

		FD_TYPE3 H;
		H.x = dev_Hsta_r[ c_index ]/2.f; H.y = dev_Hsta_r[ csize + c_index ]/2.f; 
		H.z = dev_Hsta_r[ 2*csize + c_index ]/2.f; 

    if (outputH) {
	    dev_field_loc[gid] = H.x;	dev_field_loc[gid+rsize] = H.y;	dev_field_loc[gid+2*rsize] = H.z;
    }
    if (accumFlag) {
      dev_Hsta[gid] += H.x;	dev_Hsta[gid+rsize] += H.y;		dev_Hsta[gid+2*rsize] += H.z;
    }

    FD_TYPE3 m0;
    m0.x = dev_MValue[gid]; m0.y = dev_MValue[gid+rsize]; m0.z = dev_MValue[gid+2*rsize]; 

    FD_TYPE e_tmp = CONSTANT*dev_Ms[gid]*(m0.x*H.x + m0.y*H.y + m0.z*H.z);
    if (accumFlag) {
      dev_Energy[gid] += e_tmp;

      dev_Torque[gid] += m0.y*H.z - m0.z*H.y;
      dev_Torque[gid+rsize] += m0.z*H.x - m0.x*H.z;
      dev_Torque[gid+2*rsize] += m0.x*H.y - m0.y*H.x;
    }
    
    if (outputE) {
			dev_energy_loc[gid] = e_tmp;
    }
}

// wrapper for CUDA kernel functions
void pad(const dim3 &grid_size, const dim3 &block_size,
  FD_TYPE* MValue, FD_TYPE* dev_Mtemp, FD_TYPE* dev_Ms, int rdimx, int rdimxy,
  int rsize, int cdimx, int cdimxy, int csize) {
    
  padKernel<<<grid_size, block_size>>>(MValue, dev_Mtemp,
    dev_Ms, rdimx, rdimxy, rsize, cdimx, cdimxy, csize);
}

void multiplication(const dim3 &grid_size, const dim3 &block_size,
    FD_CPLX_TYPE* dev_Mtemp, const FD_TYPE* GreenFunc, 
    const int green_x, const int green_y, const int green_z, const int cdimx, 
    const int cdimy, const int cdimz) {
  
  multiplicationKernel<<<grid_size,block_size>>>(dev_Mtemp, GreenFunc, green_x, 
    green_y, green_z, cdimx, cdimy, cdimz);
}

void unpad(const dim3 &grid_size, const dim3 &block_size,
    FD_TYPE* dev_Hsta, FD_TYPE* dev_Hsta_r, FD_TYPE* dev_MValue, 
    FD_TYPE* dev_Ms, FD_TYPE* dev_Energy, FD_TYPE* dev_Torque, 
    int rdimx, int rdimxy, int rsize, int cdimx, int cdimxy, int csize, 
    const bool outputH, const bool outputE, const bool accumFlag,
    FD_TYPE* dev_energy_loc, FD_TYPE* dev_field_loc) {
  
  unpadKernel<<<grid_size, block_size>>>(dev_Hsta, dev_Hsta_r, dev_MValue, 
      dev_Ms, dev_Energy, dev_Torque, rdimx, rdimxy, rsize, cdimx, cdimxy, csize,
      outputH, outputE, accumFlag, dev_energy_loc, dev_field_loc);  
}

string getPadKernelName() {
  return "padKernel";
}

string getMultiplicationKernelName() {
  return "multiplicationKernel";
}

string getUnpadKernelName() {
  return "unpadKernel";
}