#include "director.h"
#include "GPU_devstruct.h"

__global__ void fixedZeemanKernel(FD_TYPE* dev_MValue, FD_TYPE* dev_Ms, 
  FD_TYPE *dev_ZField, FD_TYPE* dev_Field, FD_TYPE *dev_Energy, 
  FD_TYPE* dev_Torque, int size, const bool outputE, const bool accumFlag,
  FD_TYPE* dev_energy_loc) {
	
	int gid = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x)*blockDim.x;
	if (gid >= size)	return;
	
	FD_TYPE3 H;
	H.x = dev_ZField[gid];	H.y = dev_ZField[gid+size];	H.z = dev_ZField[gid + 2*size];
  if (accumFlag) {
    dev_Field[gid] += H.x;	dev_Field[gid+size] += H.y;	dev_Field[gid+2*size] += H.z;
  }
  FD_TYPE3 m0;
  m0.x = dev_MValue[gid]; m0.y = dev_MValue[gid+size]; m0.z = dev_MValue[gid+2*size];
  
  FD_TYPE ei = (-1.0 * MU0) * dev_Ms[gid] * (H.x*m0.x + H.y*m0.y + H.z*m0.z);
  if (accumFlag) {
    dev_Energy[gid] += ei;
  }
		
  if (outputE) {
    dev_energy_loc[gid] = ei;
  }

  if (accumFlag) {
    dev_Torque[gid] += m0.y*H.z - m0.z*H.y;
    dev_Torque[gid+size] += m0.z*H.x - m0.x*H.z;
    dev_Torque[gid+2*size] += m0.x*H.y - m0.y*H.x;
  }
}

// wrapper for CUDA kernels
void Get_Fixed_Zeeman(const dim3 &grid_size, const dim3 &block_size,
    FD_TYPE* dev_MValue, FD_TYPE* dev_Ms, FD_TYPE *dev_ZField,
    FD_TYPE* dev_Field, FD_TYPE *dev_Energy, FD_TYPE* dev_Torque, int size, 
    const bool outputE, const bool accumFlag, FD_TYPE* dev_energy_loc) {
    
  fixedZeemanKernel<<<grid_size, block_size>>>(dev_MValue, dev_Ms, 
    dev_ZField, dev_Field, dev_Energy, dev_Torque, size,
    outputE, accumFlag, dev_energy_loc);
}