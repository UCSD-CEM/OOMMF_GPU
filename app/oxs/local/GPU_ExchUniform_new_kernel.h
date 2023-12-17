#ifndef _GPU_EXCHUNIFORM_NEW_KERNEL_H
#define _GPU_EXCHUNIFORM_NEW_KERNEL_H

#include "GPU_devstruct.h"

void s_exchUniform(const dim3 &grid_size, const dim3 &block_size,
  const dim3 &cubic_block_size,
  const FD_TYPE* dev_MValue, FD_TYPE* dev_H, FD_TYPE* dev_Energy, 
  FD_TYPE* dev_Torque, const uint3 dev_Dim, const FD_TYPE3 dev_wgt, 
  const FD_TYPE* dev_Msii, const int3 periodic, FD_TYPE* dev_field_loc, 
  FD_TYPE* dev_energy_loc, FD_TYPE* dev_dot, const bool outputH, 
  const bool outputE, const bool accumFlag);
  
string getExchKernelName();
#endif