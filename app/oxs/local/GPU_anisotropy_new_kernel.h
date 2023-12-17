#ifndef _GPU_ANISOTROPY_NEW_KERNEL_H
#define _GPU_ANISOTROPY_NEW_KERNEL_H

#include "GPU_devstruct.h"

void Rec_Integ(const dim3 &grid_size, const dim3 &block_size,
  const FD_TYPE* dev_Ms, const FD_TYPE* dev_inv_Ms,
  const FD_TYPE* dev_MValue, const int size,
  const FD_TYPE uniform_K1_value, const FD_TYPE uniform_Ha_value,
  const FD_TYPE3 uniform_axis_value,
  const FD_TYPE mult, const bool k1type, const uint3 flag_uniform,
  const FD_TYPE *K1, const FD_TYPE *Ha, const FD_TYPE *axis,
  FD_TYPE *dev_H, FD_TYPE *dev_Torque, FD_TYPE *dev_Energy,
  FD_TYPE* dev_field_loc, FD_TYPE* dev_energy_loc,
  const bool outputH, const bool outputE, const bool accumFlag);
  
#endif