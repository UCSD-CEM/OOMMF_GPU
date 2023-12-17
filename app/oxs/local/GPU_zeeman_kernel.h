#ifndef _GPU_ZEEMAN_KERNEL	
#define _GPU_ZEEMAN_KERNEL
#include "GPU_devstruct.h"

void Get_Fixed_Zeeman(const dim3 &grid_size, const dim3 &block_size,
  FD_TYPE* dev_MValue, FD_TYPE* dev_Ms, FD_TYPE *dev_ZField,
  FD_TYPE* dev_Field, FD_TYPE *dev_Energy, FD_TYPE* dev_Torque, int size, 
  const bool outputE, const bool accumFlag, FD_TYPE* dev_energy_loc);
    
#endif