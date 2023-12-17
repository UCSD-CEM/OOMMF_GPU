#ifndef _GPU_DEMAG_KERNEL_H	
#define _GPU_DEMAG_KERNEL_H

#include "GPU_devstruct.h"

void pad(const dim3 &grid_size, const dim3 &block_size,
  FD_TYPE* MValue, FD_TYPE* dev_Mtemp, FD_TYPE* dev_Ms, int rdimx, int rdimxy,
  int rsize, int cdimx, int cdimxy, int csize);
  
void multiplication(const dim3 &grid_size, const dim3 &block_size,
    FD_CPLX_TYPE* dev_Mtemp, const FD_TYPE* GreenFunc, 
    const int green_x, const int green_y, const int green_z, const int cdimx, 
    const int cdimy, const int cdimz);
    
void unpad(const dim3 &grid_size, const dim3 &block_size,
    FD_TYPE* dev_Hsta, FD_TYPE* dev_Hsta_r, FD_TYPE* dev_MValue, 
    FD_TYPE* dev_Ms, FD_TYPE* dev_Energy, FD_TYPE* dev_Torque, 
    int rdimx, int rdimxy, int rsize, int cdimx, int cdimxy, int csize, 
    const bool outputH, const bool outputE, const bool accumFlag,
    FD_TYPE* dev_energy_loc, FD_TYPE* dev_field_loc);
    
string getPadKernelName();

string getMultiplicationKernelName();

string getUnpadKernelName();
#endif