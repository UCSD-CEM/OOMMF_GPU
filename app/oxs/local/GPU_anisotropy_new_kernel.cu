#include "nb.h"
#include "GPU_devstruct.h"

__global__  __launch_bounds__(BLK_SIZE, 16) 
void Rec_Integ_Kernel(
    const FD_TYPE* __restrict__ dev_Ms, const FD_TYPE* __restrict__ dev_inv_Ms,
    const FD_TYPE* __restrict__ dev_MValue, const int size,
    const FD_TYPE uniform_K1_value, const FD_TYPE uniform_Ha_value,
    const FD_TYPE3 uniform_axis_value,
    const FD_TYPE mult, const bool k1type, const uint3 flag_uniform,
    const FD_TYPE * __restrict__ K1, const FD_TYPE * __restrict__ Ha, 
    const FD_TYPE * __restrict__ axis,
    FD_TYPE *dev_H, FD_TYPE *dev_Torque, FD_TYPE *dev_Energy,
    FD_TYPE* dev_field_loc, FD_TYPE* dev_energy_loc,
    const bool outputH, const bool outputE, const bool accumFlag) {

  int i = threadIdx.x + 
    (blockIdx.x + blockIdx.y * gridDim.x)*blockDim.x;
  if (i >= size) {
    return;
  }

  FD_TYPE k = uniform_K1_value;
  FD_TYPE field_mult = uniform_Ha_value;

  FD_TYPE scaling  = mult;

  //computation
  if(k1type) {
    if(!flag_uniform.x) {
      k = K1[i];
    }
    field_mult = (2.0f  / MU0) * k * dev_inv_Ms[i];
  } else {
    if(!flag_uniform.y) {
      field_mult = Ha[i];
    }
    k = 0.5f * MU0 * field_mult * dev_Ms[i];
  }
  if(k == 0.f || field_mult == 0.f) {
    if(outputH) {
      dev_field_loc[i] = 0.f; 
      dev_field_loc[i + size] = 0.f; 
      dev_field_loc[i + 2 * size] = 0.f;
    }
    if(outputE) {
      dev_energy_loc[i] = 0.f;
    }
    return;
  }

  FD_TYPE3 axisi;
  if (flag_uniform.z)	{
    axisi.x = uniform_axis_value.x; 
    axisi.y = uniform_axis_value.y; 
    axisi.z = uniform_axis_value.z;
  } else {
    axisi.x = axis[i]; 
    axisi.y = axis[i + size]; 
    axisi.z = axis[i + 2 * size];
  }
  
  FD_TYPE3 m;
  m.x = dev_MValue[i];
  m.y = dev_MValue[i + size];
  m.z = dev_MValue[i + 2 * size];
  const FD_TYPE dot = m.x * axisi.x + m.y * axisi.y + m.z * axisi.z;
  FD_TYPE3 t;
  
  if (k <= 0) {
    FD_TYPE3 H;
    H.x = scaling * field_mult * dot * axisi.x;
    H.y = scaling * field_mult * dot * axisi.y;
    H.z = scaling * field_mult * dot * axisi.z;

    t.x = m.y*H.z - m.z*H.y; // mxH
    t.y = m.z*H.x - m.x*H.z;
    t.z = m.x*H.y - m.y*H.x;

    const FD_TYPE mkdotsq = -k * dot * dot;
    const FD_TYPE ei = scaling*mkdotsq;

    if (outputH) {
      dev_field_loc[i] = H.x; 
      dev_field_loc[i + size] = H.y; 
      dev_field_loc[i + 2 * size] = H.z; 
    }

    if (accumFlag) {
      dev_H[i] += H.x; dev_H[i + size] += H.y; dev_H[i + 2 * size] += H.z; 
    }
    
    if (outputE) {
      dev_energy_loc[i] = ei;
    }
    
    if (accumFlag) {
      dev_Energy[i] += ei;
  
      dev_Torque[i] += t.x;
      dev_Torque[i + size] += t.y;
      dev_Torque[i + 2 * size] += t.z; 
    }
  } else {
    FD_TYPE Hscale = scaling * field_mult * dot;

    t.x = m.y * axisi.z - m.z * axisi.y; // mxH
    t.y = m.z * axisi.x - m.x * axisi.z;
    t.z = m.x * axisi.y - m.y * axisi.x;

    FD_TYPE ktsq = k * (t.x * t.x + t.y * t.y + t.z * t.z);
    FD_TYPE ei = scaling * ktsq;

    if (outputH) {
      dev_field_loc[i] = Hscale * axisi.x;
      dev_field_loc[i + size] = Hscale * axisi.y;
      dev_field_loc[i + 2 * size] = Hscale * axisi.z;
    }

    if (accumFlag) {
      dev_H[i] += Hscale * axisi.x;
      dev_H[i + size] += Hscale * axisi.y;
      dev_H[i + 2 * size] += Hscale * axisi.z;
    }
    
    if (outputE) {
      dev_energy_loc[i] = ei;
    }
    
    if (accumFlag) {
      dev_Energy[i] += ei; 

      dev_Torque[i] += Hscale*t.x; 
      dev_Torque[i + size] += Hscale*t.y; 
      dev_Torque[i + 2 * size] += Hscale*t.z;
    }
  }
}

// wrapper function
void Rec_Integ(const dim3 &grid_size, const dim3 &block_size,
    const FD_TYPE* dev_Ms, const FD_TYPE* dev_inv_Ms,
    const FD_TYPE* dev_MValue, const int size,
    const FD_TYPE uniform_K1_value, const FD_TYPE uniform_Ha_value,
    const FD_TYPE3 uniform_axis_value,
    const FD_TYPE mult, const bool k1type, const uint3 flag_uniform,
    const FD_TYPE *K1, const FD_TYPE *Ha, const FD_TYPE *axis,
    FD_TYPE *dev_H, FD_TYPE *dev_Torque, FD_TYPE *dev_Energy,
    FD_TYPE* dev_field_loc, FD_TYPE* dev_energy_loc,
    const bool outputH, const bool outputE, const bool accumFlag) {
    
  Rec_Integ_Kernel<<<grid_size, block_size>>>(
    dev_Ms, dev_inv_Ms, dev_MValue, size,
    uniform_K1_value, uniform_Ha_value,
    uniform_axis_value, mult, k1type, 
    flag_uniform, K1, Ha, axis, dev_H,
    dev_Torque, dev_Energy, dev_field_loc, dev_energy_loc, 
    outputH, outputE, accumFlag);
}