#ifndef _GPU_DEVSTRUCT_H
#define _GPU_DEVSTRUCT_H

#include "oc.h"

class Oxs_Mesh;

#define ODTV_VECSIZE 3
#define ODTV_COMPLEXSIZE 2
#define BLK_SIZE 128
#define CHOOSESINGLE
#define DEV_NUM 0

#ifdef CHOOSESINGLE
#ifndef FD_TYPE
#define FD_TYPE float
#endif
#ifndef FD_TYPE3
#define FD_TYPE3 float3
#endif
#ifndef FD_CPLX_TYPE
#define FD_CPLX_TYPE cuComplex
#endif
#ifndef FWDFFT_METHOD
#define FWDFFT_METHOD CUFFT_R2C
#endif
#ifndef BWDFFT_METHOD
#define BWDFFT_METHOD CUFFT_C2R
#endif
#ifndef FWDFFT_EXE
#define FWDFFT_EXE cufftExecR2C
#endif
#ifndef BWDFFT_EXE
#define BWDFFT_EXE cufftExecC2R
#endif
#endif
#ifdef CHOOSEDOUBLE
#ifndef FD_TYPE
#define FD_TYPE double
#endif
#ifndef FD_TYPE3
#define FD_TYPE3 double3
#endif
#ifndef FD_CPLX_TYPE
#define FD_CPLX_TYPE cuDoubleComplex
#endif
#ifndef FWDFFT_METHOD
#define FWDFFT_METHOD CUFFT_D2Z
#endif
#ifndef BWDFFT_METHOD
#define BWDFFT_METHOD CUFFT_Z2D
#endif
#ifndef FWDFFT_EXE
#define FWDFFT_EXE cufftExecD2Z
#endif
#ifndef BWDFFT_EXE
#define BWDFFT_EXE cufftExecZ2D
#endif
#endif

struct DEVSTRUCT {
  
  DEVSTRUCT(): dev_Ms(0), dev_Msi(0), dev_MValue(0), 
    dev_energy(0), dev_field(0), dev_torque(0),
    dev_dm_dt(0), dev_energy_bak(0), dev_dm_dt_bak(0),
    dev_local_sum(0), dev_local_field(0), 
    dev_local_energy(0), dev_vol(0), dev_dot(0),
    dev_FFT_workArea(0) {};
  
  void allocMem(const Oxs_Mesh* genmesh);
  
  void releaseMem();
  
  void purgeAccumMem(const OC_INDEX &size);

  void purgeMem(const Oxs_Mesh* genmesh);
  
  void reset();

  FD_TYPE* dev_Ms;
  FD_TYPE* dev_Msi;
  FD_TYPE* dev_MValue;
  FD_TYPE* dev_energy;
  FD_TYPE* dev_field;
  FD_TYPE* dev_torque;
  FD_TYPE* dev_dm_dt;
  FD_TYPE* dev_energy_bak;
  FD_TYPE* dev_dm_dt_bak;
  FD_TYPE* dev_local_sum;
  FD_TYPE* dev_local_field;
  FD_TYPE* dev_local_energy;
  FD_TYPE* dev_vol;
  FD_TYPE* dev_dot;
  void* dev_FFT_workArea;
};

#endif