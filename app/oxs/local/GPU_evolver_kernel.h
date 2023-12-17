#ifndef _GPU_EULEREVOLVE_KERNEL_H
#define _GPU_EULEREVOLVE_KERNEL_H

#include "GPU_devstruct.h"

void backUpEnergy(FD_TYPE *bak_ptr, const FD_TYPE *src_ptr, 
    const OC_INDEX &size);
    
void dm_dt(const dim3 &grid_size, const dim3 &block_size, const OC_INDEX size, 
  const FD_TYPE coef1, const FD_TYPE coef2, const OC_BOOL do_precess, 
  DEVSTRUCT &host_struct, const FD_TYPE *coef1Array = NULL, 
  const FD_TYPE *coef2Array = NULL);
  
void nextSpin(const dim3 &grid_size, const dim3 &block_size, const OC_INDEX &size, 
  const FD_TYPE &stepsize, const DEVSTRUCT &host_struct);

void dmDtError(const dim3 &grid_size, const dim3 &block_size,
    const OC_INDEX &size, const OC_INDEX &reduce_size, const OC_INDEX &BLKSIZE,
    const DEVSTRUCT &host_struct, FD_TYPE *dev_max_error);
    
void collectDmDtStatistics(const dim3 &grid_size, const dim3 &block_size,
  const OC_INDEX &size, const OC_INDEX &reduce_size, const OC_INDEX &BLKSIZE,
  const DEVSTRUCT &host_struct, FD_TYPE *dev_dE_dt_sum, 
  FD_TYPE *dev_max_dm_dt_sq, const FD_TYPE *coef1Array = NULL, 
  const FD_TYPE *coef2Array = NULL);
  
void collectEnergyStatistics(const dim3 &grid_size, const dim3 &block_size,
  const OC_INDEX &size, const OC_INDEX &reduce_size, const OC_INDEX &BLKSIZE,
  const DEVSTRUCT &host_struct, FD_TYPE *dev_dE, FD_TYPE *dev_var_dE, 
  FD_TYPE *dev_total_E);
  
void adjustSpin(const dim3 &grid_size, const dim3 &block_size, const OC_INDEX &size,
  const OC_INDEX &reduce_size, const OC_INDEX &BLKSIZE,
  const FD_TYPE &stepsize, const DEVSTRUCT &host_struct,
  const FD_TYPE *dev_MValue_old, FD_TYPE *dev_MValue_new,
  FD_TYPE *dev_min_magsq, FD_TYPE *dev_max_magsq);
  
void accumulate(const dim3 &grid_size, const dim3 &block_size, const OC_INDEX &size,
    const FD_TYPE &mult1, const FD_TYPE &mult2, const FD_TYPE *d_increment, const FD_TYPE *d_idata, 
    FD_TYPE *d_odata);
  
void dmDtErrorStep2(const dim3 &grid_size, const dim3 &block_size,
  const OC_INDEX &size, const OC_INDEX &reduce_size, const OC_INDEX &BLKSIZE,
  const FD_TYPE *dev_current_data, const FD_TYPE *dev_vtmpA, 
  const FD_TYPE * dev_vtmpB, FD_TYPE *dev_local_sum, FD_TYPE *dev_max_error_sq);
  
void makeUnitAndCollectMinMax(const dim3 &grid_size, const dim3 &block_size, 
  const OC_INDEX &size,const OC_INDEX &reduce_size, const OC_INDEX &BLKSIZE,
  FD_TYPE *dev_data, FD_TYPE *dev_local_sum, FD_TYPE *dev_min_magsq, 
  FD_TYPE *dev_max_magsq);
  
void maxDiff(const dim3 &grid_size, const dim3 &block_size,
  const OC_INDEX &size,const OC_INDEX &reduce_size, const OC_INDEX &BLKSIZE,
  const FD_TYPE *dev_idata1, const FD_TYPE *dev_idata2, FD_TYPE *dev_local_sum,
  FD_TYPE *dev_max_error_sq);
  
void dmDtErrorStep4(const dim3 &grid_size, const dim3 &block_size,
    const OC_INDEX &size, const OC_INDEX &reduce_size, const OC_INDEX &BLKSIZE,
    const FD_TYPE *dev_vtmpB, const FD_TYPE * dev_vtmpC, 
    FD_TYPE *dev_local_sum, FD_TYPE *dev_max_error_sq);
#endif