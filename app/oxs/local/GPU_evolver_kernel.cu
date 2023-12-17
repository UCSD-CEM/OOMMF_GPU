#include <stdexcept>
#include "GPU_devstruct.h"

template <unsigned int blockSize>
__device__ void warpReduce(volatile FD_TYPE *sdata, unsigned int tid) {
	if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
	if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
	if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
	if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
	if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
	if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int blockSize>
__device__ void warpMax(volatile FD_TYPE *sdata, unsigned int tid) {
	if (blockSize >= 64 && sdata[tid] < sdata[tid + 32]) sdata[tid] = sdata[tid + 32];
	if (blockSize >= 32 && sdata[tid] < sdata[tid + 16]) sdata[tid] = sdata[tid + 16];
	if (blockSize >= 16 && sdata[tid] < sdata[tid + 8]) sdata[tid] = sdata[tid + 8];
	if (blockSize >= 8 && sdata[tid] < sdata[tid + 4]) sdata[tid] = sdata[tid + 4];
	if (blockSize >= 4 && sdata[tid] < sdata[tid + 2]) sdata[tid] = sdata[tid + 2];
	if (blockSize >= 2 && sdata[tid] < sdata[tid + 1]) sdata[tid] = sdata[tid + 1];
}

template <unsigned int blockSize>
__global__ void dm_dt_err_kernel(unsigned int size, DEVSTRUCT dev_struct) {
	
	__shared__ FD_TYPE sdata[blockSize];
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x+blockIdx.y*gridDim.x;
	unsigned int gid = bid*blockSize + tid;
	unsigned int gridSize = blockSize*gridDim.x*gridDim.y*gridDim.z;
	
	//initialize the shared memory data
	sdata[tid] = 0.f;
	FD_TYPE3 tmp;
	FD_TYPE tmp_magn;
	while (gid < size) {
		tmp.x = (dev_struct.dev_dm_dt)[gid] - (dev_struct.dev_dm_dt_bak)[gid];
		tmp.y = (dev_struct.dev_dm_dt)[gid + size] - (dev_struct.dev_dm_dt_bak)[gid + size];
		tmp.z = (dev_struct.dev_dm_dt)[gid + 2 * size] - (dev_struct.dev_dm_dt_bak)[gid + 2 * size];
		tmp_magn = tmp.x*tmp.x + tmp.y*tmp.y + tmp.z*tmp.z;
		if( sdata[tid] < tmp_magn)	sdata[tid] = tmp_magn;
		gid += gridSize; 
	}
	__syncthreads();
	
	//reduction in shared memory
	if (blockSize >= 1024) { if (tid < 512 && sdata[tid] < sdata[tid + 512]) { sdata[tid] = sdata[tid + 512]; } __syncthreads(); }
	if (blockSize >= 512) { if (tid < 256 && sdata[tid] < sdata[tid + 256]) { sdata[tid] = sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128 && sdata[tid] < sdata[tid + 128]) { sdata[tid] = sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64 && sdata[tid] < sdata[tid + 64]) { sdata[tid] = sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32) warpMax<blockSize>(sdata, tid);
	if (tid == 0)  (dev_struct.dev_local_sum)[bid] = sdata[0];
}

inline __device__ FD_TYPE vec_magn(FD_TYPE* idata, unsigned int offset, unsigned int size){
	return idata[offset]*idata[offset] + idata[offset+size]*idata[offset+size] 
		+ idata[offset+2*size]*idata[offset+2*size];
}

template <unsigned int blockSize>
__global__ void collect_dm_dt_kernel(unsigned int size, DEVSTRUCT dev_struct) {
	
	__shared__ FD_TYPE s_sum[blockSize];
	__shared__ FD_TYPE s_max[blockSize];
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x+blockIdx.y*gridDim.x;
	unsigned int gid = bid*blockSize + tid;
	unsigned int gridSize = blockSize*gridDim.x*gridDim.y*gridDim.z;
	
	//initialize the shared memory data
	s_sum[tid] = 0.f;	s_max[tid] = 0.f;
	FD_TYPE tmp_magn;
	while (gid < size) {
		tmp_magn = vec_magn(dev_struct.dev_dm_dt, gid, size);
		if( tmp_magn > 0 ){
			if( s_max[tid] < tmp_magn)	s_max[tid] = tmp_magn;
			s_sum[tid] += ((dev_struct.dev_Ms)[gid] * (dev_struct.dev_vol)[gid] *
				 vec_magn(dev_struct.dev_torque, gid, size));
		}
		gid += gridSize; 
	}
	__syncthreads();
	
	//reduction in shared memory
	if (blockSize >= 1024) { 
		if (tid < 512){
			if(s_max[tid] < s_max[tid + 512]) s_max[tid] = s_max[tid + 512]; 
			s_sum[tid] += s_sum[tid+512];
		} 
		__syncthreads();
	}
	if (blockSize >= 512) { 
		if (tid < 256){
			if(s_max[tid] < s_max[tid + 256]) s_max[tid] = s_max[tid + 256]; 
			s_sum[tid] += s_sum[tid+256];
		} 
		__syncthreads();
	}
	if (blockSize >= 256) { 
		if (tid < 128){
			if(s_max[tid] < s_max[tid + 128]) s_max[tid] = s_max[tid + 128]; 
			s_sum[tid] += s_sum[tid+128];
		} 
		__syncthreads();
	}
	if (blockSize >= 128) { 
		if (tid < 64){
			if(s_max[tid] < s_max[tid + 64]) s_max[tid] = s_max[tid + 64]; 
			s_sum[tid] += s_sum[tid+64];
		} 
		__syncthreads();
	}
	if (tid < 32){
		warpMax<blockSize>(s_max, tid);
		warpReduce<blockSize>(s_sum, tid);
	}
	if (tid == 0){
		(dev_struct.dev_local_sum)[bid] = s_sum[0];
		(dev_struct.dev_local_sum)[bid+gridDim.x*gridDim.y*gridDim.z] = s_max[0];
	}
}

template <unsigned int blockSize>
__global__ void collect_dm_dt_kernel_freeCoef(const unsigned int size, 
    const DEVSTRUCT dev_struct, const FD_TYPE *coef1, const FD_TYPE *coef2) {
	
	__shared__ FD_TYPE s_sum[blockSize];
	__shared__ FD_TYPE s_max[blockSize];
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x+blockIdx.y*gridDim.x;
	unsigned int gid = bid*blockSize + tid;
	const unsigned int gridSize = blockSize*gridDim.x*gridDim.y*gridDim.z;
	
	//initialize the shared memory data
	s_sum[tid] = 0.f;	s_max[tid] = 0.f;
	FD_TYPE tmp_magn;
	while (gid < size) {
		tmp_magn = vec_magn(dev_struct.dev_torque, gid, size) * 
      (1 + coef2[gid] * coef2[gid]) * coef1[gid] * coef1[gid];
		if( tmp_magn > 0 ){
			if( s_max[tid] < tmp_magn)	s_max[tid] = tmp_magn;
			s_sum[tid] += ((dev_struct.dev_Ms)[gid] * (dev_struct.dev_vol)[gid] *
				 vec_magn(dev_struct.dev_torque, gid, size)) * coef1[gid] * coef2[gid];
		}
		gid += gridSize; 
	}
	__syncthreads();
	
	//reduction in shared memory
	if (blockSize >= 1024) { 
		if (tid < 512){
			if(s_max[tid] < s_max[tid + 512]) s_max[tid] = s_max[tid + 512]; 
			s_sum[tid] += s_sum[tid+512];
		} 
		__syncthreads();
	}
	if (blockSize >= 512) { 
		if (tid < 256){
			if(s_max[tid] < s_max[tid + 256]) s_max[tid] = s_max[tid + 256]; 
			s_sum[tid] += s_sum[tid+256];
		} 
		__syncthreads();
	}
	if (blockSize >= 256) { 
		if (tid < 128){
			if(s_max[tid] < s_max[tid + 128]) s_max[tid] = s_max[tid + 128]; 
			s_sum[tid] += s_sum[tid+128];
		} 
		__syncthreads();
	}
	if (blockSize >= 128) { 
		if (tid < 64){
			if(s_max[tid] < s_max[tid + 64]) s_max[tid] = s_max[tid + 64]; 
			s_sum[tid] += s_sum[tid+64];
		} 
		__syncthreads();
	}
	if (tid < 32){
		warpMax<blockSize>(s_max, tid);
		warpReduce<blockSize>(s_sum, tid);
	}
	if (tid == 0){
		(dev_struct.dev_local_sum)[bid] = s_sum[0];
		(dev_struct.dev_local_sum)[bid+gridDim.x*gridDim.y*gridDim.z] = s_max[0];
	}
}

template <unsigned int blockSize>
__global__ void energy_err_kernel(unsigned int size, DEVSTRUCT dev_struct) {
	
	__shared__ FD_TYPE s_sum_E[blockSize];
	__shared__ FD_TYPE s_sum_dE[blockSize];
	__shared__ FD_TYPE s_sum_varE[blockSize];
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x+blockIdx.y*gridDim.x;
	unsigned int gid = bid*blockSize + tid;
	unsigned int gridSize = blockSize*gridDim.x*gridDim.y*gridDim.z;
	
	//initialize the shared memory data
	s_sum_E[tid] = 0.f;	s_sum_dE[tid] = 0.f;	s_sum_varE[tid] = 0.f;
	FD_TYPE tmp_energy, tmp_energy_old, tmp_vol;
	while (gid < size) {
		tmp_energy = (dev_struct.dev_energy)[gid];
		tmp_energy_old = (dev_struct.dev_energy_bak)[gid];
		tmp_vol = (dev_struct.dev_vol)[gid];
		s_sum_E[tid] += tmp_energy * tmp_vol;
		s_sum_dE[tid] += (tmp_energy-tmp_energy_old) * tmp_vol;
		s_sum_varE[tid] += (tmp_energy*tmp_energy+tmp_energy_old*tmp_energy_old)
											*tmp_vol*tmp_vol;
		gid += gridSize; 
	}
	__syncthreads();
	
	//reduction in shared memory
	if (blockSize >= 1024) { 
		if (tid < 512){
			s_sum_E[tid] += s_sum_E[tid+512];s_sum_dE[tid] += s_sum_dE[tid+512];
			s_sum_varE[tid] += s_sum_varE[tid+512];
		} 
		__syncthreads();
	}
	if (blockSize >= 512) { 
		if (tid < 256){
			s_sum_E[tid] += s_sum_E[tid+256];s_sum_dE[tid] += s_sum_dE[tid+256];
			s_sum_varE[tid] += s_sum_varE[tid+256];
		} 
		__syncthreads();
	}
	if (blockSize >= 256) { 
		if (tid < 128){
			s_sum_E[tid] += s_sum_E[tid+128];s_sum_dE[tid] += s_sum_dE[tid+128];
			s_sum_varE[tid] += s_sum_varE[tid+128];
		} 
		__syncthreads();
	}
	if (blockSize >= 128) { 
		if (tid < 64){
			s_sum_E[tid] += s_sum_E[tid+64];s_sum_dE[tid] += s_sum_dE[tid+64];
			s_sum_varE[tid] += s_sum_varE[tid+64];
		} 
		__syncthreads();
	}
	if (tid < 32){
		warpReduce<blockSize>(s_sum_E, tid);
		warpReduce<blockSize>(s_sum_dE, tid);
		warpReduce<blockSize>(s_sum_varE, tid);
	}
	if (tid == 0){
		(dev_struct.dev_local_sum)[bid] = s_sum_E[0];
		(dev_struct.dev_local_sum)[bid+gridDim.x*gridDim.y*gridDim.z] = s_sum_dE[0];
		(dev_struct.dev_local_sum)[bid+2*gridDim.x*gridDim.y*gridDim.z] = s_sum_varE[0]*256;
	}
}

template <unsigned int blockSize>
__global__ void final_max_kernel(unsigned int size, FD_TYPE *g_idata, FD_TYPE *g_odata) {
	
	__shared__ FD_TYPE sdata[blockSize];
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x+blockIdx.y*gridDim.x;
	unsigned int gid = bid*blockSize + tid;
	unsigned int gridSize = blockSize*gridDim.x*gridDim.y*gridDim.z;
	
	//initialize the shared memory data
	sdata[tid] = 0.f;
	while (gid < size) {
		if( sdata[tid] < g_idata[gid])	sdata[tid] = g_idata[gid];
		gid += gridSize; 
	}
	__syncthreads();
	
	//reduction in shared memory
	if (blockSize >= 1024) { if (tid < 512 && sdata[tid] < sdata[tid + 512]) { sdata[tid] = sdata[tid + 512]; } __syncthreads(); }
	if (blockSize >= 512) { if (tid < 256 && sdata[tid] < sdata[tid + 256]) { sdata[tid] = sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128 && sdata[tid] < sdata[tid + 128]) { sdata[tid] = sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64 && sdata[tid] < sdata[tid + 64]) { sdata[tid] = sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32) warpMax<blockSize>(sdata, tid);
	if (tid == 0)  g_odata[0] = sdata[0];
}

template <unsigned int blockSize>
__global__ void final_reduce_kernel(unsigned int size, FD_TYPE *g_idata, FD_TYPE *g_odata) {
	
	__shared__ FD_TYPE sdata[blockSize];
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x+blockIdx.y*gridDim.x;
	unsigned int gid = bid*blockSize + tid;
	unsigned int gridSize = blockSize*gridDim.x*gridDim.y*gridDim.z;
	
	//initialize the shared memory data
	sdata[tid] = 0.f;
	while (gid < size) {sdata[tid] += g_idata[gid]; gid += gridSize; }
	__syncthreads();
	
	//reduction in shared memory
	if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32) warpReduce<blockSize>(sdata, tid);
	if (tid == 0) g_odata[0] = sdata[0];
}

inline __device__ void operator^=(FD_TYPE3 &a, FD_TYPE3 b) {
    FD_TYPE3 tmp_a;
	tmp_a.x = a.y*b.z - a.z*b.y;
	tmp_a.y = a.z*b.x - a.x*b.z;
	tmp_a.z = a.x*b.y - a.y*b.x;
	a.x = tmp_a.x; a.y = tmp_a.y; a.z = tmp_a.z;
}

__device__ FD_TYPE MakeUnit_kernel(FD_TYPE *m) {
  FD_TYPE length = m[0] * m[0] + m[1] * m[1] + m[2] * m[2];
  if (length - 0.f < 1e-4) return 0.f;

  m[0] = m[0] / (sqrtf(length));
  m[1] = m[1] / (sqrtf(length));
  m[2] = m[2] / (sqrtf(length));
  
  return length;
}

__global__ void Backup_kernel(int size, DEVSTRUCT dev_struct) {
  int tid = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x;
  if (tid > size - 1) return;
	
  (dev_struct.dev_energy_bak)[tid] = (dev_struct.dev_energy)[tid];
}

__global__ void Calculate_dm_dt_kernel(const OC_INDEX size, FD_TYPE coef1, 
    FD_TYPE coef2, OC_BOOL do_precess, DEVSTRUCT dev_struct) {
      
  const int tid = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x;
  if (tid > size - 1) {
    return;
  }
  
  if ((dev_struct.dev_Ms)[tid] == 0.f) {
    (dev_struct.dev_dm_dt)[tid] = 0.f;
    (dev_struct.dev_dm_dt)[tid + size] = 0.f;
    (dev_struct.dev_dm_dt)[tid + 2 * size] = 0.f;
  } else {
    FD_TYPE3 scratch;
    scratch.x = coef1 * (dev_struct.dev_torque)[tid];
    scratch.y = coef1 * (dev_struct.dev_torque)[tid + size];
    scratch.z = coef1 * (dev_struct.dev_torque)[tid + 2 * size];
    if (do_precess) {
      (dev_struct.dev_dm_dt)[tid] = scratch.x;
      (dev_struct.dev_dm_dt)[tid + size] = scratch.y;
      (dev_struct.dev_dm_dt)[tid + 2 * size] = scratch.z;
    } else {
      (dev_struct.dev_dm_dt)[tid] = 0.f;
      (dev_struct.dev_dm_dt)[tid + size] = 0.f;
      (dev_struct.dev_dm_dt)[tid + 2 * size] = 0.f;
    }
    //calc mxmxH
    FD_TYPE3 m0;
    m0.x = coef2 * (dev_struct.dev_MValue)[tid];
    m0.y = coef2 * (dev_struct.dev_MValue)[tid+ size];
    m0.z = coef2 * (dev_struct.dev_MValue)[tid+ 2*size];
    scratch ^= m0;						

    (dev_struct.dev_dm_dt)[tid] += scratch.x;
    (dev_struct.dev_dm_dt)[tid + size] += scratch.y;
    (dev_struct.dev_dm_dt)[tid + 2 * size] += scratch.z;
  }
  
  return;
}

// The subroutine is taken from Calculate_dm_dt_kernel(), the only change is
// the definition of coef1 and coef2
__global__ void Calculate_dm_dt_kernel_freeCoef(const OC_INDEX size, 
    const FD_TYPE *coef1, const FD_TYPE *coef2, const OC_BOOL do_precess, 
    const DEVSTRUCT dev_struct) {
      
  const int tid = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x;
  if (tid > size - 1) {
    return;
  }
  
  if ((dev_struct.dev_Ms)[tid] == 0.f) {
    (dev_struct.dev_dm_dt)[tid] = 0.f;
    (dev_struct.dev_dm_dt)[tid + size] = 0.f;
    (dev_struct.dev_dm_dt)[tid + 2 * size] = 0.f;
  } else {
    FD_TYPE3 scratch;
    scratch.x = coef1[tid] * (dev_struct.dev_torque)[tid];
    scratch.y = coef1[tid] * (dev_struct.dev_torque)[tid + size];
    scratch.z = coef1[tid] * (dev_struct.dev_torque)[tid + 2 * size];
    if (do_precess) {
      (dev_struct.dev_dm_dt)[tid] = scratch.x;
      (dev_struct.dev_dm_dt)[tid + size] = scratch.y;
      (dev_struct.dev_dm_dt)[tid + 2 * size] = scratch.z;
    } else {
      (dev_struct.dev_dm_dt)[tid] = 0.f;
      (dev_struct.dev_dm_dt)[tid + size] = 0.f;
      (dev_struct.dev_dm_dt)[tid + 2 * size] = 0.f;
    }
    //calc mxmxH
    FD_TYPE3 m0;
    m0.x = coef2[tid] * (dev_struct.dev_MValue)[tid];
    m0.y = coef2[tid] * (dev_struct.dev_MValue)[tid+ size];
    m0.z = coef2[tid] * (dev_struct.dev_MValue)[tid+ 2*size];
    scratch ^= m0;						

    (dev_struct.dev_dm_dt)[tid] += scratch.x;
    (dev_struct.dev_dm_dt)[tid + size] += scratch.y;
    (dev_struct.dev_dm_dt)[tid + 2 * size] += scratch.z;
  }
  
  return;
}

__global__ void CalcNextSpin_kernel(int size, FD_TYPE stepsize, DEVSTRUCT dev_struct)
{
  int tid = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x;
  if (tid > size - 1)
	return;
	
  FD_TYPE tempspin[3];
  tempspin[0] = stepsize * (dev_struct.dev_dm_dt)[tid];
  tempspin[1] = stepsize * (dev_struct.dev_dm_dt)[tid + size];
  tempspin[2] = stepsize * (dev_struct.dev_dm_dt)[tid + 2 * size];
  
  FD_TYPE adj = 0.5 * (tempspin[0] * tempspin[0] +
				  tempspin[1] * tempspin[1] +
				  tempspin[2] * tempspin[2]
				 );
  tempspin[0] = tempspin[0] - adj * (dev_struct.dev_MValue)[tid];
  tempspin[1] = tempspin[1] - adj * (dev_struct.dev_MValue)[tid + size];
  tempspin[2] = tempspin[2] - adj * (dev_struct.dev_MValue)[tid + 2 * size];
  tempspin[0] *= 1.0/(1.0 + adj);
  tempspin[1] *= 1.0/(1.0 + adj);
  tempspin[2] *= 1.0/(1.0 + adj);
  tempspin[0] += (dev_struct.dev_MValue)[tid];
  tempspin[1] += (dev_struct.dev_MValue)[tid + size];
  tempspin[2] += (dev_struct.dev_MValue)[tid + 2 * size];
  MakeUnit_kernel(tempspin);
  (dev_struct.dev_MValue)[tid] = tempspin[0];
  (dev_struct.dev_MValue)[tid + size] = tempspin[1];
  (dev_struct.dev_MValue)[tid + 2 * size] = tempspin[2];
}

template <unsigned int blockSize>
__device__ void warpMin(volatile FD_TYPE *sdata, unsigned int tid) {
	if (blockSize >= 64 && sdata[tid] > sdata[tid + 32]) sdata[tid] = sdata[tid + 32];
	if (blockSize >= 32 && sdata[tid] > sdata[tid + 16]) sdata[tid] = sdata[tid + 16];
	if (blockSize >= 16 && sdata[tid] > sdata[tid + 8]) sdata[tid] = sdata[tid + 8];
	if (blockSize >= 8 && sdata[tid] > sdata[tid + 4]) sdata[tid] = sdata[tid + 4];
	if (blockSize >= 4 && sdata[tid] > sdata[tid + 2]) sdata[tid] = sdata[tid + 2];
	if (blockSize >= 2 && sdata[tid] > sdata[tid + 1]) sdata[tid] = sdata[tid + 1];
}

__global__ void Accumulate_kernel(const unsigned int size, const FD_TYPE mult1,
    const FD_TYPE mult2, const FD_TYPE *d_increment, const FD_TYPE *d_idata, FD_TYPE *d_odata) {
  
  const unsigned int tid = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x) 
    * blockDim.x;
  if (tid >= size) {
    return;
  }
	
  d_odata[tid] = mult1 * d_idata[tid] + mult2 * d_increment[tid];
  d_odata[tid + size] = mult1 * d_idata[tid + size] + mult2 * d_increment[tid + size];
  d_odata[tid + 2 * size] = mult1 * d_idata[tid + 2 * size] + 
    mult2 * d_increment[tid + 2 * size];
}

// compute the next spin and collect max and min of magsq into 
// dev_struct.dev_local_sum
template <unsigned int blockSize>
__global__ void CalcNewSpinAndCollectKernel(const unsigned int size, 
  const unsigned int reduce_size, const FD_TYPE mstep, 
  const DEVSTRUCT dev_struct, const FD_TYPE *dev_MValue_old,
  FD_TYPE *dev_MValue_new) {
  
  __shared__ FD_TYPE s_min[blockSize];
	__shared__ FD_TYPE s_max[blockSize];
  const unsigned int tid = threadIdx.x;
  const unsigned int bid = blockIdx.x + blockIdx.y * gridDim.x;
  unsigned int gid = bid * blockDim.x + tid;
  const unsigned int gridSize = blockDim.x * gridDim.x * gridDim.y * gridDim.z;
  s_min[tid] = 0.0;	s_max[tid] = -FLT_MIN;
  if (gid >= size) {
    return;
  }
  
  FD_TYPE tempspin[3];
  tempspin[0] = dev_MValue_old[gid] + 
    mstep * dev_struct.dev_dm_dt[gid];
  tempspin[1] = dev_MValue_old[gid + size] + 
    mstep * dev_struct.dev_dm_dt[gid + size];
  tempspin[2] = dev_MValue_old[gid + 2 * size] + 
    mstep * dev_struct.dev_dm_dt[gid + 2 * size];
  
  const FD_TYPE magsq = MakeUnit_kernel(tempspin);
  
  dev_MValue_new[gid] = tempspin[0];
  dev_MValue_new[gid + size] = tempspin[1];
  dev_MValue_new[gid + 2 * size] = tempspin[2];
#ifdef GPU_DEBUG  
  if(gid < 10) {
    printf("dev_MValue[%d] = (%g, %g, %g)\n", gid, dev_MValue_new[gid], 
      dev_MValue_new[gid + size], dev_MValue_new[gid + 2 * size]);
  }
#endif  
  while (gid < size) {
    s_max[tid] = max(s_max[tid], magsq);
    s_min[tid] = min(s_max[tid], magsq);
		gid += gridSize; 
	}
  __syncthreads();
	
	//reduction in shared memory
  for (int stride = 512; stride > 32; stride /= 2) {
    if (blockSize > stride && tid < stride) {
      s_max[tid] = max(s_max[tid], s_max[tid + stride]);
      s_min[tid] = min(s_min[tid], s_min[tid + stride]);
    }
    __syncthreads();
  }
  
	if (tid < 32){
		warpMax<blockSize>(s_max, tid);
		warpMin<blockSize>(s_min, tid);
	}
  
	if (tid == 0) {
		(dev_struct.dev_local_sum)[bid] = s_min[0];
		(dev_struct.dev_local_sum)[bid + reduce_size] = s_max[0];
	}
}

// make unit of dev_data and collect max and min of magsq into dev_tmp
template <unsigned int blockSize>
__global__ void MakeUnitAndCollectKernel(const unsigned int size, 
    const unsigned int reduce_size, FD_TYPE *dev_data, FD_TYPE *dev_local_sum) {
  
  __shared__ FD_TYPE s_min[blockSize];
	__shared__ FD_TYPE s_max[blockSize];
  const unsigned int tid = threadIdx.x;
  const unsigned int bid = blockIdx.x + blockIdx.y * gridDim.x;
  unsigned int gid = bid * blockDim.x + tid;
  const unsigned int gridSize = blockDim.x * gridDim.x * gridDim.y * gridDim.z;
  s_min[tid] = FLT_MAX;	s_max[tid] = 0.f;
  if (gid >= size) {
    return;
  }
  
  FD_TYPE tempspin[3];
  tempspin[0] = dev_data[gid];
  tempspin[1] = dev_data[gid + size];
  tempspin[2] = dev_data[gid + 2 * size];
  
  const FD_TYPE magsq = MakeUnit_kernel(tempspin);
  
  dev_data[gid] = tempspin[0];
  dev_data[gid + size] = tempspin[1];
  dev_data[gid + 2 * size] = tempspin[2];
  
  while (gid < size) {
    s_max[tid] = max(s_max[tid], magsq);
    s_min[tid] = min(s_min[tid], magsq);
		gid += gridSize; 
	}
  __syncthreads();
	
	//reduction in shared memory
  for (int stride = 512; stride > 32; stride /= 2) {
    if (blockSize > stride && tid < stride) {
      s_max[tid] = max(s_max[tid], s_max[tid + stride]);
      s_min[tid] = min(s_min[tid], s_min[tid + stride]);
    }
    __syncthreads();
  }

	if (tid < 32){
		warpMax<blockSize>(s_max, tid);
		warpMin<blockSize>(s_min, tid);
	}

	if (tid == 0) {
		dev_local_sum[bid] = s_min[0];
		dev_local_sum[bid + reduce_size] = s_max[0];
	}
}

template <unsigned int blockSize>
__global__ void final_min_max_kernel(const unsigned int size, 
  const FD_TYPE *g_idata_min, const FD_TYPE *g_idata_max,
  FD_TYPE *g_odata_min, FD_TYPE *g_odata_max) {
	
	__shared__ FD_TYPE s_min[blockSize];
	__shared__ FD_TYPE s_max[blockSize];
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned int gid = bid * blockSize + tid;
	const unsigned int gridSize = blockSize * gridDim.x * gridDim.y * gridDim.z;
	
	//initialize the shared memory data
	s_min[tid] = FLT_MAX;
  s_max[tid] = 0.f;
	while (gid < size) {
    s_max[tid] = max(s_max[tid], g_idata_max[gid]);
    s_min[tid] = min(s_min[tid], g_idata_min[gid]);
		gid += gridSize; 
	}
	__syncthreads();
	
  //reduction in shared memory
  for (int stride = 512; stride > 32; stride /= 2) {
    if (blockSize > stride && tid < stride) {
      s_max[tid] = max(s_max[tid], s_max[tid + stride]);
      s_min[tid] = min(s_min[tid], s_min[tid + stride]);
    }
    __syncthreads();
  }
	if (tid < 32) {
    warpMax<blockSize>(s_max, tid);
    warpMin<blockSize>(s_min, tid);
  }
  
	if (tid == 0) {
    g_odata_min[0] = s_min[0];
	  g_odata_max[0] = s_max[0];
  }
}

template <unsigned int blockSize>
__global__ void Step2ErrorKernel(const unsigned int size, 
    const FD_TYPE *dev_current_vtm, const FD_TYPE *dev_vtmpA, 
    const FD_TYPE *dev_vtmpB, FD_TYPE *dev_local_sum) {
      
  __shared__ FD_TYPE sdata[blockSize];
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned int gid = bid * blockSize + tid;
	const unsigned int gridSize = blockSize * gridDim.x * gridDim.y * gridDim.z;
	
	//initialize the shared memory data
	sdata[tid] = 0.f;
	FD_TYPE3 tmp;
	FD_TYPE tmp_magn;
	while (gid < size) {
    tmp.x = (dev_current_vtm[gid] + dev_vtmpB[gid]) * 0.5 - dev_vtmpA[gid];
    tmp.y = (dev_current_vtm[gid + size] + dev_vtmpB[gid + size]) * 0.5 
      - dev_vtmpA[gid + size];
    tmp.z = (dev_current_vtm[gid + 2 * size] + dev_vtmpB[gid + 2 * size]) * 0.5 
      - dev_vtmpA[gid + 2 * size];
		tmp_magn = tmp.x * tmp.x + tmp.y * tmp.y + tmp.z * tmp.z;
    sdata[tid] = max(sdata[tid], tmp_magn);
		gid += gridSize; 
	}
	__syncthreads();
	
	//reduction in shared memory
	if (blockSize >= 1024) { if (tid < 512 && sdata[tid] < sdata[tid + 512]) { sdata[tid] = sdata[tid + 512]; } __syncthreads(); }
	if (blockSize >= 512) { if (tid < 256 && sdata[tid] < sdata[tid + 256]) { sdata[tid] = sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128 && sdata[tid] < sdata[tid + 128]) { sdata[tid] = sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64 && sdata[tid] < sdata[tid + 64]) { sdata[tid] = sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32) warpMax<blockSize>(sdata, tid);
	if (tid == 0)  dev_local_sum[bid] = sdata[0];  
}

template <unsigned int blockSize>
__global__ void Step4ErrorKernel(const unsigned int size, 
    const FD_TYPE *dev_vtmpB, const FD_TYPE *dev_vtmpC, 
    FD_TYPE *dev_local_sum) {
      
  __shared__ FD_TYPE sdata[blockSize];
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned int gid = bid * blockSize + tid;
	const unsigned int gridSize = blockSize * gridDim.x * gridDim.y * gridDim.z;
	
	//initialize the shared memory data
	sdata[tid] = 0.f;
	FD_TYPE3 tmp;
	FD_TYPE tmp_magn;
	while (gid < size) {
    tmp.x = dev_vtmpB[gid] - 0.5f * dev_vtmpC[gid];
    tmp.y = dev_vtmpB[gid + size] - 0.5f * dev_vtmpC[gid + size];
    tmp.z = dev_vtmpB[gid + 2 * size] - 0.5f * dev_vtmpC[gid + 2 * size];
		tmp_magn = tmp.x * tmp.x + tmp.y * tmp.y + tmp.z * tmp.z;
    sdata[tid] = max(sdata[tid], tmp_magn);
		gid += gridSize; 
	}
	__syncthreads();
	
	//reduction in shared memory
	if (blockSize >= 1024) { if (tid < 512 && sdata[tid] < sdata[tid + 512]) { sdata[tid] = sdata[tid + 512]; } __syncthreads(); }
	if (blockSize >= 512) { if (tid < 256 && sdata[tid] < sdata[tid + 256]) { sdata[tid] = sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128 && sdata[tid] < sdata[tid + 128]) { sdata[tid] = sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64 && sdata[tid] < sdata[tid + 64]) { sdata[tid] = sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32) warpMax<blockSize>(sdata, tid);
	if (tid == 0)  dev_local_sum[bid] = sdata[0];  
}

template <unsigned int blockSize>
__global__ void maxDiffKernel(const unsigned int size, const FD_TYPE *dev_idata1,
    const FD_TYPE *dev_idata2, FD_TYPE *dev_local_sum) {
  __shared__ FD_TYPE sdata[blockSize];
	const unsigned int tid = threadIdx.x;
	const unsigned int bid = blockIdx.x + blockIdx.y * gridDim.x;
	unsigned int gid = bid * blockSize + tid;
	const unsigned int gridSize = blockSize * gridDim.x * gridDim.y * gridDim.z;
	
	//initialize the shared memory data
	sdata[tid] = 0.f;
	FD_TYPE3 tmp;
	FD_TYPE tmp_magn;
	while (gid < size) {
		tmp.x = dev_idata1[gid] - dev_idata2[gid];
		tmp.y = dev_idata1[gid + size] - dev_idata2[gid + size];
		tmp.z = dev_idata1[gid + 2 * size] - dev_idata2[gid + 2 * size];
		tmp_magn = tmp.x * tmp.x + tmp.y * tmp.y + tmp.z * tmp.z;
    sdata[tid] = max(sdata[tid], tmp_magn);
		gid += gridSize; 
	}
	__syncthreads();
	
	//reduction in shared memory
	if (blockSize >= 1024) { if (tid < 512 && sdata[tid] < sdata[tid + 512]) { sdata[tid] = sdata[tid + 512]; } __syncthreads(); }
	if (blockSize >= 512) { if (tid < 256 && sdata[tid] < sdata[tid + 256]) { sdata[tid] = sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128 && sdata[tid] < sdata[tid + 128]) { sdata[tid] = sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64 && sdata[tid] < sdata[tid + 64]) { sdata[tid] = sdata[tid + 64]; } __syncthreads(); }
	if (tid < 32) warpMax<blockSize>(sdata, tid);
	if (tid == 0) dev_local_sum[bid] = sdata[0];
}

// wraper for CUDA kernel functions
void backUpEnergy(FD_TYPE *bak_ptr, const FD_TYPE *src_ptr, 
    const OC_INDEX &size) {
  cudaMemcpy(bak_ptr, src_ptr, size * sizeof(FD_TYPE), cudaMemcpyDeviceToDevice);
}

void dm_dt(const dim3 &grid_size, const dim3 &block_size, const OC_INDEX size, 
  const FD_TYPE coef1, const FD_TYPE coef2, const OC_BOOL do_precess, 
  DEVSTRUCT &host_struct, const FD_TYPE *coef1Array, const FD_TYPE *coef2Array) {
   
  if (coef1Array == NULL || coef2Array == NULL) {
    Calculate_dm_dt_kernel<<<grid_size, block_size>>>(size, coef1, 
      coef2, do_precess, host_struct); 
  } else {
    Calculate_dm_dt_kernel_freeCoef<<<grid_size, block_size>>>(size, coef1Array, 
      coef2Array, do_precess, host_struct); 
  }
}

void nextSpin(const dim3 &grid_size, const dim3 &block_size, const OC_INDEX &size, 
    const FD_TYPE &stepsize, const DEVSTRUCT &host_struct) {
  
  CalcNextSpin_kernel<<<grid_size, block_size>>>(size, stepsize, host_struct);
}

void dmDtError(const dim3 &grid_size, const dim3 &block_size,
    const OC_INDEX &size, const OC_INDEX &reduce_size, const OC_INDEX &BLKSIZE,
    const DEVSTRUCT &host_struct, FD_TYPE *dev_max_error) {
  switch (BLKSIZE) {
    case 1024:
    dm_dt_err_kernel<1024><<< grid_size, block_size>>>(size, host_struct);
    final_max_kernel<1024><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum, dev_max_error);
    break;
	case 512:
    dm_dt_err_kernel<512><<< grid_size, block_size>>>(size, host_struct);
    final_max_kernel<512><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum, dev_max_error);
    break;
	case 256:
    dm_dt_err_kernel<256><<< grid_size, block_size>>>(size, host_struct);
    final_max_kernel<256><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum, dev_max_error);
    break;
	case 128:
    dm_dt_err_kernel<128><<< grid_size, block_size>>>(size, host_struct);
    final_max_kernel<128><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum, dev_max_error);
    break;
	case 64:
    dm_dt_err_kernel< 64><<< grid_size, block_size>>>(size, host_struct);
    final_max_kernel<64><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum, dev_max_error);
    break;
	case 32:
    dm_dt_err_kernel< 32><<< grid_size, block_size>>>(size, host_struct);
    final_max_kernel<32><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum, dev_max_error);
    break;
	case 16:
    dm_dt_err_kernel< 16><<< grid_size, block_size>>>(size, host_struct);
    final_max_kernel<16><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum, dev_max_error);
    break;
	case 8:
    dm_dt_err_kernel< 8><<< grid_size, block_size>>>(size, host_struct);
    final_max_kernel<8><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum, dev_max_error);
    break;
	case 4:
    dm_dt_err_kernel< 4><<< grid_size, block_size>>>(size, host_struct);
    final_max_kernel<4><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum, dev_max_error);
    break;
	case 2:
    dm_dt_err_kernel< 2><<< grid_size, block_size>>>(size, host_struct);
    final_max_kernel<2><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum, dev_max_error);
    break;
	case 1:
    dm_dt_err_kernel< 1><<< grid_size, block_size>>>(size, host_struct);
    final_max_kernel<1><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum, dev_max_error);
    break;
  }    
}

void collectDmDtStatistics(const dim3 &grid_size, const dim3 &block_size,
    const OC_INDEX &size, const OC_INDEX &reduce_size, const OC_INDEX &BLKSIZE,
    const DEVSTRUCT &host_struct, FD_TYPE *dev_dE_dt_sum, 
    FD_TYPE *dev_max_dm_dt_sq, const FD_TYPE *coef1Array, 
    const FD_TYPE *coef2Array) {
      
  switch (BLKSIZE) {
    case 1024:
    if (coef1Array == NULL || coef2Array == NULL) {
      collect_dm_dt_kernel<1024><<< grid_size, block_size>>>(size, host_struct);
    } else {
      collect_dm_dt_kernel_freeCoef<1024><<< grid_size, block_size>>>(size, 
        host_struct, coef1Array, coef2Array);
    }
    final_reduce_kernel<1024><<<1,block_size>>>(reduce_size,
      host_struct.dev_local_sum, dev_dE_dt_sum);
    final_max_kernel<1024><<<1, block_size>>>(reduce_size,
      host_struct.dev_local_sum+reduce_size, dev_max_dm_dt_sq);
    break;
	case 512:
    if (coef1Array == NULL || coef2Array == NULL) {
      collect_dm_dt_kernel<512><<< grid_size, block_size>>>(size, host_struct);
    } else {
      collect_dm_dt_kernel_freeCoef<512><<< grid_size, block_size>>>(size, 
        host_struct, coef1Array, coef2Array);
    }
    final_reduce_kernel<512><<<1,block_size>>>(reduce_size,
      host_struct.dev_local_sum, dev_dE_dt_sum);
    final_max_kernel<512><<<1, block_size>>>(reduce_size,
      host_struct.dev_local_sum+reduce_size, dev_max_dm_dt_sq);
    break;
	case 256:
    if (coef1Array == NULL || coef2Array == NULL) {
      collect_dm_dt_kernel<256><<< grid_size, block_size>>>(size, host_struct);
    } else {
      collect_dm_dt_kernel_freeCoef<256><<< grid_size, block_size>>>(size, 
        host_struct, coef1Array, coef2Array);
    }
    final_reduce_kernel<256><<<1,block_size>>>(reduce_size,
      host_struct.dev_local_sum, dev_dE_dt_sum);
    final_max_kernel<256><<<1, block_size>>>(reduce_size,
      host_struct.dev_local_sum+reduce_size, dev_max_dm_dt_sq);
    break;
	case 128:
    if (coef1Array == NULL || coef2Array == NULL) {
      collect_dm_dt_kernel<128><<< grid_size, block_size>>>(size, host_struct);
    } else {
      collect_dm_dt_kernel_freeCoef<128><<< grid_size, block_size>>>(size, 
        host_struct, coef1Array, coef2Array);
    }
    final_reduce_kernel<128><<<1,block_size>>>(reduce_size,
      host_struct.dev_local_sum, dev_dE_dt_sum);
    final_max_kernel<128><<<1, block_size>>>(reduce_size,
      host_struct.dev_local_sum+reduce_size, dev_max_dm_dt_sq);
    break;
	case 64:
    if (coef1Array == NULL || coef2Array == NULL) {
      collect_dm_dt_kernel<64><<< grid_size, block_size>>>(size, host_struct);
    } else {
      collect_dm_dt_kernel_freeCoef<64><<< grid_size, block_size>>>(size, 
        host_struct, coef1Array, coef2Array);
    }
    final_reduce_kernel<64><<<1,block_size>>>(reduce_size,
      host_struct.dev_local_sum, dev_dE_dt_sum);
    final_max_kernel<64><<<1, block_size>>>(reduce_size,
      host_struct.dev_local_sum+reduce_size, dev_max_dm_dt_sq);
    break;
	case 32:
    if (coef1Array == NULL || coef2Array == NULL) {
      collect_dm_dt_kernel<32><<< grid_size, block_size>>>(size, host_struct);
    } else {
      collect_dm_dt_kernel_freeCoef<32><<< grid_size, block_size>>>(size, 
        host_struct, coef1Array, coef2Array);
    }
    final_reduce_kernel<32><<<1,block_size>>>(reduce_size,
      host_struct.dev_local_sum, dev_dE_dt_sum);
    final_max_kernel<32><<<1, block_size>>>(reduce_size,
      host_struct.dev_local_sum+reduce_size, dev_max_dm_dt_sq);
    break;
	case 16:
    if (coef1Array == NULL || coef2Array == NULL) {
      collect_dm_dt_kernel<16><<< grid_size, block_size>>>(size, host_struct);
    } else {
      collect_dm_dt_kernel_freeCoef<16><<< grid_size, block_size>>>(size, 
        host_struct, coef1Array, coef2Array);
    }
    final_reduce_kernel<16><<<1,block_size>>>(reduce_size,
      host_struct.dev_local_sum, dev_dE_dt_sum);
    final_max_kernel<16><<<1, block_size>>>(reduce_size,
      host_struct.dev_local_sum+reduce_size, dev_max_dm_dt_sq);
    break;
	case 8:
    if (coef1Array == NULL || coef2Array == NULL) {
      collect_dm_dt_kernel<8><<< grid_size, block_size>>>(size, host_struct);
    } else {
      collect_dm_dt_kernel_freeCoef<8><<< grid_size, block_size>>>(size, 
        host_struct, coef1Array, coef2Array);
    }
    final_reduce_kernel<8><<<1,block_size>>>(reduce_size,
      host_struct.dev_local_sum, dev_dE_dt_sum);
    final_max_kernel<8><<<1, block_size>>>(reduce_size,
      host_struct.dev_local_sum+reduce_size, dev_max_dm_dt_sq);
    break;
	case 4:
    if (coef1Array == NULL || coef2Array == NULL) {
      collect_dm_dt_kernel<4><<< grid_size, block_size>>>(size, host_struct);
    } else {
      collect_dm_dt_kernel_freeCoef<4><<< grid_size, block_size>>>(size, 
        host_struct, coef1Array, coef2Array);
    }
    final_reduce_kernel<4><<<1,block_size>>>(reduce_size,
      host_struct.dev_local_sum, dev_dE_dt_sum);
    final_max_kernel<4><<<1, block_size>>>(reduce_size,
      host_struct.dev_local_sum+reduce_size, dev_max_dm_dt_sq);
    break;
	case 2:
    if (coef1Array == NULL || coef2Array == NULL) {
      collect_dm_dt_kernel<2><<< grid_size, block_size>>>(size, host_struct);
    } else {
      collect_dm_dt_kernel_freeCoef<2><<< grid_size, block_size>>>(size, 
        host_struct, coef1Array, coef2Array);
    }
    final_reduce_kernel<2><<<1,block_size>>>(reduce_size,
      host_struct.dev_local_sum, dev_dE_dt_sum);
    final_max_kernel<2><<<1, block_size>>>(reduce_size,
      host_struct.dev_local_sum+reduce_size, dev_max_dm_dt_sq);
    break;
	case 1:
    if (coef1Array == NULL || coef2Array == NULL) {
      collect_dm_dt_kernel<1><<< grid_size, block_size>>>(size, host_struct);
    } else {
      collect_dm_dt_kernel_freeCoef<1><<< grid_size, block_size>>>(size, 
        host_struct, coef1Array, coef2Array);
    }
    final_reduce_kernel<1><<<1,block_size>>>(reduce_size,
      host_struct.dev_local_sum, dev_dE_dt_sum);
    final_max_kernel<1><<<1, block_size>>>(reduce_size,
      host_struct.dev_local_sum+reduce_size, dev_max_dm_dt_sq);
    break;
  }
}

void collectEnergyStatistics(const dim3 &grid_size, const dim3 &block_size,
    const OC_INDEX &size, const OC_INDEX &reduce_size, const OC_INDEX &BLKSIZE,
    const DEVSTRUCT &host_struct, FD_TYPE *dev_dE, FD_TYPE *dev_var_dE, 
    FD_TYPE *dev_total_E) {
      
  switch (BLKSIZE)
  {
    case 1024:
    energy_err_kernel<1024><<< grid_size, block_size>>>(size, host_struct);
    final_reduce_kernel<1024><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum, dev_total_E);
    final_reduce_kernel<1024><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum+reduce_size, dev_dE);
    final_reduce_kernel<1024><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum+2*reduce_size, dev_var_dE);
    break;
	case 512:
    energy_err_kernel<512><<< grid_size, block_size>>>(size, host_struct);
    final_reduce_kernel<512><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum, dev_total_E);
    final_reduce_kernel<512><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum+reduce_size, dev_dE);
    final_reduce_kernel<512><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum+2*reduce_size, dev_var_dE);
    break;
	case 256:
    energy_err_kernel<256><<< grid_size, block_size>>>(size, host_struct);
    final_reduce_kernel<256><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum, dev_total_E);
    final_reduce_kernel<256><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum+reduce_size, dev_dE);
    final_reduce_kernel<256><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum+2*reduce_size, dev_var_dE);
    break;
	case 128:
    energy_err_kernel<128><<< grid_size, block_size>>>(size, host_struct);
    final_reduce_kernel<128><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum, dev_total_E);
    final_reduce_kernel<128><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum+reduce_size, dev_dE);
    final_reduce_kernel<128><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum+2*reduce_size, dev_var_dE);
    break;
	case 64:
    energy_err_kernel<64><<< grid_size, block_size>>>(size, host_struct);
    final_reduce_kernel<64><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum, dev_total_E);
    final_reduce_kernel<64><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum+reduce_size, dev_dE);
    final_reduce_kernel<64><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum+2*reduce_size, dev_var_dE);
    break;
	case 32:
    energy_err_kernel<32><<< grid_size, block_size>>>(size, host_struct);
    final_reduce_kernel<32><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum, dev_total_E);
    final_reduce_kernel<32><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum+reduce_size, dev_dE);
    final_reduce_kernel<32><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum+2*reduce_size, dev_var_dE);
    break;
	case 16:
    energy_err_kernel<16><<< grid_size, block_size>>>(size, host_struct);
    final_reduce_kernel<16><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum, dev_total_E);
    final_reduce_kernel<16><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum+reduce_size, dev_dE);
    final_reduce_kernel<16><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum+2*reduce_size, dev_var_dE);
    break;
	case 8:
    energy_err_kernel<8><<< grid_size, block_size>>>(size, host_struct);
    final_reduce_kernel<8><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum, dev_total_E);
    final_reduce_kernel<8><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum+reduce_size, dev_dE);
    final_reduce_kernel<8><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum+2*reduce_size, dev_var_dE);
    break;
	case 4:
    energy_err_kernel<4><<< grid_size, block_size>>>(size, host_struct);
    final_reduce_kernel<4><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum, dev_total_E);
    final_reduce_kernel<4><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum+reduce_size, dev_dE);
    final_reduce_kernel<4><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum+2*reduce_size, dev_var_dE);
    break;
	case 2:
    energy_err_kernel<2><<< grid_size, block_size>>>(size, host_struct);
    final_reduce_kernel<2><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum, dev_total_E);
    final_reduce_kernel<2><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum+reduce_size, dev_dE);
    final_reduce_kernel<2><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum+2*reduce_size, dev_var_dE);
    break;
	case 1:
    energy_err_kernel<1><<< grid_size, block_size>>>(size, host_struct);
    final_reduce_kernel<1><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum, dev_total_E);
    final_reduce_kernel<1><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum+reduce_size, dev_dE);
    final_reduce_kernel<1><<<1, block_size>>>(reduce_size, host_struct.dev_local_sum+2*reduce_size, dev_var_dE);
    break;
  }    
}

void adjustSpin(const dim3 &grid_size, const dim3 &block_size, const OC_INDEX &size,
  const OC_INDEX &reduce_size, const OC_INDEX &BLKSIZE,
  const FD_TYPE &stepsize, const DEVSTRUCT &host_struct,
  const FD_TYPE *dev_MValue_old, FD_TYPE *dev_MValue_new,
  FD_TYPE *dev_min_magsq, FD_TYPE *dev_max_magsq) {
  
  if (block_size.x != BLKSIZE || block_size.y != 1 || block_size.z != 1) {
    throw std::runtime_error("block_size should be 1D and has equal size as BLKSIZE");
  }
  
  switch (BLKSIZE) {
    case 1024: 
      CalcNewSpinAndCollectKernel<1024><<<grid_size, block_size>>>(size, 
        reduce_size, stepsize, host_struct, dev_MValue_old, dev_MValue_new);
      final_min_max_kernel<1024><<<1, block_size>>>(reduce_size,
        host_struct.dev_local_sum, 
        host_struct.dev_local_sum + reduce_size, dev_min_magsq, dev_max_magsq);
      break;
    case 512: 
      CalcNewSpinAndCollectKernel<512><<<grid_size, block_size>>>(size, 
        reduce_size, stepsize, host_struct, dev_MValue_old, dev_MValue_new);
      final_min_max_kernel<512><<<1, block_size>>>(reduce_size,
        host_struct.dev_local_sum, 
        host_struct.dev_local_sum + reduce_size, dev_min_magsq, dev_max_magsq);
      break;
    case 256: 
      CalcNewSpinAndCollectKernel<256><<<grid_size, block_size>>>(size, 
        reduce_size, stepsize, host_struct, dev_MValue_old, dev_MValue_new);
      final_min_max_kernel<256><<<1, block_size>>>(reduce_size,
        host_struct.dev_local_sum, 
        host_struct.dev_local_sum + reduce_size, dev_min_magsq, dev_max_magsq);
      break;
    case 128: 
      CalcNewSpinAndCollectKernel<128><<<grid_size, block_size>>>(size, 
        reduce_size, stepsize, host_struct, dev_MValue_old, dev_MValue_new);
      final_min_max_kernel<128><<<1, block_size>>>(reduce_size,
        host_struct.dev_local_sum, 
        host_struct.dev_local_sum + reduce_size, dev_min_magsq, dev_max_magsq);
      break;
    case 64: 
      CalcNewSpinAndCollectKernel<64><<<grid_size, block_size>>>(size, 
        reduce_size, stepsize, host_struct, dev_MValue_old, dev_MValue_new);
      final_min_max_kernel<64><<<1, block_size>>>(reduce_size,
        host_struct.dev_local_sum, 
        host_struct.dev_local_sum + reduce_size, dev_min_magsq, dev_max_magsq);
      break;
    case 32: 
      CalcNewSpinAndCollectKernel<32><<<grid_size, block_size>>>(size, 
        reduce_size, stepsize, host_struct, dev_MValue_old, dev_MValue_new);
      final_min_max_kernel<32><<<1, block_size>>>(reduce_size,
        host_struct.dev_local_sum, 
        host_struct.dev_local_sum + reduce_size, dev_min_magsq, dev_max_magsq);
      break;
    case 16: 
      CalcNewSpinAndCollectKernel<16><<<grid_size, block_size>>>(size, 
        reduce_size, stepsize, host_struct, dev_MValue_old, dev_MValue_new);
      final_min_max_kernel<16><<<1, block_size>>>(reduce_size,
        host_struct.dev_local_sum, 
        host_struct.dev_local_sum + reduce_size, dev_min_magsq, dev_max_magsq);
      break;
    case 8: 
      CalcNewSpinAndCollectKernel<8><<<grid_size, block_size>>>(size, 
        reduce_size, stepsize, host_struct, dev_MValue_old, dev_MValue_new);
      final_min_max_kernel<8><<<1, block_size>>>(reduce_size,
        host_struct.dev_local_sum, 
        host_struct.dev_local_sum + reduce_size, dev_min_magsq, dev_max_magsq);
      break;
    case 4: 
      CalcNewSpinAndCollectKernel<4><<<grid_size, block_size>>>(size, 
        reduce_size, stepsize, host_struct, dev_MValue_old, dev_MValue_new);
      final_min_max_kernel<4><<<1, block_size>>>(reduce_size,
        host_struct.dev_local_sum, 
        host_struct.dev_local_sum + reduce_size, dev_min_magsq, dev_max_magsq);
      break;
    case 2: 
      CalcNewSpinAndCollectKernel<2><<<grid_size, block_size>>>(size, 
        reduce_size, stepsize, host_struct, dev_MValue_old, dev_MValue_new);
      final_min_max_kernel<2><<<1, block_size>>>(reduce_size,
        host_struct.dev_local_sum, 
        host_struct.dev_local_sum + reduce_size, dev_min_magsq, dev_max_magsq);
      break;
    case 1: 
      CalcNewSpinAndCollectKernel<1><<<grid_size, block_size>>>(size, 
        reduce_size, stepsize, host_struct, dev_MValue_old, dev_MValue_new);
      final_min_max_kernel<1><<<1, block_size>>>(reduce_size,
        host_struct.dev_local_sum, 
        host_struct.dev_local_sum + reduce_size, dev_min_magsq, dev_max_magsq);
      break;
  }
}

void accumulate(const dim3 &grid_size, const dim3 &block_size, const OC_INDEX &size,
    const FD_TYPE &mult1, const FD_TYPE &mult2, const FD_TYPE *d_increment, const FD_TYPE *d_idata, 
    FD_TYPE *d_odata) {
  Accumulate_kernel<<<grid_size, block_size>>>(size, mult1, mult2, d_increment, 
    d_idata, d_odata);
}

void dmDtErrorStep2(const dim3 &grid_size, const dim3 &block_size,
    const OC_INDEX &size, const OC_INDEX &reduce_size, const OC_INDEX &BLKSIZE,
    const FD_TYPE *dev_current_data, const FD_TYPE *dev_vtmpA, 
    const FD_TYPE * dev_vtmpB, FD_TYPE *dev_local_sum, FD_TYPE *dev_max_error_sq) {
  switch (BLKSIZE) {
    case 1024:
      Step2ErrorKernel<1024><<<grid_size, block_size>>>(size, dev_current_data, 
        dev_vtmpA, dev_vtmpB, dev_local_sum);
      final_max_kernel<1024><<<1, block_size>>>(reduce_size, dev_local_sum, 
        dev_max_error_sq);
      break;
    case 512:
      Step2ErrorKernel<512><<<grid_size, block_size>>>(size, dev_current_data, 
        dev_vtmpA, dev_vtmpB, dev_local_sum);
      final_max_kernel<512><<<1, block_size>>>(reduce_size, dev_local_sum, 
        dev_max_error_sq);
      break;
    case 256:
      Step2ErrorKernel<256><<<grid_size, block_size>>>(size, dev_current_data, 
        dev_vtmpA, dev_vtmpB, dev_local_sum);
      final_max_kernel<256><<<1, block_size>>>(reduce_size, dev_local_sum, 
        dev_max_error_sq);
      break;
    case 128:
      Step2ErrorKernel<128><<<grid_size, block_size>>>(size, dev_current_data, 
        dev_vtmpA, dev_vtmpB, dev_local_sum);
      final_max_kernel<128><<<1, block_size>>>(reduce_size, dev_local_sum, 
        dev_max_error_sq);
      break;
    case 64:
      Step2ErrorKernel<64><<<grid_size, block_size>>>(size, dev_current_data, 
        dev_vtmpA, dev_vtmpB, dev_local_sum);
      final_max_kernel<64><<<1, block_size>>>(reduce_size, dev_local_sum, 
        dev_max_error_sq);
      break;
    case 32:
      Step2ErrorKernel<32><<<grid_size, block_size>>>(size, dev_current_data, 
        dev_vtmpA, dev_vtmpB, dev_local_sum);
      final_max_kernel<32><<<1, block_size>>>(reduce_size, dev_local_sum, 
        dev_max_error_sq);
      break;
    case 16:
      Step2ErrorKernel<16><<<grid_size, block_size>>>(size, dev_current_data, 
        dev_vtmpA, dev_vtmpB, dev_local_sum);
      final_max_kernel<16><<<1, block_size>>>(reduce_size, dev_local_sum, 
        dev_max_error_sq);
      break;
    case 8:
      Step2ErrorKernel<8><<<grid_size, block_size>>>(size, dev_current_data, 
        dev_vtmpA, dev_vtmpB, dev_local_sum);
      final_max_kernel<8><<<1, block_size>>>(reduce_size, dev_local_sum, 
        dev_max_error_sq);
      break;
    case 4:
      Step2ErrorKernel<4><<<grid_size, block_size>>>(size, dev_current_data, 
        dev_vtmpA, dev_vtmpB, dev_local_sum);
      final_max_kernel<4><<<1, block_size>>>(reduce_size, dev_local_sum, 
        dev_max_error_sq);
      break;
    case 2:
      Step2ErrorKernel<2><<<grid_size, block_size>>>(size, dev_current_data, 
        dev_vtmpA, dev_vtmpB, dev_local_sum);
      final_max_kernel<2><<<1, block_size>>>(reduce_size, dev_local_sum, 
        dev_max_error_sq);
      break;
    case 1:
      Step2ErrorKernel<1><<<grid_size, block_size>>>(size, dev_current_data, 
        dev_vtmpA, dev_vtmpB, dev_local_sum);
      final_max_kernel<1><<<1, block_size>>>(reduce_size, dev_local_sum, 
        dev_max_error_sq);
      break;
  }
}

void makeUnitAndCollectMinMax(const dim3 &grid_size, const dim3 &block_size, 
    const OC_INDEX &size,const OC_INDEX &reduce_size, const OC_INDEX &BLKSIZE,
    FD_TYPE *dev_data, FD_TYPE *dev_local_sum, FD_TYPE *dev_min_magsq, 
    FD_TYPE *dev_max_magsq) {
  
  if (block_size.x != BLKSIZE || block_size.y != 1 || block_size.z != 1) {
    throw std::runtime_error("block_size should be 1D and has equal size as BLKSIZE");
  }
  
  switch (BLKSIZE) {
    case 1024: 
      MakeUnitAndCollectKernel<1024><<<grid_size, block_size>>>(size, 
        reduce_size, dev_data, dev_local_sum);
      final_min_max_kernel<1024><<<1, block_size>>>(reduce_size, dev_local_sum, 
        dev_local_sum + reduce_size, dev_min_magsq, dev_max_magsq);
      break;
    case 512: 
      MakeUnitAndCollectKernel<512><<<grid_size, block_size>>>(size, 
        reduce_size, dev_data, dev_local_sum);
      final_min_max_kernel<512><<<1, block_size>>>(reduce_size, dev_local_sum, 
        dev_local_sum + reduce_size, dev_min_magsq, dev_max_magsq);
      break;
    case 256: 
      MakeUnitAndCollectKernel<256><<<grid_size, block_size>>>(size, 
        reduce_size, dev_data, dev_local_sum);
      final_min_max_kernel<256><<<1, block_size>>>(reduce_size, dev_local_sum, 
        dev_local_sum + reduce_size, dev_min_magsq, dev_max_magsq);
      break;
    case 128: 
      MakeUnitAndCollectKernel<128><<<grid_size, block_size>>>(size, 
        reduce_size, dev_data, dev_local_sum);
      final_min_max_kernel<128><<<1, block_size>>>(reduce_size, dev_local_sum, 
        dev_local_sum + reduce_size, dev_min_magsq, dev_max_magsq);
      break;
    case 64: 
      MakeUnitAndCollectKernel<64><<<grid_size, block_size>>>(size, 
        reduce_size, dev_data, dev_local_sum);
      final_min_max_kernel<64><<<1, block_size>>>(reduce_size, dev_local_sum, 
        dev_local_sum + reduce_size, dev_min_magsq, dev_max_magsq);
      break;
    case 32: 
      MakeUnitAndCollectKernel<32><<<grid_size, block_size>>>(size, 
        reduce_size, dev_data, dev_local_sum);
      final_min_max_kernel<32><<<1, block_size>>>(reduce_size, dev_local_sum, 
        dev_local_sum + reduce_size, dev_min_magsq, dev_max_magsq);
      break;
    case 16: 
      MakeUnitAndCollectKernel<16><<<grid_size, block_size>>>(size, 
        reduce_size, dev_data, dev_local_sum);
      final_min_max_kernel<16><<<1, block_size>>>(reduce_size, dev_local_sum, 
        dev_local_sum + reduce_size, dev_min_magsq, dev_max_magsq);
      break;
    case 8: 
      MakeUnitAndCollectKernel<8><<<grid_size, block_size>>>(size, 
        reduce_size, dev_data, dev_local_sum);
      final_min_max_kernel<8><<<1, block_size>>>(reduce_size, dev_local_sum, 
        dev_local_sum + reduce_size, dev_min_magsq, dev_max_magsq);
      break;
    case 4: 
      MakeUnitAndCollectKernel<4><<<grid_size, block_size>>>(size, 
        reduce_size, dev_data, dev_local_sum);
      final_min_max_kernel<4><<<1, block_size>>>(reduce_size, dev_local_sum, 
        dev_local_sum + reduce_size, dev_min_magsq, dev_max_magsq);
      break;
    case 2: 
      MakeUnitAndCollectKernel<2><<<grid_size, block_size>>>(size, 
        reduce_size, dev_data, dev_local_sum);
      final_min_max_kernel<2><<<1, block_size>>>(reduce_size, dev_local_sum, 
        dev_local_sum + reduce_size, dev_min_magsq, dev_max_magsq);
      break;
    case 1: 
      MakeUnitAndCollectKernel<1><<<grid_size, block_size>>>(size, 
        reduce_size, dev_data, dev_local_sum);
      final_min_max_kernel<1><<<1, block_size>>>(reduce_size, dev_local_sum, 
        dev_local_sum + reduce_size, dev_min_magsq, dev_max_magsq);
      break;
  }
}

void maxDiff(const dim3 &grid_size, const dim3 &block_size, 
    const OC_INDEX &size,const OC_INDEX &reduce_size, const OC_INDEX &BLKSIZE,
    const FD_TYPE *dev_idata1, const FD_TYPE *dev_idata2, FD_TYPE *dev_local_sum,
    FD_TYPE *dev_max_error_sq) {
  switch (BLKSIZE) {
    case 1024:
      maxDiffKernel<1024><<<grid_size, block_size>>>(size, dev_idata1, dev_idata2, dev_local_sum);
      final_max_kernel<1024><<<1, block_size>>>(reduce_size, dev_local_sum, dev_max_error_sq);
      break;
    case 512:
      maxDiffKernel<512><<<grid_size, block_size>>>(size, dev_idata1, dev_idata2, dev_local_sum);
      final_max_kernel<512><<<1, block_size>>>(reduce_size, dev_local_sum, dev_max_error_sq);
      break;
    case 256:
      maxDiffKernel<256><<<grid_size, block_size>>>(size, dev_idata1, dev_idata2, dev_local_sum);
      final_max_kernel<256><<<1, block_size>>>(reduce_size, dev_local_sum, dev_max_error_sq);
      break;
    case 128:
      maxDiffKernel<128><<<grid_size, block_size>>>(size, dev_idata1, dev_idata2, dev_local_sum);
      final_max_kernel<128><<<1, block_size>>>(reduce_size, dev_local_sum, dev_max_error_sq);
      break;
    case 64:
      maxDiffKernel<64><<<grid_size, block_size>>>(size, dev_idata1, dev_idata2, dev_local_sum);
      final_max_kernel<64><<<1, block_size>>>(reduce_size, dev_local_sum, dev_max_error_sq);
      break;
    case 32:
      maxDiffKernel<32><<<grid_size, block_size>>>(size, dev_idata1, dev_idata2, dev_local_sum);
      final_max_kernel<32><<<1, block_size>>>(reduce_size, dev_local_sum, dev_max_error_sq);
      break;
    case 16:
      maxDiffKernel<16><<<grid_size, block_size>>>(size, dev_idata1, dev_idata2, dev_local_sum);
      final_max_kernel<16><<<1, block_size>>>(reduce_size, dev_local_sum, dev_max_error_sq);
      break;
    case 8:
      maxDiffKernel<8><<<grid_size, block_size>>>(size, dev_idata1, dev_idata2, dev_local_sum);
      final_max_kernel<8><<<1, block_size>>>(reduce_size, dev_local_sum, dev_max_error_sq);
      break;
    case 4:
      maxDiffKernel<4><<<grid_size, block_size>>>(size, dev_idata1, dev_idata2, dev_local_sum);
      final_max_kernel<4><<<1, block_size>>>(reduce_size, dev_local_sum, dev_max_error_sq);
      break;
    case 2:
      maxDiffKernel<2><<<grid_size, block_size>>>(size, dev_idata1, dev_idata2, dev_local_sum);
      final_max_kernel<2><<<1, block_size>>>(reduce_size, dev_local_sum, dev_max_error_sq);
      break;
    case 1:
      maxDiffKernel< 1><<<grid_size, block_size>>>(size, dev_idata1, dev_idata2, dev_local_sum);
      final_max_kernel<1><<<1, block_size>>>(reduce_size, dev_local_sum, dev_max_error_sq);
      break;
  }    
}

void dmDtErrorStep4(const dim3 &grid_size, const dim3 &block_size,
    const OC_INDEX &size, const OC_INDEX &reduce_size, const OC_INDEX &BLKSIZE,
    const FD_TYPE *dev_vtmpB, const FD_TYPE * dev_vtmpC, 
    FD_TYPE *dev_local_sum, FD_TYPE *dev_max_error_sq) {
  switch (BLKSIZE) {
    case 1024:
      Step4ErrorKernel<1024><<<grid_size, block_size>>>(size, dev_vtmpB, 
        dev_vtmpC, dev_local_sum);
      final_max_kernel<1024><<<1, block_size>>>(reduce_size, dev_local_sum, 
        dev_max_error_sq);
      break;
    case 512:
      Step4ErrorKernel<512><<<grid_size, block_size>>>(size, dev_vtmpB, 
        dev_vtmpC, dev_local_sum);
      final_max_kernel<512><<<1, block_size>>>(reduce_size, dev_local_sum, 
        dev_max_error_sq);
      break;
    case 256:
      Step4ErrorKernel<256><<<grid_size, block_size>>>(size, dev_vtmpB, 
        dev_vtmpC, dev_local_sum);
      final_max_kernel<256><<<1, block_size>>>(reduce_size, dev_local_sum, 
        dev_max_error_sq);
      break;
    case 128:
      Step4ErrorKernel<128><<<grid_size, block_size>>>(size, dev_vtmpB, 
        dev_vtmpC, dev_local_sum);
      final_max_kernel<128><<<1, block_size>>>(reduce_size, dev_local_sum, 
        dev_max_error_sq);
      break;
    case 64:
      Step4ErrorKernel<64><<<grid_size, block_size>>>(size, dev_vtmpB, 
        dev_vtmpC, dev_local_sum);
      final_max_kernel<64><<<1, block_size>>>(reduce_size, dev_local_sum, 
        dev_max_error_sq);
      break;
    case 32:
      Step4ErrorKernel<32><<<grid_size, block_size>>>(size, dev_vtmpB, 
        dev_vtmpC, dev_local_sum);
      final_max_kernel<32><<<1, block_size>>>(reduce_size, dev_local_sum, 
        dev_max_error_sq);
      break;
    case 16:
      Step4ErrorKernel<16><<<grid_size, block_size>>>(size, dev_vtmpB, 
        dev_vtmpC, dev_local_sum);
      final_max_kernel<16><<<1, block_size>>>(reduce_size, dev_local_sum, 
        dev_max_error_sq);
      break;
    case 8:
      Step4ErrorKernel<8><<<grid_size, block_size>>>(size, dev_vtmpB, 
        dev_vtmpC, dev_local_sum);
      final_max_kernel<8><<<1, block_size>>>(reduce_size, dev_local_sum, 
        dev_max_error_sq);
      break;
    case 4:
      Step4ErrorKernel<4><<<grid_size, block_size>>>(size, dev_vtmpB, 
        dev_vtmpC, dev_local_sum);
      final_max_kernel<4><<<1, block_size>>>(reduce_size, dev_local_sum, 
        dev_max_error_sq);
      break;
    case 2:
      Step4ErrorKernel<2><<<grid_size, block_size>>>(size, dev_vtmpB, 
        dev_vtmpC, dev_local_sum);
      final_max_kernel<2><<<1, block_size>>>(reduce_size, dev_local_sum, 
        dev_max_error_sq);
      break;
    case 1:
      Step4ErrorKernel<1><<<grid_size, block_size>>>(size, dev_vtmpB, 
        dev_vtmpC, dev_local_sum);
      final_max_kernel<1><<<1, block_size>>>(reduce_size, dev_local_sum, 
        dev_max_error_sq);
      break;
  }
}