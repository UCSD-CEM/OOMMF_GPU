#include "nb.h"
#include "GPU_devstruct.h"

inline __device__ void glb2share( const FD_TYPE* d_MValue, FD_TYPE* s_MValue,
	const unsigned int glb_idx, const unsigned int s_idx, const unsigned int d_size,
	const unsigned int s_size){
	
	s_MValue[s_idx]   = d_MValue[glb_idx];
	s_MValue[s_idx+s_size] = d_MValue[glb_idx+d_size];
	s_MValue[s_idx+2*s_size] = d_MValue[glb_idx+2*d_size];
}

inline __device__ void set_FD_TYPE3M(const unsigned int index, const uint3 dev_Dim, FD_TYPE* dev_HValue,
	const FD_TYPE valueX, const FD_TYPE valueY, const FD_TYPE valueZ){
	
	unsigned int Msize = dev_Dim.x * dev_Dim.y * dev_Dim.z;
	dev_HValue[index  ] = valueX;
	dev_HValue[index+Msize] = valueY;
	dev_HValue[index+2*Msize] = valueZ;
}

inline __device__ void add_FD_TYPE3M(const unsigned int index, const uint3 dev_Dim, FD_TYPE* dev_HValue,
	const FD_TYPE valueX, const FD_TYPE valueY, const FD_TYPE valueZ){
	
	unsigned int Msize = dev_Dim.x * dev_Dim.y * dev_Dim.z;
	dev_HValue[index  ] += valueX;
	dev_HValue[index+Msize] += valueY;
	dev_HValue[index+2*Msize] += valueZ;
}

inline __device__ FD_TYPE sqr_FD_TYPE3M(FD_TYPE3 a){
	return a.x * a.x + a.y * a.y + a.z * a.z;
}
inline __device__ void operator+=(FD_TYPE3 &a, FD_TYPE3 b) {
    a.x += b.x; 
	a.y += b.y; 
	a.z += b.z;
}
inline __device__ void operator*=(FD_TYPE3 &a, FD_TYPE b) {
    a.x *= b; 
	a.y *= b; 
	a.z *= b;
}
inline __device__ FD_TYPE3 operator-(FD_TYPE3 a, FD_TYPE3 b) {
	FD_TYPE3 m0;
	m0.x = a.x - b.x;
	m0.y = a.y - b.y;
	m0.z = a.z - b.z;
    return m0;
}
inline __device__ FD_TYPE3 operator*(FD_TYPE s, FD_TYPE3 a) {
	FD_TYPE3 m0;
	m0.x = s*a.x;
	m0.y = s*a.y;
	m0.z = s*a.z;
    return m0;
}

__global__  __launch_bounds__(1024, 2) 
void s_exchUniform_kernel(const FD_TYPE* __restrict__ dev_MValue,
		FD_TYPE* dev_H, FD_TYPE* dev_Energy, FD_TYPE* dev_Torque,
    const uint3 dev_Dim, const FD_TYPE3 dev_wgt, 
    const FD_TYPE* __restrict__ dev_Msii, const int3 periodic,
		FD_TYPE* dev_field_loc, FD_TYPE* dev_energy_loc,
		FD_TYPE* dev_dot, const bool outputH, const bool outputE,
    const bool accumFlag) {
	
	extern __shared__ FD_TYPE s_MValue[  ];
	
	// identify the position in glb
	uint3 pos; //position of current position
	pos.x = blockIdx.x * blockDim.x + threadIdx.x;
	pos.y = blockIdx.y * blockDim.y + threadIdx.y;
	pos.z = blockIdx.z * blockDim.z + threadIdx.z;
	
	if ( pos.x >= dev_Dim.x || pos.y >= dev_Dim.y || pos.z >= dev_Dim.z)	{
    return;
  }
	
	unsigned int index =  pos.x +  (pos.y + pos.z * dev_Dim.y)*dev_Dim.x;
	
	const unsigned int s_index  = (threadIdx.x + 1) + ((threadIdx.y + 1)
		+ (threadIdx.z + 1)*(blockDim.y + 2)) * (blockDim.x + 2);
	const unsigned int s_index_zp = s_index + (blockDim.x + 2)*(blockDim.y + 2);
	const unsigned int s_index_zm = s_index - (blockDim.x + 2)*(blockDim.y + 2);
	const unsigned int s_index_yp = s_index + (blockDim.x + 2);
	const unsigned int s_index_ym = s_index - (blockDim.x + 2);
	const unsigned int s_index_xp = s_index + 1;
	const unsigned int s_index_xm = s_index - 1;

	unsigned int Dsize = dev_Dim.x * dev_Dim.y * dev_Dim.z;
	const unsigned int Ssize = (blockDim.x + 2) * (blockDim.y + 2) *
    (blockDim.z + 2);
	//load glb memory to shared memory
	
	glb2share(dev_MValue, s_MValue, index, s_index, Dsize, Ssize);

	//x--
	if(pos.x == 0){
		if(periodic.x) {
			glb2share( dev_MValue, s_MValue, index+dev_Dim.x-1, s_index_xm, Dsize, Ssize);
		} else {
			glb2share( dev_MValue, s_MValue, index, s_index_xm, Dsize, Ssize);
    }
	} else if(threadIdx.x == 0) {
		glb2share( dev_MValue, s_MValue, index-1, s_index_xm, Dsize, Ssize);
  }
	//x++
	if(pos.x == dev_Dim.x-1) {
		if(periodic.x) {
			glb2share( dev_MValue, s_MValue, index-dev_Dim.x+1, s_index_xp, Dsize, Ssize);
		} else {
			glb2share( dev_MValue, s_MValue, index, s_index_xp, Dsize, Ssize);
    }
	} else if(threadIdx.x == blockDim.x-1) {
		glb2share( dev_MValue, s_MValue, index+1, s_index_xp, Dsize, Ssize);
  }
  
	//y--
	if(pos.y == 0){
		if(periodic.y) {
			glb2share( dev_MValue, s_MValue, index+dev_Dim.x*(dev_Dim.y-1), s_index_ym, Dsize, Ssize);
		} else {
			glb2share( dev_MValue, s_MValue, index, s_index_ym, Dsize, Ssize);
    }
	} else if(threadIdx.y == 0) {
		glb2share( dev_MValue, s_MValue, index-dev_Dim.x, s_index_ym, Dsize, Ssize);
  }
  
	//y++
	if(pos.y == dev_Dim.y-1){
		if(periodic.y) {
			glb2share( dev_MValue, s_MValue, index-dev_Dim.x*(dev_Dim.y-1), s_index_yp, Dsize, Ssize);
		} else {
			glb2share( dev_MValue, s_MValue, index, s_index_yp, Dsize, Ssize);
    }
	} else if(threadIdx.y == blockDim.y-1) {
		glb2share( dev_MValue, s_MValue, index+dev_Dim.x, s_index_yp, Dsize, Ssize);
  }
  
	//z--
	if(pos.z == 0){
		if(periodic.z) {
			glb2share( dev_MValue, s_MValue, index+dev_Dim.x*dev_Dim.y*(dev_Dim.z-1), s_index_zm, Dsize, Ssize);
		} else {
			glb2share( dev_MValue, s_MValue, index, s_index_zm, Dsize, Ssize);
    }
	} else if(threadIdx.z == 0) {
		glb2share( dev_MValue, s_MValue, index-dev_Dim.y*dev_Dim.x, s_index_zm, Dsize, Ssize);
  }
  
	//z++
	if(pos.z == dev_Dim.z - 1) {
		if(periodic.z) {
			glb2share( dev_MValue, s_MValue, index-dev_Dim.x*dev_Dim.y*(dev_Dim.z-1), s_index_zp, Dsize, Ssize);
		} else {
			glb2share( dev_MValue, s_MValue, index, s_index_zp, Dsize, Ssize);
    }
	} else if(threadIdx.z == blockDim.z - 1) {
		glb2share( dev_MValue, s_MValue, index+dev_Dim.y*dev_Dim.x, s_index_zp, Dsize, Ssize);
  }
	
  __syncthreads();

	FD_TYPE3 Hex;
	Hex.x = (s_MValue[ s_index_xp] + s_MValue[s_index_xm] - 2*s_MValue[ s_index])*dev_wgt.x + (s_MValue[ s_index_yp] + s_MValue[ s_index_ym]- 2*s_MValue[ s_index])*dev_wgt.y + (s_MValue[ s_index_zp] + s_MValue[ s_index_zm]- 2*s_MValue[ s_index])*dev_wgt.z; 
	Hex.y = (s_MValue[ s_index_xp+Ssize] + s_MValue[s_index_xm+Ssize] - 2*s_MValue[ s_index+Ssize] )*dev_wgt.x + (s_MValue[ s_index_yp+Ssize] + s_MValue[ s_index_ym+Ssize] - 2*s_MValue[ s_index+Ssize] )*dev_wgt.y + (s_MValue[ s_index_zp+Ssize] + s_MValue[ s_index_zm+Ssize] - 2*s_MValue[ s_index+Ssize] )*dev_wgt.z; 
	Hex.z = (s_MValue[ s_index_xp+2*Ssize] + s_MValue[s_index_xm+2*Ssize] - 2*s_MValue[ s_index+2*Ssize] )*dev_wgt.x + (s_MValue[ s_index_yp+2*Ssize] + s_MValue[ s_index_ym+2*Ssize] - 2*s_MValue[ s_index+2*Ssize] )*dev_wgt.y + (s_MValue[ s_index_zp+2*Ssize] + s_MValue[ s_index_zm+2*Ssize] - 2*s_MValue[ s_index+2*Ssize] )*dev_wgt.z; 
	
	FD_TYPE tmp_energy = Hex.x*s_MValue[s_index] + Hex.y*s_MValue[s_index+Ssize] + Hex.z*s_MValue[s_index+2*Ssize];

  if (accumFlag) {
    dev_Energy[index] += tmp_energy;
  }
	Hex *= (-2 / MU0) * dev_Msii[index];

  if (accumFlag) {
    add_FD_TYPE3M(index, dev_Dim, dev_H, Hex.x, Hex.y, Hex.z);
  }
  
	FD_TYPE3 mxH; 
	mxH.x = s_MValue[s_index+Ssize]*Hex.z - s_MValue[s_index+2*Ssize]*Hex.y; 
	mxH.y = s_MValue[s_index+2*Ssize]*Hex.x - s_MValue[s_index]*Hex.z;
	mxH.z = s_MValue[s_index]*Hex.y - s_MValue[s_index+Ssize]*Hex.x;
  if (accumFlag) {
    add_FD_TYPE3M(index, dev_Dim, dev_Torque, mxH.x, mxH.y, mxH.z);
  }
	
	//The following is implemented for max_angle output
	//this part could be triggered by a flag upon request
	FD_TYPE3 tmp;
	tmp.x = s_MValue[s_index_xm] - s_MValue[ s_index]; 
	tmp.y = s_MValue[ s_index_xm+Ssize] - s_MValue[ s_index+Ssize];
	tmp.z = s_MValue[ s_index_xm+2*Ssize] - s_MValue[ s_index+2*Ssize];
	FD_TYPE dot_x = sqr_FD_TYPE3M(tmp);
	
	tmp.x = s_MValue[s_index_ym] - s_MValue[ s_index]; 
	tmp.y = s_MValue[ s_index_ym+Ssize] - s_MValue[ s_index+Ssize];
	tmp.z = s_MValue[ s_index_ym+2*Ssize] - s_MValue[ s_index+2*Ssize];
	FD_TYPE dot_y = sqr_FD_TYPE3M(tmp);
	
	tmp.x = s_MValue[s_index_zm] - s_MValue[ s_index]; 
	tmp.y = s_MValue[ s_index_zm+Ssize] - s_MValue[ s_index+Ssize];
	tmp.z = s_MValue[ s_index_zm+2*Ssize] - s_MValue[ s_index+2*Ssize];
	FD_TYPE dot_z = sqr_FD_TYPE3M(tmp);
				
	FD_TYPE dot = dot_x;
	if(dot_y > dot)	dot = dot_y;	if(dot_z > dot) dot = dot_z;
	dev_dot[index] = dot; 
  
  if (outputE) {
    dev_energy_loc[index] = tmp_energy;
  }

  if (outputH) {
    set_FD_TYPE3M(index, dev_Dim, dev_field_loc, Hex.x, Hex.y, Hex.z);
  }
}

void s_exchUniform(const dim3 &grid_size, const dim3 &block_size,
    const dim3 &cubic_block_size,
    const FD_TYPE* dev_MValue, FD_TYPE* dev_H, FD_TYPE* dev_Energy, 
    FD_TYPE* dev_Torque, const uint3 dev_Dim, const FD_TYPE3 dev_wgt, 
    const FD_TYPE* dev_Msii, const int3 periodic, FD_TYPE* dev_field_loc, 
    FD_TYPE* dev_energy_loc, FD_TYPE* dev_dot, const bool outputH, 
    const bool outputE, const bool accumFlag) {
      
  OC_INDEX shareMemSize = ODTV_VECSIZE * (cubic_block_size.x + 2) * 
    (cubic_block_size.y + 2) * (cubic_block_size.z + 2) * sizeof(FD_TYPE);
    
  s_exchUniform_kernel<<<grid_size, block_size, shareMemSize>>>
    (dev_MValue, dev_H, dev_Energy, dev_Torque, dev_Dim, 
    dev_wgt, dev_Msii, periodic, dev_field_loc, dev_energy_loc,
    dev_dot, outputH, outputE, accumFlag);
}

string getExchKernelName() {
  return "s_exchUniform_kernel";
}