#include <fstream>
#include "cufft.h"
#include "GPU_helper.h"

string reportCUDAError(const cudaError_t &cuda_error_code) {
  if (cuda_error_code != cudaSuccess) {
    return string(cudaGetErrorString(cuda_error_code));
  }
  return string("No CUDA error");
}

// alloc_device
template <class T> 
string alloc_device(T* &array_for_alloc, 
  const OC_INDEX &num_elements, 
  const OC_INDEX &dev, const string &array_name) {
  
  if (array_for_alloc != NULL) {
    return array_name + string(" already allocated");
  }
  // normal allocation
  cudaSetDevice(dev);
  cudaMalloc((void**)&array_for_alloc, sizeof(T) * num_elements);
  return array_name + string(": ") + checkError();
}

template string alloc_device<FD_TYPE>(
  FD_TYPE* &array_for_alloc, 
  const OC_INDEX &num_elements, 
  const OC_INDEX &dev, const string &array_name);
template string alloc_device<DEVSTRUCT>(
  DEVSTRUCT* &array_for_alloc, 
  const OC_INDEX &num_elements, 
  const OC_INDEX &dev, const string &array_name);
template string alloc_device<FD_CPLX_TYPE>(
  FD_CPLX_TYPE* &array_for_alloc, 
  const OC_INDEX &num_elements, 
  const OC_INDEX &dev, const string &array_name);
template<> string alloc_device<void>(
  void* &array_for_alloc, 
  const OC_INDEX &num_elements, 
  const OC_INDEX &dev, const string &array_name) {
    
  if (array_for_alloc != NULL) {
    return array_name + string(" already allocated");
  }
  // normal allocation
  cudaSetDevice(dev);
  cudaMalloc(&array_for_alloc, num_elements);
  return array_name + string(": ") + checkError();
}


// release_device
template <class T>
string release_device(T* &array_for_release,
    const OC_INDEX &dev, const string &array_name) {

  if (array_for_release == NULL) {
    return array_name + string(" already deallocated");
  }
	
  cudaSetDevice(dev);
	cudaFree(array_for_release);
  array_for_release = NULL;
  return array_name + string(": ") + checkError();
}

template string release_device<FD_TYPE>(
    FD_TYPE* &array_for_release,
    const OC_INDEX &dev, const string &array_name);
template string release_device<DEVSTRUCT>(
    DEVSTRUCT* &array_for_release,
    const OC_INDEX &dev, const string &array_name);
template string release_device<FD_CPLX_TYPE>(
    FD_CPLX_TYPE* &array_for_release,
    const OC_INDEX &dev, const string &array_name);
template string release_device<void>(
    void* &array_for_release,
    const OC_INDEX &dev, const string &array_name);
    
// memDownload  
template <class T>
string memDownload_device(T *_h_des, 
    const T *_d_src, const int &_num_elements, 
    const int &dev_index) {
    
  return reportCUDAError(cudaMemcpy(
    _h_des, _d_src, sizeof(T) * _num_elements, 
    cudaMemcpyDeviceToHost));
}

template string memDownload_device<FD_TYPE>(
  FD_TYPE *_h_des, const FD_TYPE *_d_src,
  const int &_num_elements, const int &dev_index);
    
// memUpload
template <class T> 
string memUpload_device(T *_d_des, 
    const T *_h_src, const int &_num_elements, 
    const int &dev_index) {
    
  cudaSetDevice(dev_index);
  return reportCUDAError(cudaMemcpy(
    _d_des, _h_src, sizeof(T) * _num_elements, 
    cudaMemcpyHostToDevice));
}

template string memUpload_device<FD_TYPE>(
    FD_TYPE *_d_des, const FD_TYPE *_h_src, 
    const int &_num_elements, const int &dev_index);
template string memUpload_device<DEVSTRUCT>(
    DEVSTRUCT *_d_des, const DEVSTRUCT *_h_src, 
    const int &_num_elements, const int &dev_index);

// memPurge
template <class T>     
string memPurge_device(T *_d_ptr, 
    const size_t &_num_elements, const int dev_index) {
    
  // cudaSetDevice(dev_index);	
  return reportCUDAError(cudaMemset(
    _d_ptr, 0, sizeof(T) * _num_elements));
}

template string memPurge_device<FD_TYPE>(FD_TYPE *_d_ptr, 
    const size_t &_num_elements, const int dev_index);
template <> string memPurge_device<void>(void *_d_ptr, 
    const size_t &_num_elements, const int dev_index) {
  return reportCUDAError(cudaMemset( _d_ptr, 0, _num_elements));    
}

std::ostream& operator<<(std::ostream& os, const FD_CPLX_TYPE& obj)
{
  return os << obj.x << " " << obj.y;
}

// print device array to file, for debug purpose only
template <class T>
string memDownload_toFile(const T *_output, const OC_INDEX _n, 
    std::string filename, const OC_INDEX _dim, 
    const OC_INDEX mode) {
      
  if (filename == "") {
    filename = std::string("TestArrayDevice.txt");
  } else {
    filename += std::string(".txt");
  }
  std::ofstream out(filename.c_str(), std::ofstream::out);

  T* _host_image = new T[ _dim * _n];
  
  const string result = memDownload_device(_host_image, _output, _n * _dim, 0);
  
  for (OC_INDEX i = 0; i < _n; i++ ){
    for(OC_INDEX j = 0; j < _dim; j++ ){
      if( mode == 0 ) {
        out << _host_image[_dim * i + j] << " \t";
      } else if( mode == 1 ) {
        out << _host_image[i + j * _n] << " \t";
      }
    }
    out << std::endl;
  }
  out.close();

  delete[] _host_image;
  return result;
}

template string memDownload_toFile<FD_TYPE>(const FD_TYPE *_output, 
  const OC_INDEX _n, std::string filename, const OC_INDEX _dim, 
  const OC_INDEX mode);
template string memDownload_toFile<FD_CPLX_TYPE>(const FD_CPLX_TYPE *_output, 
    const OC_INDEX _n, std::string filename, const OC_INDEX _dim, 
    const OC_INDEX mode);
    
// check CUDA error
string checkError() {
  return reportCUDAError(cudaGetLastError());
}

void getFlatKernelSize(const unsigned int num_threads, 
  const unsigned int &set_blk_size, dim3 &gridsize, dim3 &blocksize) {
  
   blocksize.x = set_blk_size;
   int num_blk = (num_threads-1)/set_blk_size + 1;

   gridsize.y = (num_blk - 1) / 65535 + 1;
   gridsize.x = (num_blk - 1) / gridsize.y + 1;
}

string fetchInfo_device(int &maxGridSize,
    FD_TYPE &maxTotalThreads, const int &dev_num) {
    
  cudaDeviceProp prop;
  const string str = reportCUDAError(
    cudaGetDeviceProperties(&prop, dev_num));
  maxGridSize = prop.maxGridSize[0];
  maxTotalThreads = 
    (FD_TYPE)prop.maxGridSize[0] * prop.maxThreadsPerBlock;
  return str;
}

__global__ void dotProductKernel(const unsigned int size,
    const FD_TYPE *d_idata1, const FD_TYPE *d_idata2, 
    FD_TYPE *d_odata) {
  
  int i = threadIdx.x + (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x;
  
  if(i >= size) {
    return;
  }
  
  d_odata[i] = d_idata1[i] * d_idata2[i];
}

// wrapper for dotProduct kernel
void dotProduct(const unsigned int &size,
    const unsigned int &set_blk_size,
    const FD_TYPE *d_idata1, const FD_TYPE *d_idata2, 
    FD_TYPE *d_odata) {
  
  dim3 gridSize;
  dim3 blockSize;
  getFlatKernelSize(size, set_blk_size, gridSize, blockSize);
  dotProductKernel<<<gridSize, blockSize>>>(size,
    d_idata1, d_idata2, d_odata);
}

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double> {
    __device__ inline operator       double *()
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};

/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
reduce6(T *g_idata, T *g_odata, unsigned int n) {
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    T mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            mySum += g_idata[i+blockSize];

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if ((blockSize >= 512) && (tid < 256))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 256];
    }

    __syncthreads();

    if ((blockSize >= 256) &&(tid < 128))
    {
            sdata[tid] = mySum = mySum + sdata[tid + 128];
    }

     __syncthreads();

    if ((blockSize >= 128) && (tid <  64))
    {
       sdata[tid] = mySum = mySum + sdata[tid +  64];
    }

    __syncthreads();

#if (__CUDA_ARCH__ >= 300 )
    if ( tid < 32 )
    {
        // Fetch final intermediate sum from 2nd warp
        if (blockSize >=  64) mySum += sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2) 
        {
            mySum += __shfl_down(mySum, offset);
        }
    }
#else
    // fully unroll reduction within a single warp
    if ((blockSize >=  64) && (tid < 32))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 32];
    }

    __syncthreads();

    if ((blockSize >=  32) && (tid < 16))
    {
        sdata[tid] = mySum = mySum + sdata[tid + 16];
    }

    __syncthreads();

    if ((blockSize >=  16) && (tid <  8))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  8];
    }

    __syncthreads();

    if ((blockSize >=   8) && (tid <  4))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  4];
    }

    __syncthreads();

    if ((blockSize >=   4) && (tid <  2))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  2];
    }

    __syncthreads();

    if ((blockSize >=   2) && ( tid <  1))
    {
        sdata[tid] = mySum = mySum + sdata[tid +  1];
    }

    __syncthreads();
#endif

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}

extern "C"
bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}

unsigned int nextPow2(unsigned int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T>
void
reduceSelect(int size, int threads, int blocks, T *d_idata, T *d_odata) {

    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    // choose which of the optimized versions of reduction to launch
    if (isPow2(size)) {
      switch (threads) {
        case 512:
            reduce6<T, 512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;

        case 256:
            reduce6<T, 256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;

        case 128:
            reduce6<T, 128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;

        case 64:
            reduce6<T,  64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;

        case 32:
            reduce6<T,  32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;

        case 16:
            reduce6<T,  16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;

        case  8:
            reduce6<T,   8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;

        case  4:
            reduce6<T,   4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;

        case  2:
            reduce6<T,   2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;

        case  1:
            reduce6<T,   1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;
      }
    } else {
      switch (threads) {
        case 512:
            reduce6<T, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;

        case 256:
            reduce6<T, 256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;

        case 128:
            reduce6<T, 128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;

        case 64:
            reduce6<T,  64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;

        case 32:
            reduce6<T,  32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;

        case 16:
            reduce6<T,  16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;

        case  8:
            reduce6<T,   8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;

        case  4:
            reduce6<T,   4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;

        case  2:
            reduce6<T,   2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;

        case  1:
            reduce6<T,   1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
            break;
      }
    }
}

template void
reduceSelect<FD_TYPE>(int size, int threads, int blocks,
               FD_TYPE *d_idata, FD_TYPE *d_odata);

void getNumBlocksAndThreads(const int n, 
  const int maxBlocks, const int maxThreads, int &blocks, 
  int &threads, const int &maxGridSize, 
  const FD_TYPE &maxTotalThreads) {

    //get device capability, to avoid block/grid size excceed the upbound
    threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
    blocks = (n + (threads * 2 - 1)) / (threads * 2);

    if ((float)threads*blocks > maxTotalThreads) {
      printf("n is too large, please choose a smaller number!\n");
    }

    if (blocks > maxGridSize) {
      printf("Grid size <%d> excceeds the device capability <%d>, set block size as %d (original %d)\n",
             blocks, maxGridSize, threads*2, threads);

      blocks /= 2;
      threads *= 2;
    }

    blocks = std::min(maxBlocks, blocks);
}
               
FD_TYPE sum_device(int size, FD_TYPE *d_idata, 
  FD_TYPE *d_odata, const int &dev_index, 
  const int &maxGridSize, 
  const FD_TYPE &maxTotalThreads) {
 
  int numBlocks = 0;
  int numThreads = 0;
  int maxThreads = 512;  // number of threads per block
  int maxBlocks = 64;
  const int cpuFinalThreshold = 1;
  getNumBlocksAndThreads(size, maxBlocks, 
    maxThreads, numBlocks, numThreads, maxGridSize, 
    maxTotalThreads);

  // execute the kernel
  reduceSelect<FD_TYPE>(size, numThreads, numBlocks,
    d_idata, d_odata);
    
  // sum partial block sums on GPU
  while (numBlocks > cpuFinalThreshold) {
    int threads = 0, blocks = 0;
    getNumBlocksAndThreads(numBlocks, maxBlocks, 
      maxThreads, blocks, threads, maxGridSize, 
      maxTotalThreads);

    reduceSelect<FD_TYPE>(numBlocks, threads, blocks, 
      d_odata, d_odata);

    numBlocks = (numBlocks + (threads * 2 - 1)) / (threads * 2);
  }

  // copy final sum from device to host
  FD_TYPE result;
  memDownload_device(&result, d_odata, 1, dev_index);
  return result;
}

// Max kernel

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
__global__ void collect_dot_kernel(unsigned int size, FD_TYPE *dev_dot,
	FD_TYPE *dev_dot_max) {
	
	__shared__ FD_TYPE s_max[blockSize];
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x+blockIdx.y*gridDim.x;
	unsigned int gid = bid*blockSize + tid;
	unsigned int gridSize = blockSize*gridDim.x*gridDim.y*gridDim.z;
	
	//initialize the shared memory data
	s_max[tid] = 0.f;
	FD_TYPE tmp_magn;
	while (gid < size) {
		tmp_magn = dev_dot[gid];
		if( s_max[tid] < tmp_magn)	s_max[tid] = tmp_magn;
		gid += gridSize; 
	}
	__syncthreads();
	
	//reduction in shared memory
	if (blockSize >= 1024) { 
		if (tid < 512){
			if(s_max[tid] < s_max[tid + 512]) s_max[tid] = s_max[tid + 512]; 
		} 
		__syncthreads();
	}
	if (blockSize >= 512) { 
		if (tid < 256){
			if(s_max[tid] < s_max[tid + 256]) s_max[tid] = s_max[tid + 256]; 
		} 
		__syncthreads();
	}
	if (blockSize >= 256) { 
		if (tid < 128){
			if(s_max[tid] < s_max[tid + 128]) s_max[tid] = s_max[tid + 128]; 
		} 
		__syncthreads();
	}
	if (blockSize >= 128) { 
		if (tid < 64){
			if(s_max[tid] < s_max[tid + 64]) s_max[tid] = s_max[tid + 64]; 
		} 
		__syncthreads();
	}
	if (tid < 32){
		warpMax<blockSize>(s_max, tid);
	}
	if (tid == 0)
		dev_dot_max[bid] = s_max[0];
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

template __device__ void warpMax<1024>(volatile FD_TYPE *sdata, unsigned int tid);
template __device__ void warpMax<512>(volatile FD_TYPE *sdata, unsigned int tid);
template __device__ void warpMax<256>(volatile FD_TYPE *sdata, unsigned int tid);
template __device__ void warpMax<128>(volatile FD_TYPE *sdata, unsigned int tid);
template __device__ void warpMax<64>(volatile FD_TYPE *sdata, unsigned int tid);
template __device__ void warpMax<32>(volatile FD_TYPE *sdata, unsigned int tid);
template __device__ void warpMax<16>(volatile FD_TYPE *sdata, unsigned int tid);
template __device__ void warpMax<8>(volatile FD_TYPE *sdata, unsigned int tid);
template __device__ void warpMax<4>(volatile FD_TYPE *sdata, unsigned int tid);
template __device__ void warpMax<2>(volatile FD_TYPE *sdata, unsigned int tid);
template __device__ void warpMax<1>(volatile FD_TYPE *sdata, unsigned int tid);

template __global__ void collect_dot_kernel<1024>(unsigned int size, FD_TYPE *dev_dot, FD_TYPE *dev_dot_max);
template __global__ void collect_dot_kernel<512>(unsigned int size, FD_TYPE *dev_dot, FD_TYPE *dev_dot_max);
template __global__ void collect_dot_kernel<256>(unsigned int size, FD_TYPE *dev_dot, FD_TYPE *dev_dot_max);
template __global__ void collect_dot_kernel<128>(unsigned int size, FD_TYPE *dev_dot, FD_TYPE *dev_dot_max);
template __global__ void collect_dot_kernel<64>(unsigned int size, FD_TYPE *dev_dot, FD_TYPE *dev_dot_max);
template __global__ void collect_dot_kernel<32>(unsigned int size, FD_TYPE *dev_dot, FD_TYPE *dev_dot_max);
template __global__ void collect_dot_kernel<16>(unsigned int size, FD_TYPE *dev_dot, FD_TYPE *dev_dot_max);
template __global__ void collect_dot_kernel<8>(unsigned int size, FD_TYPE *dev_dot, FD_TYPE *dev_dot_max);
template __global__ void collect_dot_kernel<4>(unsigned int size, FD_TYPE *dev_dot, FD_TYPE *dev_dot_max);
template __global__ void collect_dot_kernel<2>(unsigned int size, FD_TYPE *dev_dot, FD_TYPE *dev_dot_max);
template __global__ void collect_dot_kernel<1>(unsigned int size, FD_TYPE *dev_dot, FD_TYPE *dev_dot_max);

template __global__ void final_max_kernel<1024>(unsigned int size, FD_TYPE *g_idata, FD_TYPE *g_odata);
template __global__ void final_max_kernel<512>(unsigned int size, FD_TYPE *g_idata, FD_TYPE *g_odata);
template __global__ void final_max_kernel<256>(unsigned int size, FD_TYPE *g_idata, FD_TYPE *g_odata);
template __global__ void final_max_kernel<128>(unsigned int size, FD_TYPE *g_idata, FD_TYPE *g_odata);
template __global__ void final_max_kernel<64>(unsigned int size, FD_TYPE *g_idata, FD_TYPE *g_odata);
template __global__ void final_max_kernel<32>(unsigned int size, FD_TYPE *g_idata, FD_TYPE *g_odata);
template __global__ void final_max_kernel<16>(unsigned int size, FD_TYPE *g_idata, FD_TYPE *g_odata);
template __global__ void final_max_kernel<8>(unsigned int size, FD_TYPE *g_idata, FD_TYPE *g_odata);
template __global__ void final_max_kernel<4>(unsigned int size, FD_TYPE *g_idata, FD_TYPE *g_odata);
template __global__ void final_max_kernel<2>(unsigned int size, FD_TYPE *g_idata, FD_TYPE *g_odata);
template __global__ void final_max_kernel<1>(unsigned int size, FD_TYPE *g_idata, FD_TYPE *g_odata);

FD_TYPE maxDot(FD_TYPE *dev_iData, FD_TYPE *dev_buffer, const OC_INDEX &size,
  const unsigned int blockSize, const OC_INDEX &dev_index) {

  dim3 grid_size, block_size;
  getFlatKernelSize(size, blockSize, grid_size, block_size);
  unsigned int reduce_size = grid_size.x * grid_size.y * grid_size.z; 
  switch(blockSize) {
    case 1024:
      collect_dot_kernel<1024><<< grid_size, block_size>>>(size, dev_iData, dev_buffer);
      final_max_kernel<1024><<<1, block_size>>>(reduce_size, dev_buffer, dev_iData);
    break;
    case 512:
      collect_dot_kernel<512><<< grid_size, block_size>>>(size, dev_iData, dev_buffer);
      final_max_kernel<512><<<1, block_size>>>(reduce_size, dev_buffer, dev_iData);
    break;
    case 256:
      collect_dot_kernel<256><<< grid_size, block_size>>>(size, dev_iData, dev_buffer);
      final_max_kernel<256><<<1, block_size>>>(reduce_size, dev_buffer, dev_iData);
    break;
    case 128:
      collect_dot_kernel<128><<< grid_size, block_size>>>(size, dev_iData, dev_buffer);
      final_max_kernel<128><<<1, block_size>>>(reduce_size, dev_buffer, dev_iData);
    break;
    case 64:
      collect_dot_kernel<64><<< grid_size, block_size>>>(size, dev_iData, dev_buffer);
      final_max_kernel<64><<<1, block_size>>>(reduce_size, dev_buffer, dev_iData);
    break;
    case 32:
      collect_dot_kernel<32><<< grid_size, block_size>>>(size, dev_iData, dev_buffer);
      final_max_kernel<32><<<1, block_size>>>(reduce_size, dev_buffer, dev_iData);
    break;
    case 16:
      collect_dot_kernel<16><<< grid_size, block_size>>>(size, dev_iData, dev_buffer);
      final_max_kernel<16><<<1, block_size>>>(reduce_size, dev_buffer, dev_iData);
    break;
    case 8:
      collect_dot_kernel<8><<< grid_size, block_size>>>(size, dev_iData, dev_buffer);
      final_max_kernel<8><<<1, block_size>>>(reduce_size, dev_buffer, dev_iData);
    break;
    case 4:
      collect_dot_kernel<4><<< grid_size, block_size>>>(size, dev_iData, dev_buffer);
      final_max_kernel<4><<<1, block_size>>>(reduce_size, dev_buffer, dev_iData);
    break;
    case 2:
      collect_dot_kernel<2><<< grid_size, block_size>>>(size, dev_iData, dev_buffer);
      final_max_kernel<2><<<1, block_size>>>(reduce_size, dev_buffer, dev_iData);
    break;
    case 1:
      collect_dot_kernel<1><<< grid_size, block_size>>>(size, dev_iData, dev_buffer);
      final_max_kernel<1><<<1, block_size>>>(reduce_size, dev_buffer, dev_iData);
  }
  FD_TYPE maxdot;
  memDownload_device(&maxdot, dev_iData, 1, dev_index);
  return maxdot;
}