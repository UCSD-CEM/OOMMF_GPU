#ifndef _GPU_HELPER_H
#define _GPU_HELPER_H

#include <string>
#include "oc.h"

#include <cuda_runtime_api.h>
#include <vector_types.h>
#include "GPU_devstruct.h"

template <class T>
string memDownload_device(T *_h_des, 
    const T *_d_src, const int &_num_elements, 
    const int &dev_index);

template <class T> 
string memUpload_device(T *_d_des, 
    const T *_h_src, const int &_num_elements, 
    const int &dev_index);
    
template <class T> 
string alloc_device(T* &array_for_alloc, 
  const OC_INDEX &num_elements, 
  const OC_INDEX &dev, const string &array_name);

template <class T>
string release_device(T* &array_for_release,
    const OC_INDEX &dev, const string &array_name);

template <class T>      
string memPurge_device(T *_d_ptr, 
    const size_t &_num_elements, const int dev_index);

template <class T>
string memDownload_toFile(const T *_output, const OC_INDEX _n, 
    std::string filename = "", const OC_INDEX _dim = 1, 
    const OC_INDEX mode = 0);
    
string checkError();

string fetchInfo_device(int &maxGridSize,
    FD_TYPE &maxTotalThreads, const int &dev_num);

void getFlatKernelSize(const unsigned int num_threads, 
  const unsigned int &set_blk_size, 
  dim3 &gridsize, dim3 &blocksize);

void dotProduct(const unsigned int &size,
  const unsigned int &set_blk_size,
  const FD_TYPE *d_idata1, const FD_TYPE *d_idata2, 
  FD_TYPE *d_odata);
    
FD_TYPE sum_device(int size, FD_TYPE *d_idata, 
  FD_TYPE *d_odata, const int &dev_index,
  const int &maxGridSize,
  const FD_TYPE &maxTotalThreads);

FD_TYPE maxDot(FD_TYPE *dev_iData, FD_TYPE *dev_buffer, const OC_INDEX &size,
  const unsigned int blockSize, const OC_INDEX &dev_index);  
#endif