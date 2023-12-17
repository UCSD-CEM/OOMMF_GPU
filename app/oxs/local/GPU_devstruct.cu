#include "oc.h"
#include "mesh.h"
#include "rectangularmesh.h"
#include "fft3v.h"

#include "cufft.h"

#include "GPU_helper.h"

OC_INDEX getLocalSumSize(const OC_INDEX &size) {
  dim3 blockSize, gridSize;
  getFlatKernelSize(size, BLK_SIZE, gridSize, blockSize);
  return std::max(
    (unsigned int)(gridSize.x * gridSize.y * gridSize.z * ODTV_VECSIZE),
    (unsigned int)(size));
}

std::vector<int> getFFTSize(const Oxs_CommonRectangularMesh* mesh) {
  const OC_INDEX rdimx = mesh->DimX();
  const OC_INDEX rdimy = mesh->DimY();
  const OC_INDEX rdimz = mesh->DimZ();
  OC_INDEX cdimx;
  OC_INDEX cdimy;
  OC_INDEX cdimz;
  Oxs_FFT3DThreeVector::RecommendDimensions((rdimx==1 ? 1 : 2*rdimx),
                                            (rdimy==1 ? 1 : 2*rdimy),
                                            (rdimz==1 ? 1 : 2*rdimz),
                                            cdimx,cdimy,cdimz);
  //********This is where demam.cc allocate temporary data
  //****Mtemp = new OXS_FFT_REAL_TYPE[ODTV_VECSIZE*rdimx*rdimy*rdimz];
  /// Temporary space to hold Ms[]*m[].  The plan is to make this space
  /// unnecessary by introducing FFT functions that can take Ms as input
  /// and do the multiplication on the fly.


  // The following 3 statements are cribbed from
  // Oxs_FFT3DThreeVector::SetDimensions().  The corresponding
  // code using that class is
  //
  //  Oxs_FFT3DThreeVector fft;
  //  fft.SetDimensions(rdimx,rdimy,rdimz,cdimx,cdimy,cdimz);
  //  fft.GetLogicalDimensions(ldimx,ldimy,ldimz);
  //
  //*******CAREFUL, THIS MAY BE NECESSARY****************************
  Oxs_FFT1DThreeVector fftx;
  Oxs_FFTStrided ffty;
  Oxs_FFTStrided fftz;
  fftx.SetDimensions(rdimx, (cdimx==1 ? 1 : 2*(cdimx-1)), rdimy);
  ffty.SetDimensions(rdimy, cdimy,
                     ODTV_COMPLEXSIZE*ODTV_VECSIZE*cdimx,
                     ODTV_VECSIZE*cdimx);
  fftz.SetDimensions(rdimz, cdimz,
                     ODTV_COMPLEXSIZE*ODTV_VECSIZE*cdimx*cdimy,
                     ODTV_VECSIZE*cdimx*cdimy);
  // The following 3 statements are cribbed from
  // Oxs_FFT3DThreeVector::GetLogicalDimensions()
  std::vector<int> result(3);
  result[0] = fftx.GetLogicalDimension();
  result[1] = ffty.GetLogicalDimension();
  result[2] = fftz.GetLogicalDimension();
  return result;
}

OC_INDEX getCUFFTWorkAreaSize(const Oxs_CommonRectangularMesh* mesh) {
  
  cufftHandle plan_fwd;
  cufftHandle plan_bwd;
  
  if (cufftCreate(&plan_fwd) != CUFFT_SUCCESS) {
    string msg("error when building create plan_fwd on GPU,  try to reduce problem size...\n");
    // throw Oxs_ExtError(this, msg.c_str());  
  }
  
  if (cufftCreate(&plan_bwd) != CUFFT_SUCCESS) {
    string msg("error when building create plan_bwd on GPU,  try to reduce problem size...\n");
    // throw Oxs_ExtError(this, msg.c_str());  
  }
  
  if (cufftSetCompatibilityMode(plan_fwd, CUFFT_COMPATIBILITY_NATIVE) != CUFFT_SUCCESS) {
    string msg("error when cufftSetCompatibilityMode plan_fwd on GPU,  try to reduce problem size...\n");
    // throw Oxs_ExtError(this, msg.c_str());  
  }
  
  if (cufftSetCompatibilityMode(plan_bwd, CUFFT_COMPATIBILITY_NATIVE) != CUFFT_SUCCESS) {
    string msg("error when cufftSetCompatibilityMode plan_bwd on GPU,  try to reduce problem size...\n");
    // throw Oxs_ExtError(this, msg.c_str());  
  }
  
  if (cufftSetAutoAllocation(plan_fwd, 0) != CUFFT_SUCCESS) {
    string msg("error when cufftSetAutoAllocation plan_fwd on GPU\n");
    // throw Oxs_ExtError(this, msg.c_str());  
  }
  
  if (cufftSetAutoAllocation(plan_bwd, 0) != CUFFT_SUCCESS) {
    string msg("error when cufftSetAutoAllocation plan_bwd on GPU\n");
    // throw Oxs_ExtError(this, msg.c_str());  
  }
  
  size_t workSize_fwd;
  size_t workSize_bwd;
  std::vector<int> fftSize = getFFTSize(mesh);
  int fftSize_inverse[3] = {fftSize[2], fftSize[1], fftSize[0]};
  if (cufftGetSizeMany(plan_fwd, 3, fftSize_inverse, NULL, 0, 0, NULL, 0, 0, FWDFFT_METHOD, 3, &workSize_fwd) != CUFFT_SUCCESS) {
    string msg("error when cufftGetSizeMany plan_fwd on GPU\n");
    // throw Oxs_ExtError(this, msg.c_str());  
  }
  
  if (cufftGetSizeMany(plan_bwd, 3, fftSize_inverse, NULL, 0, 0, NULL, 0, 0, BWDFFT_METHOD, 3, &workSize_bwd) != CUFFT_SUCCESS) {
    string msg("error when cufftGetSizeMany plan_bwd on GPU\n");
    // throw Oxs_ExtError(this, msg.c_str());  
  }
  
  cufftResult_t cufftResult = cufftDestroy(plan_fwd);
  if (cufftResult != CUFFT_SUCCESS) {
    FILE *locate = fopen ("location.txt","a");
    fprintf(locate,"cufft error after cufftDestroy(plan_fwd) in dev_struct.cu, error code = %d\n",
      cufftResult); /**/
    fclose (locate);
  };
  
  cufftResult = cufftDestroy(plan_bwd);
  if (cufftResult != CUFFT_SUCCESS) {
    FILE *locate = fopen ("location.txt","a");
    fprintf(locate,"cufft error after cufftDestroy(plan_bwd) in dev_struct.cu, error code = %d\n",
      cufftResult); /**/
    fclose (locate);
  };
  
  return std::max(workSize_fwd, workSize_bwd);
}

OC_INDEX getWorkAreaSize(const Oxs_CommonRectangularMesh* mesh) {
  const OC_INDEX meshSize = mesh->DimX() * mesh->DimY() * mesh->DimZ();
  const OC_INDEX cufftSizeInByte = getCUFFTWorkAreaSize(mesh);
  
  const OC_INDEX size = mesh->DimX() * mesh->DimY() * mesh->DimZ();
  return std::max((unsigned long)((7 * meshSize + getLocalSumSize(size)) 
    * sizeof(FD_TYPE)), (unsigned long)(getCUFFTWorkAreaSize(mesh)));
}

void DEVSTRUCT::purgeAccumMem(const OC_INDEX &size) {
  std::string res = memPurge_device(dev_energy, size, DEV_NUM);
  if (res.compare(std::string("No CUDA error")) != 0) {
    FILE *location = fopen ("location.txt","a");
    fprintf(location, "memPurge_device problem %s at %s Ln %d\n", res.c_str(),
      __FILE__, __LINE__);
    fclose(location);
  };
  res = memPurge_device(dev_torque, ODTV_VECSIZE * size, DEV_NUM);
  if (res.compare(std::string("No CUDA error")) != 0) {
    FILE *location = fopen ("location.txt","a");
    fprintf(location, "memPurge_device problem %s at %s Ln %d\n", res.c_str(),
      __FILE__, __LINE__);
    fclose(location);
  };
  res = memPurge_device(dev_field, ODTV_VECSIZE * size, DEV_NUM);
  if (res.compare(std::string("No CUDA error")) != 0) {
    FILE *location = fopen ("location.txt","a");
    fprintf(location, "memPurge_device problem %s at %s Ln %d\n", res.c_str(),
      __FILE__, __LINE__);
    fclose(location);
  };
}

void DEVSTRUCT::purgeMem(const Oxs_Mesh* genmesh) {
  const Oxs_CommonRectangularMesh* mesh
    = dynamic_cast<const Oxs_CommonRectangularMesh*>(genmesh);
  const OC_INDEX size = mesh->DimX() * mesh->DimY() * mesh->DimZ();
  purgeAccumMem(size); 
  std::string res = memPurge_device(dev_Ms, size, DEV_NUM);
  if (res.compare(std::string("No CUDA error")) != 0) {
    FILE *location = fopen ("location.txt","a");
    fprintf(location, "memPurge_device problem %s at %s Ln %d\n", res.c_str(),
      __FILE__, __LINE__);
    fclose(location);
  };
  res = memPurge_device(dev_Msi, size, DEV_NUM);
  if (res.compare(std::string("No CUDA error")) != 0) {
    FILE *location = fopen ("location.txt","a");
    fprintf(location, "memPurge_device problem %s at %s Ln %d\n", res.c_str(),
      __FILE__, __LINE__);
    fclose(location);
  };
  res = memPurge_device(dev_MValue, ODTV_VECSIZE * size, DEV_NUM);
  if (res.compare(std::string("No CUDA error")) != 0) {
    FILE *location = fopen ("location.txt","a");
    fprintf(location, "memPurge_device problem %s at %s Ln %d\n", res.c_str(),
      __FILE__, __LINE__);
    fclose(location);
  };
  res = memPurge_device(dev_energy_bak, size, DEV_NUM);
  if (res.compare(std::string("No CUDA error")) != 0) {
    FILE *location = fopen ("location.txt","a");
    fprintf(location, "memPurge_device problem %s at %s Ln %d\n", res.c_str(),
      __FILE__, __LINE__);
    fclose(location);
  };
  res = memPurge_device(dev_vol, size, DEV_NUM);
  if (res.compare(std::string("No CUDA error")) != 0) {
    FILE *location = fopen ("location.txt","a");
    fprintf(location, "memPurge_device problem %s at %s Ln %d\n", res.c_str(),
      __FILE__, __LINE__);
    fclose(location);
  };
  res = memPurge_device(dev_dot, size, DEV_NUM);
  if (res.compare(std::string("No CUDA error")) != 0) {
    FILE *location = fopen ("location.txt","a");
    fprintf(location, "memPurge_device problem %s at %s Ln %d\n", res.c_str(),
      __FILE__, __LINE__);
    fclose(location);
  };
  res = memPurge_device(dev_dm_dt, ODTV_VECSIZE * size, DEV_NUM);
  if (res.compare(std::string("No CUDA error")) != 0) {
    FILE *location = fopen ("location.txt","a");
    fprintf(location, "memPurge_device problem %s at %s Ln %d\n", res.c_str(),
      __FILE__, __LINE__);
    fclose(location);
  };
  res = memPurge_device(dev_dm_dt_bak, ODTV_VECSIZE * size, DEV_NUM);
  if (res.compare(std::string("No CUDA error")) != 0) {
    FILE *location = fopen ("location.txt","a");
    fprintf(location, "memPurge_device problem %s at %s Ln %d\n", res.c_str(),
      __FILE__, __LINE__);
    fclose(location);
  };
  res = memPurge_device(dev_FFT_workArea, getWorkAreaSize(mesh), DEV_NUM);
  if (res.compare(std::string("No CUDA error")) != 0) {
    FILE *location = fopen ("location.txt","a");
    fprintf(location, "memPurge_device problem %s at %s Ln %d\n", res.c_str(),
      __FILE__, __LINE__);
    fclose(location);
  };
}

void DEVSTRUCT::allocMem(const Oxs_Mesh* genmesh) {
  const Oxs_CommonRectangularMesh* mesh
    = dynamic_cast<const Oxs_CommonRectangularMesh*>(genmesh);
  const OC_INDEX size = mesh->DimX() * mesh->DimY() * mesh->DimZ();
  std::string res;
  res = alloc_device(dev_Ms, size, DEV_NUM, "dev_Ms");
  if (res.find(std::string("No CUDA error")) == std::string::npos) {
    FILE *location = fopen ("location.txt","a");
    fprintf(location, "alloc_device problem %s at %s Ln %d\n", res.c_str(),
      __FILE__, __LINE__);
    fclose(location);
  };
  res = alloc_device(dev_Msi, size, DEV_NUM, "dev_Msi");
  if (res.find(std::string("No CUDA error")) == std::string::npos) {
    FILE *location = fopen ("location.txt","a");
    fprintf(location, "alloc_device problem %s at %s Ln %d\n", res.c_str(),
      __FILE__, __LINE__);
    fclose(location);
  };
  res = alloc_device(dev_MValue, ODTV_VECSIZE * size, DEV_NUM, "dev_MValue");
  if (res.find(std::string("No CUDA error")) == std::string::npos) {
    FILE *location = fopen ("location.txt","a");
    fprintf(location, "alloc_device problem %s at %s Ln %d\n", res.c_str(),
      __FILE__, __LINE__);
    fclose(location);
  };
  res = alloc_device(dev_energy, size, DEV_NUM, "dev_energy");
  if (res.find(std::string("No CUDA error")) == std::string::npos) {
    FILE *location = fopen ("location.txt","a");
    fprintf(location, "alloc_device problem %s at %s Ln %d\n", res.c_str(),
      __FILE__, __LINE__);
    fclose(location);
  };
  res = alloc_device(dev_field, ODTV_VECSIZE * size, DEV_NUM, "dev_field");
  if (res.find(std::string("No CUDA error")) == std::string::npos) {
    FILE *location = fopen ("location.txt","a");
    fprintf(location, "alloc_device problem %s at %s Ln %d\n", res.c_str(),
      __FILE__, __LINE__);
    fclose(location);
  };
  res = alloc_device(dev_torque, ODTV_VECSIZE * size, DEV_NUM, "dev_torque");
  if (res.find(std::string("No CUDA error")) == std::string::npos) {
    FILE *location = fopen ("location.txt","a");
    fprintf(location, "alloc_device problem %s at %s Ln %d\n", res.c_str(),
      __FILE__, __LINE__);
    fclose(location);
  };
  res = alloc_device(dev_energy_bak, size, DEV_NUM, "dev_energy_bak");
  if (res.find(std::string("No CUDA error")) == std::string::npos) {
    FILE *location = fopen ("location.txt","a");
    fprintf(location, "alloc_device problem %s at %s Ln %d\n", res.c_str(),
      __FILE__, __LINE__);
    fclose(location);
  };
  res = alloc_device(dev_vol, size, DEV_NUM, "dev_vol");
  if (res.find(std::string("No CUDA error")) == std::string::npos) {
    FILE *location = fopen ("location.txt","a");
    fprintf(location, "alloc_device problem %s at %s Ln %d\n", res.c_str(),
      __FILE__, __LINE__);
    fclose(location);
  };
  res = alloc_device(dev_dot, size, DEV_NUM, "dev_dot");
  if (res.find(std::string("No CUDA error")) == std::string::npos) {
    FILE *location = fopen ("location.txt","a");
    fprintf(location, "alloc_device problem %s at %s Ln %d\n", res.c_str(),
      __FILE__, __LINE__);
    fclose(location);
  };
  res = alloc_device(dev_dm_dt, ODTV_VECSIZE * size, DEV_NUM, "dev_dm_dt");
  if (res.find(std::string("No CUDA error")) == std::string::npos) {
    FILE *location = fopen ("location.txt","a");
    fprintf(location, "alloc_device problem %s at %s Ln %d\n", res.c_str(),
      __FILE__, __LINE__);
    fclose(location);
  };
  res = alloc_device(dev_dm_dt_bak, ODTV_VECSIZE * size, DEV_NUM, "dev_dm_dt_bak");
  if (res.find(std::string("No CUDA error")) == std::string::npos) {
    FILE *location = fopen ("location.txt","a");
    fprintf(location, "alloc_device problem %s at %s Ln %d\n", res.c_str(),
      __FILE__, __LINE__);
    fclose(location);
  };
  res = alloc_device(dev_FFT_workArea, getWorkAreaSize(mesh), DEV_NUM, "dev_FFT_workArea");
  if (res.find(std::string("No CUDA error")) == std::string::npos) {
    FILE *location = fopen ("location.txt","a");
    fprintf(location, "alloc_device problem %s at %s Ln %d\n", res.c_str(),
      __FILE__, __LINE__);
    fclose(location);
  };
  OC_INDEX memOffset = 0;
  dev_local_field = (FD_TYPE*)dev_FFT_workArea + memOffset;
  memOffset += ODTV_VECSIZE * size;
  dev_local_energy = (FD_TYPE*)dev_FFT_workArea + memOffset;
  memOffset += size;
  dev_local_sum = (FD_TYPE*)dev_FFT_workArea + memOffset;
}

void DEVSTRUCT::releaseMem() {
  release_device(dev_Ms, DEV_NUM, "dev_Ms");
  release_device(dev_Msi, DEV_NUM, "dev_Msi");
  release_device(dev_MValue, DEV_NUM, "dev_MValue");
  release_device(dev_energy, DEV_NUM, "dev_energy");
  release_device(dev_field, DEV_NUM, "dev_field");
  release_device(dev_torque, DEV_NUM, "dev_torque");
  release_device(dev_energy_bak, DEV_NUM, "dev_energy_bak");
  release_device(dev_vol, DEV_NUM, "dev_vol");
  release_device(dev_dot, DEV_NUM, "dev_dot");
  release_device(dev_dm_dt, DEV_NUM, "dev_dm_dt");
  release_device(dev_dm_dt_bak, DEV_NUM, "dev_dm_dt_bak");
  release_device(dev_FFT_workArea, DEV_NUM, "dev_FFT_workArea");
}

void DEVSTRUCT::reset() {
  dev_Ms = NULL;
  dev_Msi = NULL;
  dev_MValue = NULL;
  dev_energy = NULL;
  dev_field = NULL;
  dev_torque = NULL;
  dev_dm_dt = NULL;
  dev_energy_bak = NULL;
  dev_dm_dt_bak = NULL;
  dev_local_sum = NULL;
  dev_local_field = NULL;
  dev_local_energy = NULL;
  dev_vol = NULL;
  dev_dot = NULL;
  dev_FFT_workArea = NULL;
}