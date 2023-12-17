/* FILE: GPU_ExchUniform_new.cu           -*-Mode: cuda-*-
 *
 * Uniform exchange field, derived from Oxs_Energy class.
 *
 */

#include "oc.h"
#include "nb.h"
#include "threevector.h"
#include "energy.h"
#include "simstate.h"
#include "mesh.h"
#include "meshvalue.h"
#include "vectorfield.h"
#include "uniformscalarfield.h"
#include "uniformvectorfield.h"
#include "rectangularmesh.h"  // For QUAD-style integration
#include "GPU_ExchUniform_new.h"
#include "GPU_ExchUniform_new_kernel.h"
#include "energy.h"		// Needed to make MSVC++ 5 happy

// Oxs_Ext registration support
OXS_EXT_REGISTER(Oxs_GPU_UniformExchange_New);

/* End includes */

// Constructor
Oxs_GPU_UniformExchange_New::Oxs_GPU_UniformExchange_New(
  const char* name,     // Child instance id
  Oxs_Director* newdtr, // App director
  const char* argstr)   // MIF input block parameters
  : Oxs_GPU_Energy(name,newdtr,argstr),
  	excoeftype(A_UNKNOWN), A(-1.), lex(-1.),
    kernel(NGBR_UNKNOWN), 
    xperiodic(0),yperiodic(0),zperiodic(0),
    mesh_id(0),dev_MValue(0), dev_Msii(0), 
    dev_Field(0), dev_Energy(0), dev_Torque(0),
    tmp_field(0), tmp_energy(0), dev_field_loc(0), 
    dev_energy_loc(0), dev_dot(0),dev_dot_max(0),
    dev_final_dot_max(0) {
  // GPU supported
  _dev_num = DEV_NUM;
  cudaDeviceProp tmp_devProp;
   cudaGetDeviceProperties(&tmp_devProp, _dev_num);
   myBlk3DSize.x = 16;myBlk3DSize.y = 8;myBlk3DSize.z = 8;
   while(tmp_devProp.maxThreadsPerBlock < myBlk3DSize.x*
		myBlk3DSize.y*myBlk3DSize.z && myBlk3DSize.x>1){
			myBlk3DSize.x /= 2;
   }
   while(tmp_devProp.maxThreadsPerBlock < myBlk3DSize.x*
		myBlk3DSize.y*myBlk3DSize.z && myBlk3DSize.y>1){
			myBlk3DSize.y /= 2;
   }
   while(tmp_devProp.maxThreadsPerBlock < myBlk3DSize.x*
		myBlk3DSize.y*myBlk3DSize.z && myBlk3DSize.z>1){
			myBlk3DSize.z /= 2;
   }
   Knl_Blk_size.x = 0;Knl_Blk_size.y = 0;Knl_Blk_size.z = 0;
   Knl_Grid_size.x = 0;Knl_Grid_size.y = 0;Knl_Grid_size.z = 0;
   host_Periodic.x = host_Periodic.y = host_Periodic.z = 0;

  // Process arguments
  OC_BOOL has_A = HasInitValue("A");
  OC_BOOL has_lex = HasInitValue("lex");
  if(has_A && has_lex) {
    throw Oxs_ExtError(this,"Invalid exchange coefficient request:"
			 " both A and lex specified; only one should"
			 " be given.");
  } else if(has_lex) {
    lex = GetRealInitValue("lex");
    excoeftype = LEX_TYPE;
  } else {
    A = GetRealInitValue("A");
    excoeftype = A_TYPE;
  }

  String kernel_request = GetStringInitValue("kernel", "6ngbr");
  if(kernel_request.compare("6ngbrmirror") == 0 ||
      kernel_request.compare("6ngbr") == 0) {
    kernel = NGBR_6_MIRROR;
  } else {
    String msg = String("Invalid kernel request: ")
      + kernel_request
      + String("\n Should be one of 6ngbr, 6ngbrmirror using GPU,");
    throw Oxs_ExtError(this, msg.c_str());
  }

  VerifyAllInitArgsUsed();
}

Oxs_GPU_UniformExchange_New::~Oxs_GPU_UniformExchange_New() {
#if REPORT_TIME
  Oc_TimeVal cpu,wall;
  FILE* gputimeWall;
  
  gputimeWall = fopen ("gputime_wall.txt","a");

  Exchtime.GetTimes(cpu,wall);
  if (double(wall) > 0.0) {
    fprintf(gputimeWall,"      Exchtime ...   total%7.2f cpu /%7.2f wall, (%.1000s)\n",
            double(cpu),double(wall),InstanceName());
  }
  fclose(gputimeWall);
#endif  	
	ReleaseGPUMemory();
	ReleaseCPUMemory();
}

OC_BOOL Oxs_GPU_UniformExchange_New::Init() {

#if REPORT_TIME
  Oc_TimeVal cpu,wall;

  Exchtime.GetTimes(cpu,wall);
  Exchtime.Reset();
#endif

  mesh_id = 0;
  
  return Oxs_GPU_Energy::Init();
}

void Oxs_GPU_UniformExchange_New::ReInitGPU(const DEVSTRUCT& dev_struct) const {
    
  if(dev_struct.dev_MValue) {
    dev_MValue = dev_struct.dev_MValue;
  } else {
    String msg=String("dev_struct.dev_MValue not initiated in : \"")
      + String(ClassName()) + String("\".");
    throw Oxs_ExtError(this, msg.c_str());
  }

  if(dev_struct.dev_Msi) {
    dev_Msii = dev_struct.dev_Msi;
  } else {
    String msg=String("dev_struct.dev_Msi not initiated in : \"")
      + String(ClassName()) + String("\".");
    throw Oxs_ExtError(this,msg.c_str());
  }
  	
  if(dev_struct.dev_field) {
    dev_Field = dev_struct.dev_field;
  } else {
		String msg=String("dev_struct.dev_field not initiated in : \"")
		  + String(ClassName()) + String("\".");
		throw Oxs_ExtError(this,msg.c_str());
	}

	if(dev_struct.dev_energy) {
    dev_Energy = dev_struct.dev_energy;
  } else {
		String msg=String("dev_struct.dev_Energy not initiated in : \"")
		  + String(ClassName()) + String("\".");
		throw Oxs_ExtError(this,msg.c_str());
	}

	if(dev_struct.dev_torque) {
    dev_Torque = dev_struct.dev_torque;
  } else {
		String msg=String("dev_struct.dev_torque not initiated in : \"")
		  + String(ClassName()) + String("\".");
		throw Oxs_ExtError(this,msg.c_str());
	}
  
  if(dev_struct.dev_local_sum) {
    dev_tmp = dev_struct.dev_local_sum;
  } else{
		String msg=String("dev_struct.dev_local_sum not initiated in : \"")
		  + String(ClassName()) + String("\".");
		throw Oxs_ExtError(this,msg.c_str());
	}

  if(dev_struct.dev_vol) {
    dev_volume = dev_struct.dev_vol;
  } else {
		String msg=String("dev_struct.dev_vol not initiated in : \"")
		  + String(ClassName()) + String("\".");
		throw Oxs_ExtError(this,msg.c_str());
	}
  
  if(dev_struct.dev_dot) {
    dev_dot = dev_struct.dev_dot;
  } else {
		String msg=String("dev_struct.dev_dot not initiated in : \"")
		  + String(ClassName()) + String("\".");
		throw Oxs_ExtError(this,msg.c_str());
	}
  
  if(dev_struct.dev_local_field) {
    dev_field_loc = dev_struct.dev_local_field;
  } else {
		String msg=String("dev_struct.dev_local_field not initiated in : \"")
		  + String(ClassName()) + String("\".");
		throw Oxs_ExtError(this,msg.c_str());
	}
  
  if(dev_struct.dev_local_energy) {
    dev_energy_loc = dev_struct.dev_local_energy;
  } else {
		String msg=String("dev_struct.dev_local_energy not initiated in : \"")
		  + String(ClassName()) + String("\".");
		throw Oxs_ExtError(this,msg.c_str());
	}
}

void Oxs_GPU_UniformExchange_New::InitGPU (
  const OC_INDEX &size, DEVSTRUCT& dev_struct) const {

  fetchInfo_device(maxGridSize, maxTotalThreads, DEV_NUM);

  AllocGPUMemory(size, dev_struct);
  
  // cudaFuncSetCacheConfig(getExchKernelName().c_str(), 
    // cudaFuncCachePreferShared);
  
  AllocCPUMemory(size);
}

void Oxs_GPU_UniformExchange_New::AllocCPUMemory(
    const OC_INDEX &size) const {
    
	if(tmp_field == 0)	{
    tmp_field = new FD_TYPE[ODTV_VECSIZE * size];
  }
  
	if(tmp_energy == 0)	{
    tmp_energy = new FD_TYPE[size];
  }
}

void Oxs_GPU_UniformExchange_New::ReleaseCPUMemory() const {
	if(tmp_field != 0) { 
    delete[] tmp_field;
    tmp_field = 0; 
  }
	if(tmp_energy != 0) {
    delete[] tmp_energy; 
    tmp_energy = 0;
  }
}

void Oxs_GPU_UniformExchange_New::GPU_GetEnergy
(const Oxs_SimState& state,
 Oxs_EnergyData& oed, DEVSTRUCT& dev_struct,
 unsigned int flag_outputH, unsigned int flag_outputE,
 unsigned int flag_outputSumE, const OC_BOOL &flag_accum) const {
#if REPORT_TIME
  cudaDeviceSynchronize();
  Exchtime.Start();
#endif

  const OC_INDEX &size = state.mesh->Size();
  if(size < 1) return; // Nothing to do
  if(mesh_id != state.mesh->Id()) {
    InitGPU(size, dev_struct);
    mesh_id = state.mesh->Id();
  }
  if(dev_MValue != dev_struct.dev_MValue) {
    ReInitGPU(dev_struct);
  }

  const Oxs_CommonRectangularMesh* mesh
    = dynamic_cast<const Oxs_CommonRectangularMesh*>(state.mesh);
  if(mesh==NULL) {
    String msg=String("Object ")
      + String(state.mesh->InstanceName())
      + String(" is not a rectangular mesh.");
    throw Oxs_ExtError(this,msg);
  }

  // Check periodicity.  Note that the following kernels have not been
  // upgraded to supported periodic meshes:
  //   NGBR_12_FREE, NGBR_12_ZD1, NGBR_12_ZD1B, NGBR_26
  // This is checked for and reported in the individual arms of the
  // kernel if-test below.
  const Oxs_RectangularMesh* rmesh 
    = dynamic_cast<const Oxs_RectangularMesh*>(mesh);
  const Oxs_PeriodicRectangularMesh* pmesh
    = dynamic_cast<const Oxs_PeriodicRectangularMesh*>(mesh);
  if(pmesh!=NULL) {
    // Rectangular, periodic mesh
    host_Periodic.x = xperiodic = pmesh->IsPeriodicX();
    host_Periodic.y = yperiodic = pmesh->IsPeriodicY();
    host_Periodic.z = zperiodic = pmesh->IsPeriodicZ();
  } else if (rmesh!=NULL) {
    xperiodic=0; yperiodic=0; zperiodic=0;
	host_Periodic.x = host_Periodic.y = host_Periodic.z = 0;
  } else {
    String msg=String("Unknown mesh type: \"")
      + String(ClassName())
      + String("\".");
    throw Oxs_ExtError(this,msg.c_str());
  }
  
  uint3 host_Dim;
  host_Dim.x = mesh->DimX();
  host_Dim.y = mesh->DimY();
  host_Dim.z = mesh->DimZ();
  
  FD_TYPE3 host_wgt;
  host_wgt.x = -A/(mesh->EdgeLengthX()*mesh->EdgeLengthX());
  host_wgt.y = -A/(mesh->EdgeLengthY()*mesh->EdgeLengthY());
  host_wgt.z = -A/(mesh->EdgeLengthZ()*mesh->EdgeLengthZ());
  
  // GPU kernel size setup
  myBlk3DSize.x = min(myBlk3DSize.x, host_Dim.x);
  myBlk3DSize.y = min(myBlk3DSize.y, host_Dim.y);
  myBlk3DSize.z = min(myBlk3DSize.z, host_Dim.z);
  
  get_cubic_kernel_size(host_Dim, myBlk3DSize, &Knl_Blk_size,
    &Knl_Grid_size);
    
#ifdef GPU_CPU_TRANS
  s_exchUniform(Knl_Grid_size, Knl_Blk_size, myBlk3DSize,
    dev_MValue, dev_Field, dev_Energy, dev_Torque, host_Dim, 
    host_wgt, dev_Msii, host_Periodic, dev_field_loc, dev_energy_loc,
    dev_dot, flag_outputH != 0, flag_outputE != 0 || flag_outputSumE != 0,
    flag_accum);

	// Use supplied buffer space, and reflect that use in oed.
  oed.energy = oed.energy_buffer;
  oed.field = oed.field_buffer;
  Oxs_MeshValue<OC_REAL8m>& energy = *oed.energy_buffer;
  Oxs_MeshValue<ThreeVector>& field = *oed.field_buffer;
  
  if (flag_outputE) {
    memDownload_device(tmp_energy, dev_energy_loc, size, DEV_NUM);
    for (OC_INDEX i = 0; i < size; i++) {
      energy[i] = tmp_energy[i];
    }
  }
  
  if (flag_outputH) {
    memDownload_device(tmp_field, dev_field_loc, ODTV_VECSIZE * size, DEV_NUM);
    for (OC_INDEX i = 0; i < size; i++) {
      field[i] = ThreeVector(tmp_field[i], tmp_field[i+size], tmp_field[i+2*size]);
    }
  }
  
  if (flag_outputSumE) {
    FD_TYPE* &dev_energyVolumeProduct = dev_tmp;
    dotProduct(size, BLK_SIZE, dev_energy_loc, 
      dev_volume, dev_energyVolumeProduct);
    FD_TYPE energy_sum = sum_device(size, 
      dev_energyVolumeProduct, dev_tmp, DEV_NUM, 
      maxGridSize, maxTotalThreads);
    oed.energy_sum = energy_sum;
  }
#endif
		
#if REPORT_TIME
  cudaDeviceSynchronize();
  Exchtime.Stop();
#endif
}