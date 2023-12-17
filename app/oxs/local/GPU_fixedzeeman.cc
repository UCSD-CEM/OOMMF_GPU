/* FILE: fixedzeeman.cc           -*-Mode: c++-*-
 *
 * Fixed (in time) Zeeman energy/field, derived from Oxs_Energy class.
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
#include "GPU_fixedzeeman.h"
#include "GPU_zeeman_kernel.h"

// Oxs_Ext registration support
OXS_EXT_REGISTER(GPU_FixedZeeman);

/* End includes */


// Constructor
GPU_FixedZeeman::GPU_FixedZeeman(
  const char* name,     // Child instance id
  Oxs_Director* newdtr, // App director
  const char* argstr)   // MIF input block parameters
  : Oxs_GPU_Energy(name,newdtr,argstr), mesh_id(0),
  dev_ZField(0), dev_Field(0), dev_Energy(0),
  dev_MValue(0), dev_Torque(0), dev_Ms(0),
  dev_energy_loc(0), tmp_energy(0), tmp_field(0) {
  // Process arguments
  field_mult = GetRealInitValue("multiplier",1.0);
  OXS_GET_INIT_EXT_OBJECT("field",Oxs_VectorField,fixedfield_init);
  VerifyAllInitArgsUsed();
  
  //GPU parameter
  _dev_num = DEV_NUM;
  fetchInfo_device(maxGridSize, maxTotalThreads, DEV_NUM);
}

GPU_FixedZeeman::~GPU_FixedZeeman() {
#if REPORT_TIME
  Oc_TimeVal cpu,wall;
  FILE* gputimeWall;
  
  gputimeWall = fopen ("gputime_wall.txt","a");

  Zeemantime.GetTimes(cpu,wall);
  if(double(wall)>0.0) {
    fprintf(gputimeWall,"      Zeemantime ...   total%7.2f cpu /%7.2f wall, (%.1000s)\n",
            double(cpu),double(wall),InstanceName());
  }
  fclose(gputimeWall);
#endif  	
	ReleaseDevMemory();
	ReleaseTmpMemory();
}

OC_BOOL GPU_FixedZeeman::Init()
{

#if REPORT_TIME
  Oc_TimeVal cpu,wall;

  Zeemantime.GetTimes(cpu,wall);
  Zeemantime.Reset();
#endif

  mesh_id = 0;
  fixedfield.Release();
  
  return Oxs_GPU_Energy::Init();
}

void GPU_FixedZeeman::ReInitGPU(const DEVSTRUCT &dev_struct) const {
    	if(dev_struct.dev_MValue)	dev_MValue = dev_struct.dev_MValue;
	else{
			String msg=String("dev_struct.dev_MValue not initiated in : \"")
			  + String(ClassName()) + String("\".");
			throw Oxs_ExtError(this,msg.c_str());
	}
	if(dev_struct.dev_Ms) dev_Ms = dev_struct.dev_Ms;
	else{
		String msg=String("dev_struct.dev_Ms not initiated in : \"")
		  + String(ClassName()) + String("\".");
		throw Oxs_ExtError(this,msg.c_str());
	}
	if(dev_struct.dev_field) dev_Field = dev_struct.dev_field;
	else{
		String msg=String("dev_struct.dev_field not initiated in : \"")
		  + String(ClassName()) + String("\".");
		throw Oxs_ExtError(this,msg.c_str());
	}
	if(dev_struct.dev_energy) dev_Energy = dev_struct.dev_energy;
	else{
		String msg=String("dev_struct.dev_energy not initiated in : \"")
		  + String(ClassName()) + String("\".");
		throw Oxs_ExtError(this,msg.c_str());
	}
	if(dev_struct.dev_torque) dev_Torque = dev_struct.dev_torque;
	else{
		String msg=String("dev_struct.dev_torque not initiated in : \"")
		  + String(ClassName()) + String("\".");
		throw Oxs_ExtError(this,msg.c_str());
	}
  if(dev_struct.dev_local_sum) dev_tmp = dev_struct.dev_local_sum;
	else{
		String msg=String("dev_struct.dev_local_sum not initiated in : \"")
		  + String(ClassName()) + String("\".");
		throw Oxs_ExtError(this,msg.c_str());
	}
  
  if(dev_struct.dev_vol) {
    dev_volume = dev_struct.dev_vol;
  } else {
		String msg=String("dev_struct.dev_local_sum not initiated in : \"")
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

#ifdef GPU_DEBUG
  cudaDeviceSynchronize();   
  _cuda_error_code = cudaGetLastError();           
  debugInfo = fopen("location.txt", "a");
  fprintf(debugInfo, __FILE__);fprintf(debugInfo, " ");
  fprintf(debugInfo, cudaGetErrorString(_cuda_error_code));
  fprintf(debugInfo, " in line %d\n", __LINE__);
  if(_cuda_error_code != cudaSuccess ){
	fprintf(debugInfo, "ERROR REPORTED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n", __LINE__);
  }
  fclose(debugInfo);

#endif

}

void GPU_FixedZeeman::AllocDevMemory(int size, 
	DEVSTRUCT& dev_struct) const
{
  release_device(dev_ZField, _dev_num, "dev_ZField");
  alloc_device(dev_ZField, size * ODTV_VECSIZE, _dev_num, "dev_ZField");
}

void GPU_FixedZeeman::ReleaseDevMemory() const{
  release_device(dev_ZField, _dev_num, "dev_ZField");
}


void GPU_FixedZeeman::AllocTmpMemory(OC_INDEX size) const {

	if( tmp_energy == 0 )	tmp_energy = new FD_TYPE[size];
	if( tmp_field == 0 )	tmp_field = new FD_TYPE[size*ODTV_VECSIZE];
	
}

void GPU_FixedZeeman::ReleaseTmpMemory() const {

	if( tmp_energy != 0 )		{ delete[] tmp_energy; tmp_energy = 0; };
	if( tmp_field != 0 )		{ delete[] tmp_field; tmp_field = 0; };
}

void GPU_FixedZeeman::GPU_GetEnergy
(const Oxs_SimState& state,
 Oxs_EnergyData& oed, DEVSTRUCT& dev_struct,
 unsigned int flag_outputH, unsigned int flag_outputE,
 unsigned int flag_outputSumE, const OC_BOOL &flag_accum) const
{
#if REPORT_TIME
  cudaDeviceSynchronize();
  Zeemantime.Start();
#endif
  OC_INDEX size = state.mesh->Size();
  if(size<1) return; // Nothing to do
  if(mesh_id != state.mesh->Id()) {
    // This is either the first pass through, or else mesh
    // has changed.
	AllocDevMemory(size, dev_struct);
	AllocTmpMemory(size);
    mesh_id = 0;
    fixedfield_init->FillMeshValue(state.mesh,fixedfield);
    if(field_mult!=1.0) {
      for(OC_INDEX i=0;i<size;i++) fixedfield[i] *= field_mult;
    }
	for (int i=0; i<size; i++) {
		tmp_field[i  ] = (FD_TYPE)fixedfield[i].x;
		tmp_field[i+size] = (FD_TYPE)fixedfield[i].y;
		tmp_field[i+2*size] = (FD_TYPE)fixedfield[i].z;
	}
  memUpload_device(dev_ZField, tmp_field, size * ODTV_VECSIZE, _dev_num);
  getFlatKernelSize(size, BLK_SIZE, grid_size, block_size);
    mesh_id = state.mesh->Id();
  } 
  
  if (dev_MValue != dev_struct.dev_MValue) {
    ReInitGPU(dev_struct);
  }

  // Use supplied buffer space, and reflect that use in oed.
#ifdef GPU_CPU_TRANS
  Get_Fixed_Zeeman(grid_size, block_size, dev_MValue, dev_Ms, 
    dev_ZField, dev_Field, dev_Energy, dev_Torque, size,
    flag_outputE != 0 || flag_outputSumE != 0, flag_accum, dev_energy_loc);
#endif
#ifdef GPU_CPU_TRANS
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
    for (OC_INDEX i = 0; i < size; i++) {
      field[i] = ThreeVector(tmp_field[i  ], tmp_field[i+size], tmp_field[i+2*size]);
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
  Zeemantime.Stop();
#endif
}