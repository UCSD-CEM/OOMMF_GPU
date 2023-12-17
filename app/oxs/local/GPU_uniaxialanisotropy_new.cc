/* FILE: GPU_uniaxialanisotropy.cu           -*-Mode: cuda-*-
 *
 * Uniaxial anisotropy field, derived from Oxs_Energy class.
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
#include "GPU_uniaxialanisotropy_new.h"
#include "GPU_anisotropy_new_kernel.h"
#include "energy.h"		// Needed to make MSVC++ 5 happy

// OC_USE_STRING;

// Oxs_Ext registration support
OXS_EXT_REGISTER(GPU_UniaxialAnisotropy_New);

/* End includes */

// Constructor
GPU_UniaxialAnisotropy_New::GPU_UniaxialAnisotropy_New(
  const char* name,     // Child instance id
  Oxs_Director* newdtr, // App director
  const char* argstr)   // MIF input block parameters
  : Oxs_GPU_Energy(name,newdtr,argstr),
  	aniscoeftype(ANIS_UNKNOWN), mesh_id(0),
  	K1_is_uniform(0), Ha_is_uniform(0), axis_is_uniform(0),
    uniform_K1_value(0.0), uniform_Ha_value(0.0),
    integration_method(UNKNOWN_INTEG),
    has_multscript(0), number_of_stages(0),
    mult_state_id(0), mult(1.0), dmult(0.0),
    dev_MValue(0), dev_Ms(0), dev_inv_Ms(0),
    dev_Energy(0), dev_Field(0), dev_Torque(0),
    dev_K1(0), dev_Ha(0), dev_axis(0), dev_field_loc(0), 
    dev_energy_loc(0), tmp_field(0),  tmp_energy(0) {
  // GPU supported
  _dev_num = DEV_NUM;
  dev_uniform_axis_value.x = 0.;dev_uniform_axis_value.y = 0.;
  dev_uniform_axis_value.z = 0.;
  dev_uniform_K1_value = 0.; dev_uniform_Ha_value = 0.;
  dev_mult = 0.;

  // Process arguments
  OC_BOOL has_K1 = HasInitValue("K1");
  OC_BOOL has_Ha = HasInitValue("Ha");
  if (has_K1 && has_Ha) {
    throw Oxs_ExtError(this,"Invalid anisotropy coefficient request:"
			 " both K1 and Ha specified; only one should"
			 " be given.");
  } else if (has_K1) {
    OXS_GET_INIT_EXT_OBJECT("K1",Oxs_ScalarField,K1_init);
    Oxs_UniformScalarField* tmpK1ptr
      = dynamic_cast<Oxs_UniformScalarField*>(K1_init.GetPtr());
    if (tmpK1ptr) {
      // Initialization is via a uniform field; set up uniform
      // K1 variables.
      K1_is_uniform = 1;
      uniform_K1_value = tmpK1ptr->SoleValue();
    }
    aniscoeftype = K1_TYPE;
  } else {
    OXS_GET_INIT_EXT_OBJECT("Ha",Oxs_ScalarField,Ha_init);
    Oxs_UniformScalarField* tmpHaptr
      = dynamic_cast<Oxs_UniformScalarField*>(Ha_init.GetPtr());
    if (tmpHaptr) {
      // Initialization is via a uniform field; set up uniform
      // Ha variables.
      Ha_is_uniform = 1;
      uniform_Ha_value = tmpHaptr->SoleValue();
    }
    aniscoeftype = Ha_TYPE;
  }

  OXS_GET_INIT_EXT_OBJECT("axis",Oxs_VectorField,axis_init);
  Oxs_UniformVectorField* tmpaxisptr
    = dynamic_cast<Oxs_UniformVectorField*>(axis_init.GetPtr());
  if(tmpaxisptr) {
    // Initialization is via a uniform field.  For convenience,
    // modify the size of the field components to norm 1, as
    // required for the axis specification.  This allows the
    // user to specify the axis direction as, for example, {1,1,1},
    // as opposed to {0.57735027,0.57735027,0.57735027}, or
    //
    //      Specify Oxs_UniformVectorField {
    //        norm 1 
    //        vector { 1 1 1 } 
    //    }
    // Also setup uniform axis variables
    tmpaxisptr->SetMag(1.0);
    axis_is_uniform = 1;
    uniform_axis_value = tmpaxisptr->SoleValue();
  }


  String integration_request = GetStringInitValue("integration","rect");
  if (integration_request.compare("rect") == 0) {
    integration_method = RECT_INTEG;
  } else if (integration_request.compare("quad") == 0) {
    integration_method = QUAD_INTEG;
    throw Oxs_ExtError(this, "QUAD_INTEG is not supported by GPU library yet");
  } else {
    String msg=String("Invalid integration request: ")
      + integration_request
      + String("\n Should be either \"rect\" or \"quad\".");
    throw Oxs_ExtError(this,msg.c_str());
  }

  has_multscript = HasInitValue("multscript");
  String cmdoptreq;
  String runscript;
  if(has_multscript) {
    throw Oxs_ExtError(this, "multscript is not supported by GPU library yet");
  }
  number_of_stages = GetUIntInitValue("stage_count",0);
  /// Default number_of_stages is 0, i.e., no preference.

  VerifyAllInitArgsUsed();

  #ifdef GPU_DEBUG
    FILE* mylocation = fopen ("location.txt","a");
    fprintf(mylocation,"finish GPU uniaxialAnisotropy_new constructor...\n");
    fclose(mylocation);
  #endif
}

GPU_UniaxialAnisotropy_New::~GPU_UniaxialAnisotropy_New() {
#if REPORT_TIME
  Oc_TimeVal cpu,wall;
  FILE* gputimeWall;
  
  gputimeWall = fopen ("gputime_wall.txt","a");

  Anistime.GetTimes(cpu,wall);
  if (double(wall) > 0.0) {
    fprintf(gputimeWall,"      Anistime ...   total%7.2f cpu /%7.2f wall, (%.1000s)\n",
            double(cpu),double(wall),InstanceName());
  }
  fclose(gputimeWall);
#endif  	
	ReleaseGPUMemory();
	ReleaseCPUMemory();
}

OC_BOOL GPU_UniaxialAnisotropy_New::Init() {

#if REPORT_TIME
  Oc_TimeVal cpu,wall;

  Anistime.GetTimes(cpu,wall);
  Anistime.Reset();
#endif

  mesh_id = 0;
  K1.Release();
  Ha.Release();
  axis.Release();

  mult_state_id = 0;
  mult = 1.0;
  dmult = 0.0;
  
  return Oxs_GPU_Energy::Init();
}

void GPU_UniaxialAnisotropy_New::ReInitGPU(const DEVSTRUCT& dev_struct) const {
    
  if(dev_struct.dev_MValue) {
    dev_MValue = dev_struct.dev_MValue;
  } else {
    String msg=String("dev_struct.dev_MValue not initiated in : \"")
      + String(ClassName()) + String("\".");
    throw Oxs_ExtError(this, msg.c_str());
  }
  
	
  if(dev_struct.dev_Ms) {
    dev_Ms = dev_struct.dev_Ms;
  } else {
		String msg=String("dev_struct.dev_Ms not initiated in : \"")
		  + String(ClassName()) + String("\".");
		throw Oxs_ExtError(this,msg.c_str());
	}
  
  if(dev_struct.dev_Msi) {
    dev_inv_Ms = dev_struct.dev_Msi;
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
		String msg=String("dev_struct.dev_local_sum not initiated in : \"")
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

void GPU_UniaxialAnisotropy_New::InitGPU (
  const OC_INDEX &size, DEVSTRUCT& dev_struct) const {

  fetchInfo_device(maxGridSize, maxTotalThreads, DEV_NUM);
  
  AllocGPUMemory (size, dev_struct);

  // copy CPU array to GPU memory
  FD_TYPE* tmp_container = new FD_TYPE[ODTV_VECSIZE * size];
  if(aniscoeftype == K1_TYPE && !K1_is_uniform) {
    for(int i = 0; i < size; i++) {
      tmp_container[i] = (FD_TYPE)K1[i];
    }
    memUpload_device(dev_K1, tmp_container, size, _dev_num);
    release_device(dev_Ha, _dev_num, "dev_Ha");
  } else if(aniscoeftype == Ha_TYPE && !Ha_is_uniform) {
    for(int i = 0; i < size; i++)	{
      tmp_container[i] = (FD_TYPE)Ha[i];
    }
    memUpload_device(dev_Ha, tmp_container, size, _dev_num);
    release_device(dev_K1, _dev_num, "dev_K1");
  }
  
  if (!axis_is_uniform) {
    for(int i = 0; i < size; i++) {
      tmp_container[i] = (FD_TYPE)axis[i].x;
      tmp_container[i + size] = (FD_TYPE)axis[i].y;
      tmp_container[i + 2 * size] = (FD_TYPE)axis[i].z;
    }
    memUpload_device(dev_axis, tmp_container, ODTV_VECSIZE * size, _dev_num);
  }
  if (tmp_container) {
    delete[] tmp_container;
  }
  
  AllocCPUMemory(size);
  
  //flags definition for the kernel computation
  dev_uniform_axis_value.x = (FD_TYPE)uniform_axis_value.x;
  dev_uniform_axis_value.y = (FD_TYPE)uniform_axis_value.y;
  dev_uniform_axis_value.z = (FD_TYPE)uniform_axis_value.z;

  dev_uniform_K1_value = (FD_TYPE)uniform_K1_value;
  dev_uniform_Ha_value = (FD_TYPE)uniform_Ha_value;
  dev_mult = (FD_TYPE)mult;
          
  if(K1_is_uniform)	{
    flag_uniform.x = 1;
  } else {
    flag_uniform.x = 0;
  }
  
  if(Ha_is_uniform)	{
    flag_uniform.y = 1; 
  } else {
    flag_uniform.y = 0;
  }
  
  if(axis_is_uniform)	{
    flag_uniform.z = 1; 
  } else {
    flag_uniform.z = 0;
  }

	// kernel size of GPU
  getFlatKernelSize(size, BLK_SIZE, grid_size, block_size);
}

void GPU_UniaxialAnisotropy_New::AllocCPUMemory(
    const OC_INDEX &size) const {
    
	if(tmp_field == 0)	{
    tmp_field = new FD_TYPE[ODTV_VECSIZE * size];
  }
  
	if(tmp_energy == 0)	{
    tmp_energy = new FD_TYPE[size];
  }
}

void GPU_UniaxialAnisotropy_New::ReleaseCPUMemory() const {
	if(tmp_field != 0) { 
    delete[] tmp_field;
    tmp_field = 0; 
  }
	if(tmp_energy != 0) {
    delete[] tmp_energy; 
    tmp_energy = 0;
  }
}

void GPU_UniaxialAnisotropy_New::AllocGPUMemory(
  const OC_INDEX &size, DEVSTRUCT& dev_struct) const {
  release_device(dev_K1, _dev_num, "dev_K1");
  alloc_device(dev_K1, size, _dev_num, "dev_K1");

  release_device(dev_Ha, _dev_num, "dev_Ha");
  alloc_device(dev_Ha, size, _dev_num, "dev_Ha");
      
  release_device(dev_axis, _dev_num, "dev_axis");
  alloc_device(dev_axis, ODTV_VECSIZE * size, _dev_num, "dev_axis");
}

void GPU_UniaxialAnisotropy_New::ReleaseGPUMemory() const {
  release_device(dev_K1, _dev_num, "dev_K1");
  release_device(dev_Ha, _dev_num, "dev_Ha");
  release_device(dev_axis, _dev_num, "dev_axis");
}

void GPU_UniaxialAnisotropy_New::GPU_GetEnergy
(const Oxs_SimState& state,
 Oxs_EnergyData& oed, DEVSTRUCT& dev_struct,
 unsigned int flag_outputH, unsigned int flag_outputE,
 unsigned int flag_outputSumE, const OC_BOOL &flag_accum) const {
#if REPORT_TIME
  cudaDeviceSynchronize();
  Anistime.Start();
#endif

  const OC_INDEX &size = state.mesh->Size();
  if(size < 1) return; // Nothing to do
  if(mesh_id != state.mesh->Id()) {
    if(aniscoeftype == K1_TYPE) {
      if(!K1_is_uniform) K1_init->FillMeshValue(state.mesh,K1);
    } else if(aniscoeftype == Ha_TYPE) {
      if(!Ha_is_uniform) Ha_init->FillMeshValue(state.mesh,Ha);
    }
    if(!axis_is_uniform) {
      axis_init->FillMeshValue(state.mesh,axis);
      for(OC_INDEX i=0;i<size;i++) {
        // Check that axis is a unit vector:
        const OC_REAL8m eps = 1e-14;
        if(axis[i].MagSq()<eps*eps) {
          throw Oxs_ExtError(this,"Invalid initialization detected:"
                             " Zero length anisotropy axis");
        } else {
          axis[i].MakeUnit();
        }
      }
    }
    if(has_multscript && mult_state_id !=  state.Id()) {
      throw Oxs_ExtError(this, "multiscript is not supported by GPU library");
    } else {
      GetMultiplier(state, mult, dmult);
      mult_state_id = state.Id();
    }
    InitGPU(size, dev_struct);
    mesh_id = state.mesh->Id();
  }
  
  if(dev_MValue != dev_struct.dev_MValue) {
    ReInitGPU(dev_struct);
  }

  const bool dev_k1_type = (aniscoeftype == K1_TYPE);
  
#ifdef GPU_CPU_TRANS
  Rec_Integ(grid_size, block_size, dev_Ms, dev_inv_Ms, dev_MValue, size,
    dev_uniform_K1_value, dev_uniform_Ha_value,
    dev_uniform_axis_value, dev_mult, dev_k1_type, 
    flag_uniform, dev_K1, dev_Ha, dev_axis, dev_Field,
    dev_Torque, dev_Energy, dev_field_loc, dev_energy_loc, 
    flag_outputH != 0, flag_outputE != 0 || flag_outputSumE != 0,
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
  Anistime.Stop();
#endif
}
