/* FILE: fixedzeeman.h            -*-Mode: c++-*-
 *
 * Fixed (in time) Zeeman energy/field, derived from Oxs_Energy class.
 *
 */

#ifndef _GPU_FIXEDZEEMAN
#define _GPU_FIXEDZEEMAN

#ifndef BLK_SIZE
#define BLK_SIZE 128
#endif

#define GPU_CPU_TRANS

#include "oc.h"
#include "director.h"
#include "threevector.h"
#include "GPU_energy.h"
#include "simstate.h"
#include "mesh.h"
#include "meshvalue.h"
#include "vectorfield.h"

#include "GPU_helper.h"

/* End includes */

class GPU_FixedZeeman
  : public Oxs_GPU_Energy, public Oxs_EnergyPreconditionerSupport  {
private:
  mutable OC_UINT4m mesh_id;
  OC_REAL8m field_mult;
  Oxs_OwnedPointer<Oxs_VectorField> fixedfield_init;
  mutable Oxs_MeshValue<ThreeVector> fixedfield;
  /// fixedfield is a cached value filled by
  /// fixedfield_init when a change in mesh is
  /// detected.
#if REPORT_TIME
  mutable Nb_StopWatch Zeemantime;
#endif
  mutable int maxGridSize;
  mutable FD_TYPE maxTotalThreads;
  mutable dim3 block_size;
  mutable dim3 grid_size;
  mutable FD_TYPE *dev_ZField;
  mutable FD_TYPE *dev_Field;
  mutable FD_TYPE *dev_Energy;
  mutable FD_TYPE *dev_MValue;
  mutable FD_TYPE *dev_Torque;
  mutable FD_TYPE *dev_Ms;
  mutable FD_TYPE *dev_energy_loc;
  mutable FD_TYPE *dev_tmp;
  mutable FD_TYPE *tmp_energy;
  mutable FD_TYPE *tmp_field;
  mutable FD_TYPE* dev_volume;
  mutable int _dev_num;
  void ReInitGPU(const DEVSTRUCT &dev_struct) const;
  void AllocDevMemory(int rsize, DEVSTRUCT& dev_struct) const;
  void ReleaseDevMemory() const;
  void ReleaseTmpMemory() const;
  void AllocTmpMemory(OC_INDEX size) const ;
  
protected:
  virtual void GetEnergy(const Oxs_SimState& state,
			 Oxs_EnergyData& oed) const{};
  virtual void GPU_GetEnergy(const Oxs_SimState& state,
			 Oxs_EnergyData& oed, DEVSTRUCT& dev_struct,
			 unsigned int flag_outputH, unsigned int flag_outputE,
			 unsigned int flag_outputSumE, const OC_BOOL &flag_accum) const;
public:
  virtual const char* ClassName() const; // ClassName() is
  /// automatically generated by the OXS_EXT_REGISTER macro.
  GPU_FixedZeeman(const char* name,  // Child instance id
		  Oxs_Director* newdtr, // App director
		  const char* argstr);  // MIF input block parameters

  virtual ~GPU_FixedZeeman();// {ReleaseDevMemory();}
  virtual OC_BOOL Init();

  // Optional interface for conjugate-gradient evolver.
  virtual OC_INT4m IncrementPreconditioner(PreconditionerData& pcd) {
    throw Oxs_ExtError(this, "preconditioner is not supported by GPU libraries yet");
  }
};


#endif // _GPU_FIXEDZEEMAN