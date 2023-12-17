/* FILE: GPU energy.cc                 -*-Mode: c++-*-
 *
 * Abstract GPU energy class, derived from Oxs_Energy class.  Note: The
 * implementation of the Oxs_GPU_ComputeEnergies() friend function is
 * in the file GPU_chunkenergy.cc.
 */

#include <assert.h>
#include <string>

#include "GPU_energy.h"
#include "mesh.h"

#include "GPU_helper.h"

#define ODTV_VECSIZE 3

OC_USE_STRING;
/* End includes */

#ifdef EXPORT_CALC_COUNT
void Oxs_GPU_Energy::FillCalcCountOutput(const Oxs_SimState& state)
{
  calc_count_output.cache.state_id = state.Id();
  calc_count_output.cache.value    = static_cast<OC_REAL8m>(calc_count);
}
#endif // EXPORT_CALC_COUNT

void Oxs_GPU_Energy::SetupOutputs()
{
  energy_sum_output.Setup(this,InstanceName(),"Energy","J",1,
                          &Oxs_GPU_Energy::UpdateStandardOutputs);
  field_output.Setup(this,InstanceName(),"Field","A/m",1,
                     &Oxs_GPU_Energy::UpdateStandardOutputs);
  energy_density_output.Setup(this,InstanceName(),"Energy density","J/m^3",1,
                     &Oxs_GPU_Energy::UpdateStandardOutputs);
#ifdef EXPORT_CALC_COUNT
  calc_count_output.Setup(this,InstanceName(),"Calc count","",0,
                          &Oxs_GPU_Energy::FillCalcCountOutput);
#endif // EXPORT_CALC_COUNT
  // Note: MS VC++ 6.0 requires fully qualified member names

  // Register outputs
  energy_sum_output.Register(director,0);
  field_output.Register(director,0);
  energy_density_output.Register(director,0);
#ifdef EXPORT_CALC_COUNT
  calc_count_output.Register(director,0);
#endif // EXPORT_CALC_COUNT
}

// Constructors
Oxs_GPU_Energy::Oxs_GPU_Energy
( const char* name,     // Child instance id
  Oxs_Director* newdtr  // App director
  ) : Oxs_Energy(name,newdtr, true), calc_count(0), initialized(false)
{ 
	host_struct_copy.dev_Ms = NULL;
	SetupOutputs();
}

Oxs_GPU_Energy::Oxs_GPU_Energy
( const char* name,     // Child instance id
  Oxs_Director* newdtr, // App director
  const char* argstr    // MIF block argument string
  ) : Oxs_Energy(name,newdtr,argstr, true), calc_count(0), 
  initialized(false)
{
	SetupOutputs();
}
	
//Destructor
Oxs_GPU_Energy::~Oxs_GPU_Energy() {
#if REPORT_TIME
  Oc_TimeVal cpu,wall;
  energytime.GetTimes(cpu,wall);
  if(double(wall)>0.0) {
    fprintf(stderr,"GetEnergy time (secs)%7.2f cpu /%7.2f wall,"
            " module %.1000s (%u evals)\n",double(cpu),double(wall),
            InstanceName(),GetEnergyEvalCount());
  }
#endif // REPORT_TIME
}

// following GPU initialization and deallocation functions are copied
// from class GPU_timeevolver
OC_BOOL Oxs_GPU_Energy::InitGPU(const Oxs_SimState& state) {
  OC_INDEX size = state.mesh->Size();
  initialized = false;
  host_struct_copy.allocMem(state.mesh);
  host_struct_copy.purgeMem(state.mesh);
  
  FD_TYPE *tmp = new FD_TYPE[ODTV_VECSIZE * size];
  for (int i = 0; i < size; i++) {
    tmp[i] = (*(state.Ms))[i];
  }
  memUpload_device(host_struct_copy.dev_Ms, tmp, size, DEV_NUM);
  
  for (int i = 0; i < size; i++) {
    tmp[i] = (*(state.Ms_inverse))[i];
  }
  memUpload_device(host_struct_copy.dev_Msi, tmp, size, DEV_NUM);
  
  for (int i = 0; i < size; i++) {
    tmp[i] = state.mesh->Volume(i);
  }
  memUpload_device(host_struct_copy.dev_vol, tmp, size, DEV_NUM);

  for (int i = 0; i < size; i++) {
    tmp[i] = state.spin[i].x;
    tmp[i + size] = state.spin[i].y;
    tmp[i + 2 * size] = state.spin[i].z;
  }
  memUpload_device(host_struct_copy.dev_MValue, tmp, 
    ODTV_VECSIZE * size, DEV_NUM);

  delete[] tmp;
  initialized = true;
  return true;
}

// Default problem initializer routine.  Any child that redefines
// this function should embed a call to this Init() inside
// the child specific version.
OC_BOOL Oxs_GPU_Energy::Init()
{
  if(!Oxs_Energy::Init()) return 0;

#if REPORT_TIME
  Oc_TimeVal cpu,wall;
  energytime.GetTimes(cpu,wall);
  if(double(wall)>0.0) {
    fprintf(stderr,"GetEnergy time (secs)%7.2f cpu /%7.2f wall,"
            " module %.1000s (%u evals)\n",double(cpu),double(wall),
            InstanceName(),GetEnergyEvalCount());
  }
  energytime.Reset();
#endif // REPORT_TIME

  calc_count=0;
  host_struct_copy.reset();
  initialized = false;
  return 1;
}

// Standard output object update interface
void Oxs_GPU_Energy::UpdateStandardOutputs(const Oxs_SimState& state)
{ 
  if(state.Id()==0) { // Safety
    return;
  }

#if REPORT_TIME
  energytime.Start();
#endif // REPORT_TIME

  Oxs_ComputeEnergyData oced(state);

  // Dummy buffer space.
  Oxs_MeshValue<OC_REAL8m> dummy_energy;
  Oxs_MeshValue<ThreeVector> dummy_field;
  oced.scratch_energy = &dummy_energy;
  oced.scratch_H      = &dummy_field;

  if(energy_density_output.GetCacheRequestCount()>0) {
    energy_density_output.cache.state_id=0;
    oced.scratch_energy = oced.energy = &energy_density_output.cache.value;
    oced.energy->AdjustSize(state.mesh);
  }

  if(field_output.GetCacheRequestCount()>0) {
    field_output.cache.state_id=0;
    oced.scratch_H = oced.H = &field_output.cache.value;
    oced.H->AdjustSize(state.mesh);
  }
  ++calc_count;
  
  if (!initialized) {
    // initiate temp GPU memories
    InitGPU(state);
    // compute
    GPU_ComputeEnergy(state, oced, host_struct_copy, false);
    // deallocate temp GPU memories
    host_struct_copy.releaseMem();
    initialized = false;
  } else {
    GPU_ComputeEnergy(state, oced, host_struct_copy, false);
  }
  
  
  if(energy_density_output.GetCacheRequestCount()>0) {
    energy_density_output.cache.state_id=state.Id();
  }
  if(field_output.GetCacheRequestCount()>0) {
    field_output.cache.state_id=state.Id();
  }
  if(energy_sum_output.GetCacheRequestCount()>0) {
    energy_sum_output.cache.value=oced.energy_sum;
    energy_sum_output.cache.state_id=state.Id();
  }

#if REPORT_TIME
  energytime.Stop();
#endif // REPORT_TIME

}

////////////////////////////////////////////////////////////////////////
// The ComputeEnergy interface replaces the older GetEnergy interface.
// The parameter list is similar, but ComputeEnergy uses the
// Oxs_ComputeEnergyData data struct in place Oxs_EnergyData.  The
// state_id, scratch_energy and scratch_H members of
// Oxs_ComputeEnergyData must be set on entry to ComputeEnergy.  The
// scratch_* members must be non-NULL, but the underlying
// Oxs_MeshValue objects will be size adjusted as (and if) needed.
// The scratch_* members are need for backward compatibility with
// older (pre Oct 2008) Oxs_Energy child classes, but also for
// Oxs_Energy classes like Oxs_Demag that always require space for
// field output.  Member "scratch_energy" is expressly allowed to be
// the same as member "energy", and likewise for "scratch_H" and "H".
//
// The remaining Oxs_MeshValue pointers are output requests.  They can
// be NULL, in which case the output is not requested, or non-NULL, in
// which case output is requested.  If output is requested, then the
// corresponding Oxs_MeshValue object will be filled.  (Note that the
// usual ComputeEnergy caller, AccumEnergyAndTorque, may adjust some
// of these pointers to point into Oxs_Energy standard output cache
// space, but the ComputeEnergy function itself plays no such games.)
// Any of these members that are non-NULL must be pre-sized
// appropriately for the given mesh.  This sizing is done automatically
// by AccumEnergyAndTorque for the "energy", "H", and "mxH" members,
// but not for the "accum" members.
//
// The remaining members, energy_sum and pE_pt are exports that are
// always filled by ComputeEnergy.
//
// The main practical advantage of ComputeEnergy over GetEnergy
// is that use of the "accum" fields can allow significant reduction
// in memory bandwidth use in evolvers.  This can be especially
// important in multi-threaded settings.
//
// The following is a default implementation of the virtual
// ComputeEnergy function.  It is effectively a wrapper/adapter
// to the deprecated GetEnergy function.  New Oxs_Energy child
// classes should override this function with a child-specific
// version, and define their GetEnergy function as a simple wrapper
// to GetEnergyAlt (q.v.).
//
void Oxs_GPU_Energy::GPU_ComputeEnergy
(const Oxs_SimState& state,
 Oxs_ComputeEnergyData& oced,
 DEVSTRUCT& host_struct, 
 const OC_BOOL &flag_accum) const
{

  if (!initialized) {
    host_struct_copy = host_struct;
    initialized = true;
  }
  const Oxs_Mesh* mesh = state.mesh;

  if(oced.scratch_energy==NULL || oced.scratch_H==NULL) {
    // Bad input
    String msg = String("Oxs_ComputeEnergyData object in function"
                        " Oxs_GPU_Energy::GPU_ComputeEnergy"
                        " contains NULL scratch pointers.");
    throw Oxs_ExtError(this,msg.c_str());
  }

  if((oced.energy_accum!=0 && !oced.energy_accum->CheckMesh(mesh))
     || (oced.H_accum!=0   && !oced.H_accum->CheckMesh(mesh))
     || (oced.mxH_accum!=0 && !oced.mxH_accum->CheckMesh(mesh))
     || (oced.energy!=0    && !oced.energy->CheckMesh(mesh))
     || (oced.H!=0         && !oced.H->CheckMesh(mesh))
     || (oced.mxH!=0       && !oced.mxH->CheckMesh(mesh))) {
    // Bad input
    String msg = String("Oxs_ComputeEnergyData object in function"
                        " Oxs_GPU_Energy::GPU_ComputeEnergy"
                        " contains ill-sized buffers.");
    throw Oxs_ExtError(this,msg.c_str());
  }

  Oxs_EnergyData oed(state);
  if(oced.energy) oed.energy_buffer = oced.energy;
  else            oed.energy_buffer = oced.scratch_energy;
  if(oced.H)      oed.field_buffer  = oced.H;
  else            oed.field_buffer  = oced.scratch_H;

  // Although not stated in the interface docs, some Oxs_Energy children
  // assume that the oed energy and field buffers are pre-sized on
  // entry.  For backwards compatibility, make this so.
  oed.energy_buffer->AdjustSize(mesh);
  oed.field_buffer->AdjustSize(mesh);

  unsigned int flag_outputE = 
    (energy_density_output.GetCacheRequestCount()>0);
  unsigned int flag_outputSumE =
    (energy_sum_output.GetCacheRequestCount() > 0);
  unsigned int flag_outputH = 
    (field_output.GetCacheRequestCount() > 0);
	
  GPU_GetEnergy(state,oed,host_struct, flag_outputH,
		flag_outputE, flag_outputSumE, flag_accum);  
  // Accum as requested
  OC_INDEX i;
  const OC_INDEX size = mesh->Size();

  const OC_BOOL have_energy_sum = oed.energy_sum.IsSet();
  if(have_energy_sum) {
    oced.energy_sum = oed.energy_sum;
  } else {
    oced.energy_sum = 0.0;
  }

  // Copy energy and field results, as needed
  if(oced.energy && oced.energy != oed.energy.Get()) (*oced.energy) = (*oed.energy);
  if(oced.H      && oced.H      != oed.field.Get())  (*oced.H)      = (*oed.field);
  
  // pE_pt
 if(oed.pE_pt.IsSet()) oced.pE_pt = oed.pE_pt;
 else                  oced.pE_pt = 0.0;  
}
