/* FILE: timeevolver.cc                 -*-Mode: c++-*-
 *
 * Abstract time evolver class
 *
 */

#include "director.h"
#include "GPU_timeevolver.h"

#include "GPU_energy.h"
#include "GPU_helper.h"

/* End includes */
#define ODTV_VECSIZE 3

// Constructors
Oxs_GPU_TimeEvolver::Oxs_GPU_TimeEvolver
(const char* name,
 Oxs_Director* newdtr,
 const char* argstr, DEVSTRUCT &host_struct_in)      // MIF block argument string
  : Oxs_TimeEvolver(name,newdtr,argstr,true), energy_calc_count(0),
  energy_state_id(0), initialized(false),
  host_struct(host_struct_in) {
  total_energy_output.Setup(this,InstanceName(),
                            "Total energy","J",1,
                            &Oxs_GPU_TimeEvolver::UpdateEnergyOutputs);
  total_energy_density_output.Setup(this,InstanceName(),
                           "Total energy density","J/m^3",1,
                           &Oxs_GPU_TimeEvolver::UpdateEnergyOutputs);
  total_field_output.Setup(this,InstanceName(),
                           "Total field","A/m",1,
                           &Oxs_GPU_TimeEvolver::UpdateEnergyOutputs);
  energy_calc_count_output.Setup(this,InstanceName(),
                           "Energy calc count","",0,
                           &Oxs_GPU_TimeEvolver::FillEnergyCalcCountOutput);
  /// Note: MSVC++ 6.0 requires fully qualified member names

  total_energy_output.Register(director,-5);
  total_energy_density_output.Register(director,-5);
  total_field_output.Register(director,-5);
  energy_calc_count_output.Register(director,-5);

  //GPU initialization
  fetchInfo_device(maxGridSize, maxTotalThreads, DEV_NUM);
}

OC_BOOL Oxs_GPU_TimeEvolver::Init() {
#if REPORT_TIME
  Oc_TimeVal cpu,wall;
  steponlytime.GetTimes(cpu,wall);
  if(double(wall)>0.0) {
    fprintf(stderr,"   Step-only    .....   %7.2f cpu /%7.2f wall,"
            " (%.1000s)\n",
            double(cpu),double(wall),InstanceName());
  }
  steponlytime.Reset();
  myEnergyTime.Reset();
#endif // REPORT_TIME

  energy_calc_count=0;
  energy_state_id = 0;

  // Release scratch space.
  temp_energy.Release();
  temp_field.Release();

  host_struct.releaseMem();
  initialized = false;
  return Oxs_TimeEvolver::Init();
}

OC_BOOL Oxs_GPU_TimeEvolver::InitGPU(const Oxs_SimState& state) {
  OC_INDEX size = state.mesh->Size();
  initialized = false;
  host_struct.allocMem(state.mesh);
  host_struct.purgeMem(state.mesh);
  
  FD_TYPE *tmp = new FD_TYPE[ODTV_VECSIZE * size];
  for (int i = 0; i < size; i++) {
    tmp[i] = (*(state.Ms))[i];
  }
  memUpload_device(host_struct.dev_Ms, tmp, size, DEV_NUM);
  
  for (int i = 0; i < size; i++) {
    tmp[i] = (*(state.Ms_inverse))[i];
  }
  memUpload_device(host_struct.dev_Msi, tmp, size, DEV_NUM);
  
  for (int i = 0; i < size; i++) {
    tmp[i] = state.mesh->Volume(i);
  }
  memUpload_device(host_struct.dev_vol, tmp, size, DEV_NUM);

  for (int i = 0; i < size; i++) {
    tmp[i] = state.spin[i].x;
    tmp[i + size] = state.spin[i].y;
    tmp[i + 2 * size] = state.spin[i].z;
  }
  memUpload_device(host_struct.dev_MValue, tmp, 
    ODTV_VECSIZE * size, DEV_NUM);

  delete[] tmp;
  return true;
}

Oxs_GPU_TimeEvolver::~Oxs_GPU_TimeEvolver() {
#if REPORT_TIME
  Oc_TimeVal cpu,wall;
  steponlytime.GetTimes(cpu,wall);
  
  FILE* gputimeWall;
  gputimeWall = fopen ("gputime_wall.txt","a");
   if(double(wall)>0.0) {
    fprintf(gputimeWall,"   Step-only    .....   %7.2f cpu /%7.2f wall,"
            " (%.1000s)\n",
            double(cpu),double(wall),InstanceName());
  }
  fclose(gputimeWall);
  
  myEnergyTime.GetTimes(cpu,wall);
  gputimeWall = fopen("gputime_wall.txt","a");
  if(double(wall)>0.0) {
    fprintf(gputimeWall,"   Energy    .....   %7.2f cpu /%7.2f wall,"
            " (%.1000s)\n",
            double(cpu),double(wall),InstanceName());
  }
  fclose(gputimeWall);
#endif // REPORT_TIME

  host_struct.releaseMem();
}


// GetEnergyDensity: Note that mxH is returned, as opposed to MxH.
// This relieves this routine from needing to know what Ms is, and saves
// an unneeded multiplication (since the evolver is just going to divide
// it back out again to calculate dm/dt (as opposed again to dM/dt)).
// The returned energy array is average energy density for the
// corresponding cell in J/m^3; mxH is in A/m, pE_pt (partial derivative
// of E with respect to t) is in J/s.  Any of mxH or H may be
// NULL, which disables assignment for that variable.
void Oxs_GPU_TimeEvolver::GPU_GetEnergyDensity
(const Oxs_SimState& state,
 Oxs_MeshValue<OC_REAL8m>& energy,
 Oxs_MeshValue<ThreeVector>* mxH_req,
 Oxs_MeshValue<ThreeVector>* H_req,
 OC_REAL8m& pE_pt) {
  if (!initialized) {
    InitGPU(state);
    initialized = true;
  }
  
  host_struct.purgeAccumMem(state.mesh->Size());

  // Update call count
  ++energy_calc_count;

  Oxs_MeshValue<ThreeVector>* H_fill = H_req;

  // Set up energy computation output data structure
  Oxs_ComputeEnergyData oced(state);
  oced.scratch_energy = &temp_energy;
  oced.scratch_H      = &temp_field;
  oced.energy_accum   = &energy;
  oced.H_accum        = H_fill;
  oced.mxH_accum      = mxH_req;
  oced.energy         = NULL;  // Required null
  oced.H              = NULL;  // Required null
  oced.mxH            = NULL;  // Required null

  UpdateFixedSpinList(state.mesh);
  Oxs_ComputeEnergyExtraData oceed(GetFixedSpinList(),0);

  // Compute total energy and torque
#if REPORT_TIME
  OC_BOOL sot_running = steponlytime.IsRunning();
  if(sot_running) {
    steponlytime.Stop();
  }
  myEnergyTime.Start();
#endif // REPORT_TIME

  Oxs_GPU_ComputeEnergies(state,oced,director->GetEnergyObjects(),
    oceed, host_struct);
 
#if REPORT_TIME
  myEnergyTime.Stop();
  if(sot_running) {
    steponlytime.Start();
  }
#endif // REPORT_TIME

  if (H_fill != NULL) {
    OC_INDEX size = state.mesh->Size();
    FD_TYPE *tmp_field = new FD_TYPE[3 * size];
    memDownload_device(tmp_field, 
      host_struct.dev_field, 3 * size, DEV_NUM);
    for (OC_INDEX i = 0; i < size; i++) {
      (*H_fill)[i] = ThreeVector(tmp_field[i], tmp_field[i + size], tmp_field[i + 2 * size]);
    }
    if(tmp_field) delete[] tmp_field;
  }

  energy_state_id = state.Id();
  pE_pt = oced.pE_pt;  // Export pE_pt value
}

void Oxs_GPU_TimeEvolver::UpdateEnergyOutputs(const Oxs_SimState& state) {

  if(state.Id()==0) { // Safety
    return;
  }
  
  if (!initialized) {
    InitGPU(state);
    initialized = true;
  }
  
  Oxs_MeshValue<OC_REAL8m> energy(state.mesh);
  OC_REAL8m pE_pt;
  
  if(energy_state_id != state.Id()) {
    GPU_GetEnergyDensity(state, energy, NULL, NULL, pE_pt);
  }
  // Store total energy sum if output object total_energy_output
  // has cache enabled.
  if (total_energy_output.GetCacheRequestCount()>0) {
    total_energy_output.cache.state_id=0;
    OC_INDEX size = state.mesh->Size();
    dotProduct(size, BLK_SIZE, host_struct.dev_energy, 
      host_struct.dev_vol, host_struct.dev_local_sum);
    FD_TYPE energy_sum = sum_device(size, 
      host_struct.dev_local_sum, host_struct.dev_local_sum,
      DEV_NUM, maxGridSize, maxTotalThreads);
    total_energy_output.cache.value = energy_sum;
    total_energy_output.cache.state_id=state.Id();
  }

  if(total_energy_density_output.GetCacheRequestCount() > 0) {
    // Energy density field output requested.  Copy results
    // to output cache.
    total_energy_density_output.cache.state_id=0;
    OC_INDEX size = state.mesh->Size();
    Oxs_MeshValue<OC_REAL8m>& energyBuffer = 
      total_energy_density_output.cache.value;
    FD_TYPE *tmp_energy = new FD_TYPE[size];
    const string errorString = memDownload_device(tmp_energy, 
      host_struct.dev_energy, size, DEV_NUM);
    for (OC_INDEX i = 0; i < size; i++) {
      energyBuffer[i] = tmp_energy[i];
    }
    if(tmp_energy) delete[] tmp_energy;
    total_energy_density_output.cache.state_id=state.Id();
  }
  
  if(total_field_output.GetCacheRequestCount()>0) {
    OC_INDEX size = state.mesh->Size();
    Oxs_MeshValue<ThreeVector>& fieldBuffer = 
      total_field_output.cache.value;
    FD_TYPE *tmp_field = new FD_TYPE[3 * size];
    memDownload_device(tmp_field, 
      host_struct.dev_field, 3 * size, DEV_NUM);
    for (OC_INDEX i = 0; i < size; i++) {
      fieldBuffer[i] = ThreeVector(tmp_field[i], tmp_field[i + size],
        tmp_field[i + 2 * size]);
    }
    if(tmp_field) delete[] tmp_field;
    total_field_output.cache.state_id=state.Id();
  }
}

void Oxs_GPU_TimeEvolver::FillEnergyCalcCountOutput(const Oxs_SimState& state) {
  energy_calc_count_output.cache.state_id=state.Id();
  energy_calc_count_output.cache.value
    = static_cast<OC_REAL8m>(energy_calc_count);
}
