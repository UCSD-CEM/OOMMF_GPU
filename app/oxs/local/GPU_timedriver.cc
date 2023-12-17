/* FILE: timedriver.cc            -*-Mode: c++-*-
 *
 * Example concrete Oxs_Driver class.
 *
 */

#include <string>

#include "nb.h"
#include "GPU_timedriver.h"
#include "director.h"
#include "simstate.h"
#include "GPU_timeevolver.h"
#include "key.h"
#include "energy.h"		// Needed to make MSVC++ 5 happy
#include "scalarfield.h"

#include "GPU_helper.h"

OC_USE_STRING;

// Oxs_Ext registration support
OXS_EXT_REGISTER(Oxs_GPU_TimeDriver);

/* End includes */

// Constructor
Oxs_GPU_TimeDriver::Oxs_GPU_TimeDriver(
  const char* name,     // Child instance id
  Oxs_Director* newdtr, // App director
  const char* argstr)   // MIF input block parameters
  : Oxs_Driver(name,newdtr,argstr, scaling_aveM, normalize_aveM), 
  max_dm_dt_obj_ptr(NULL), stage_maxang_loc(-1.), run_maxang_loc(-1.)
{
  // Process arguments
  OXS_GET_INIT_EXT_OBJECT("evolver",Oxs_GPU_TimeEvolver,evolver_obj);
  evolver_key.Set(evolver_obj.GetPtr());
  // Dependency lock on Oxs_GPU_TimeEvolver object is
  // held until *this is destroyed.

  if(!HasInitValue("stopping_dm_dt")) {
    stopping_dm_dt.push_back(0.0); // Default is no control
  } else {
    GetGroupedRealListInitValue("stopping_dm_dt",stopping_dm_dt);
  }

  if(!HasInitValue("stopping_time")) {
    stopping_time.push_back(0.0); // Default is no control
  } else {
    GetGroupedRealListInitValue("stopping_time",stopping_time);
  }

  VerifyAllInitArgsUsed();

  last_timestep_output.Setup(
           this,InstanceName(),"Last time step","s",0,
	   &Oxs_GPU_TimeDriver::Fill__last_timestep_output);
  simulation_time_output.Setup(
	   this,InstanceName(),"Simulation time","s",0,
	   &Oxs_GPU_TimeDriver::Fill__simulation_time_output);

  last_timestep_output.Register(director,0);
  simulation_time_output.Register(director,0);

  spin_output.Setup(this,InstanceName(),"Spin","",1,
                    &Oxs_GPU_TimeDriver::UpdateDerivedOutputs);
  spin_output.Register(director,0);
  
  magnetization_output.Setup(this,InstanceName(),
                           "Magnetization","A/m",1,
                           &Oxs_GPU_TimeDriver::UpdateDerivedOutputs);
  magnetization_output.Register(director,0);
  
  if(normalize_aveM) {
    aveMx_output.Setup(this,InstanceName(),"mx","",1,
                       &Oxs_GPU_TimeDriver::UpdateDerivedOutputs);
    aveMy_output.Setup(this,InstanceName(),"my","",1,
                       &Oxs_GPU_TimeDriver::UpdateDerivedOutputs);
    aveMz_output.Setup(this,InstanceName(),"mz","",1,
                       &Oxs_GPU_TimeDriver::UpdateDerivedOutputs);
  } else {
    aveMx_output.Setup(this,InstanceName(),"Mx","A/m",1,
                       &Oxs_GPU_TimeDriver::UpdateDerivedOutputs);
    aveMy_output.Setup(this,InstanceName(),"My","A/m",1,
                       &Oxs_GPU_TimeDriver::UpdateDerivedOutputs);
    aveMz_output.Setup(this,InstanceName(),"Mz","A/m",1,
                       &Oxs_GPU_TimeDriver::UpdateDerivedOutputs);
  }
  
  aveMx_output.Register(director, 0);
  aveMy_output.Register(director, 0);
  aveMz_output.Register(director, 0);

  maxSpinAng_output.Setup(this,InstanceName(),"Max Spin Ang","deg",1,
                              &Oxs_GPU_TimeDriver::Fill__maxSpinAng_output);
  stage_maxSpinAng_output.Setup(this,InstanceName(),
                              "Stage Max Spin Ang","deg",1,
                              &Oxs_GPU_TimeDriver::Fill__maxSpinAng_output);
  run_maxSpinAng_output.Setup(this,InstanceName(),
                              "Run Max Spin Ang","deg",1,
                              &Oxs_GPU_TimeDriver::Fill__maxSpinAng_output);
                              
  maxSpinAng_output.Register(director,0);
  stage_maxSpinAng_output.Register(director,0);
  run_maxSpinAng_output.Register(director,0);   
  
  // Reserve space for initial state (see GetInitialState() below)
  director->ReserveSimulationStateRequest(1);
}

Oxs_ConstKey<Oxs_SimState>
Oxs_GPU_TimeDriver::GetInitialState() const
{
  Oxs_Key<Oxs_SimState> initial_state;
  director->GetNewSimulationState(initial_state);
  Oxs_SimState& istate = initial_state.GetWriteReference();
  SetStartValues(istate);
  initial_state.GetReadReference();  // Release write lock.
  /// The read lock will be automatically released when the
  /// key "initial_state" is destroyed.
  return initial_state;
}

OC_BOOL Oxs_GPU_TimeDriver::Init()
{ 
  Oxs_Driver::Init();  // Run init routine in parent.
  /// This will call Oxs_GPU_TimeDriver::GetInitialState().

  // Get pointer to output object providing max dm/dt data
  const Oxs_GPU_TimeEvolver* evolver = evolver_key.GetPtr();
  if(evolver==NULL) {
    throw Oxs_ExtError(this,"PROGRAMMING ERROR: No evolver found?");
  }
  String output_name = String(evolver->InstanceName());
  output_name += String(":Max dm/dt");
  max_dm_dt_obj_ptr
    =  director->FindOutputObjectExact(output_name.c_str());
  if(max_dm_dt_obj_ptr==NULL) {
    throw Oxs_ExtError(this,"Unable to identify unique"
                         " Max dm/dt output object");
  }

  // Adjust spin output to always use full precision
  String default_format = spin_output.GetOutputFormat();
  Nb_SplitList arglist;
  if(arglist.Split(default_format.c_str())!=TCL_OK) {
    char bit[4000];
    Oc_EllipsizeMessage(bit,sizeof(bit),default_format.c_str());
    char temp_buf[4500];
    Oc_Snprintf(temp_buf,sizeof(temp_buf),
                "Format error in spin output format string---"
                "not a proper Tcl list: %.4000s",
                bit);
    throw Oxs_ExtError(this,temp_buf);
  }
  if(arglist.Count()!=2) {
    OXS_THROW(Oxs_ProgramLogicError,
              "Wrong number of arguments in spin output format string, "
              "detected in Oxs_Driver Init");
  } else {
    vector<String> sarr;
    sarr.push_back(arglist[0]); // Data type
    if(sarr[0].compare("binary") == 0) {
      sarr.push_back("8");      // Precision
    } else {
      sarr.push_back("%.17g");
    }
    String precise_format = Nb_MergeList(&sarr);
    spin_output.SetOutputFormat(precise_format.c_str());
  }
  
  stage_maxang_loc = -1.;
  run_maxang_loc = -1.;
  host_struct.reset();
  return 1;
}

Oxs_GPU_TimeDriver::~Oxs_GPU_TimeDriver()
{}

void Oxs_GPU_TimeDriver::StageRequestCount
(unsigned int& min,
 unsigned int& max) const
{ // Number of stages wanted by driver

  Oxs_Driver::StageRequestCount(min,max);

  unsigned int count = static_cast<OC_UINT4m>(stopping_dm_dt.size());
  if(count>min) min=count;
  if(count>1 && count<max) max=count;
  // Treat length 1 lists as imposing no upper constraint.

  count =  static_cast<OC_UINT4m>(stopping_time.size());
  if(count>min) min=count;
  if(count>1 && count<max) max=count;
  // Treat length 1 lists as imposing no upper constraint.
}

OC_BOOL
Oxs_GPU_TimeDriver::ChildIsStageDone(const Oxs_SimState& state) const
{
  OC_UINT4m stage_index = state.stage_number;

  // Stage time check
  OC_REAL8m stop_time=0.;
  if(stage_index >= stopping_time.size()) {
    stop_time = stopping_time[stopping_time.size()-1];
  } else {
    stop_time = stopping_time[stage_index];
  }
  if(stop_time>0.0
     && stop_time-state.stage_elapsed_time<=stop_time*OC_REAL8_EPSILON*2) {
    return 1; // Stage done
  }
  // dm_dt check
  Tcl_Interp* mif_interp = director->GetMifInterp();
  if(max_dm_dt_obj_ptr==NULL ||
     max_dm_dt_obj_ptr->Output(&state,mif_interp,0,NULL) != TCL_OK) {
    String msg=String("Unable to obtain Max dm/dt output: ");
    if(max_dm_dt_obj_ptr==NULL) {
      msg += String("PROGRAMMING ERROR: max_dm_dt_obj_ptr not set."
		    " Driver Init() probably not called.");
    } else {
      msg += String(Tcl_GetStringResult(mif_interp));
    }
    throw Oxs_ExtError(this,msg.c_str());
  }
  OC_BOOL err;
  OC_REAL8m max_dm_dt = Nb_Atof(Tcl_GetStringResult(mif_interp),err);
  if(err) {
    String msg=String("Error detected in StageDone method"
		      " --- Invalid Max dm/dt output: ");
    msg += String(Tcl_GetStringResult(mif_interp));
    throw Oxs_ExtError(this,msg.c_str());
  }
  OC_REAL8m stop_dm_dt=0.;
  if(stage_index >= stopping_dm_dt.size()) {
    stop_dm_dt = stopping_dm_dt[stopping_dm_dt.size()-1];
  } else {
    stop_dm_dt = stopping_dm_dt[stage_index];
  }
  if(stop_dm_dt>0.0 && max_dm_dt <= stop_dm_dt) {
    return 1; // Stage done
  }
  // If control gets here, then stage not done
  return 0;
}

OC_BOOL
Oxs_GPU_TimeDriver::ChildIsRunDone(const Oxs_SimState& /* state */) const
{
  // No child-specific checks at this time...
  return 0; // Run not done
}

void Oxs_GPU_TimeDriver::FillStateSupplemental(Oxs_SimState& work_state) const
{
  OC_REAL8m work_step = work_state.last_timestep;
  OC_REAL8m base_time = work_state.stage_elapsed_time - work_step;

  // Insure that step does not go past stage stopping time
  OC_UINT4m stop_index = work_state.stage_number;
  OC_REAL8m stop_value=0.0;
  if(stop_index >= stopping_time.size()) {
    stop_value = stopping_time[stopping_time.size()-1];
  } else {
    stop_value = stopping_time[stop_index];
  }
  if(stop_value>0.0) {
    OC_REAL8m timediff = stop_value-work_state.stage_elapsed_time;
    if(timediff<=0) { // Over step
      // In the degenerate case where dm_dt=0, work_step will be
      // large (==1) and work_state.stage_elapsed_time will also
      // be large.  In that case, timediff will be numerically
      // poor because stop_value << work_state.stage_elapsed_time.
      // Check for this, and adjust sums accordingly.
      if(work_step>stop_value) { // Degenerate case
        work_step -= work_state.stage_elapsed_time;
        work_step += stop_value;
      } else {                   // Normal case
        work_step += timediff;
      }
      if(work_step<=0.0) work_step = stop_value*OC_REAL8_EPSILON; // Safety
      work_state.last_timestep = work_step;
      work_state.stage_elapsed_time = stop_value;
    } else if(timediff < 2*stop_value*OC_REAL8_EPSILON) {
      // Under step, but close enough for government work
      work_state.last_timestep += timediff;
      work_state.stage_elapsed_time = stop_value;
    } else if(0.25*work_step>timediff) {
      // Getting close to stage boundary.  Foreshorten.
      OC_REAL8m tempstep = (3*work_step+timediff)*0.25;
      work_state.last_timestep = tempstep;
      work_state.stage_elapsed_time = base_time+tempstep;
    }
  }
}

OC_BOOL
Oxs_GPU_TimeDriver::Step
(Oxs_ConstKey<Oxs_SimState> base_state,
 const Oxs_DriverStepInfo& stepinfo,
 Oxs_Key<Oxs_SimState>& next_state)
{ // Returns true if step was successful, false if
  // unable to step as requested.

  // Put write lock on evolver in order to get a non-const
  // pointer.  Use a temporary variable, temp_key, so
  // write lock is automatically removed when temp_key
  // is destroyed.
  Oxs_Key<Oxs_GPU_TimeEvolver> temp_key = evolver_key;
  Oxs_GPU_TimeEvolver& evolver = temp_key.GetWriteReference();
  return evolver.Step(this,base_state,stepinfo,next_state, host_struct);
}

OC_BOOL
Oxs_GPU_TimeDriver::InitNewStage
(Oxs_ConstKey<Oxs_SimState> state,
 Oxs_ConstKey<Oxs_SimState> prevstate)
{
  // Put write lock on evolver in order to get a non-const
  // pointer.  Use a temporary variable, temp_key, so
  // write lock is automatically removed when temp_key
  // is destroyed.
  Oxs_Key<Oxs_GPU_TimeEvolver> temp_key = evolver_key;
  Oxs_GPU_TimeEvolver& evolver = temp_key.GetWriteReference();
  return evolver.InitNewStage(this,state,prevstate);
}


////////////////////////////////////////////////////////////////////////
// State-based outputs, maintained by the driver.  These are
// conceptually public, but are specified private to force
// clients to use the output_map interface in Oxs_Director.

#define OSO_FUNC(NAME) \
void Oxs_GPU_TimeDriver::Fill__##NAME##_output(const Oxs_SimState& state) \
{ NAME##_output.cache.state_id=state.Id(); \
  NAME##_output.cache.value=state.NAME; }

OSO_FUNC(last_timestep)

void
Oxs_GPU_TimeDriver::Fill__simulation_time_output(const Oxs_SimState& state)
{
  simulation_time_output.cache.state_id = state.Id();
  simulation_time_output.cache.value =
    state.stage_start_time + state.stage_elapsed_time;
}

void Oxs_GPU_TimeDriver::UpdateDerivedOutputs(const Oxs_SimState& stateConst) {

  Oxs_SimState state = stateConst;
  const OC_INDEX size = state.spin.Size();
  const OC_BOOL aveMx_outputRequest = 
    aveMx_output.GetCacheRequestCount() > 0 && 
    aveMx_output.cache.state_id != state.Id();
  const OC_BOOL aveMy_outputRequest = 
    aveMy_output.GetCacheRequestCount() > 0 && 
    aveMy_output.cache.state_id != state.Id();
  const OC_BOOL aveMz_outputRequest = 
    aveMz_output.GetCacheRequestCount() > 0 && 
    aveMz_output.cache.state_id != state.Id();
  const OC_BOOL spin_outputRequest =
    spin_output.GetCacheRequestCount() > 0 &&
    spin_output.cache.state_id != state.Id();
  const OC_BOOL magnetization_outputRequest =
    magnetization_output.GetCacheRequestCount() > 0 &&
    magnetization_output.cache.state_id != state.Id();
    
  if (host_struct.dev_MValue != NULL &&
     (aveMx_outputRequest || aveMy_outputRequest ||
      aveMz_outputRequest || spin_outputRequest ||
      magnetization_outputRequest)) {

    FD_TYPE *tmp_spin = new FD_TYPE[3 * size];
    memDownload_device(tmp_spin, host_struct.dev_MValue, 3 * size, 
      DEV_NUM);
		for (int i = 0; i < size; i++) {
			state.spin[i].x = tmp_spin[i];
			state.spin[i].y = tmp_spin[i + size];
			state.spin[i].z = tmp_spin[i + 2 * size];
		}
		if(tmp_spin) delete[] tmp_spin;
  }
  
  if (spin_outputRequest) {
    spin_output.cache.state_id=0;
    spin_output.cache.value = state.spin;
    spin_output.cache.state_id=state.Id();
  }
  
  if (magnetization_outputRequest) {
    magnetization_output.cache.state_id=0;
    magnetization_output.cache.value.AdjustSize(state.mesh);
    const Oxs_MeshValue<ThreeVector>& spin = state.spin;
    const Oxs_MeshValue<OC_REAL8m>& sMs = *(state.Ms);
    Oxs_MeshValue<ThreeVector>& mag = magnetization_output.cache.value;
    for(OC_INDEX i=0;i<size;i++) {
      mag[i] = spin[i];
      mag[i] *= sMs[i];
    }
    magnetization_output.cache.state_id=state.Id();
  }
  
  const Oxs_MeshValue<ThreeVector>& spin = state.spin;
  const Oxs_MeshValue<OC_REAL8m>& sMs = *(state.Ms);

  if(aveMx_outputRequest && aveMy_outputRequest &&
     aveMz_outputRequest) {
    // Preferred case: All three components desired
    // This does not appear to be the usual case, however...
    aveMx_output.cache.state_id=0;
    aveMy_output.cache.state_id=0;
    aveMz_output.cache.state_id=0;
    OC_REAL8m Mx=0.0;
    OC_REAL8m My=0.0;
    OC_REAL8m Mz=0.0;
    for(OC_INDEX i=0;i<size;++i) {
      OC_REAL8m sat_mag = sMs[i];
      Mx += sat_mag*(spin[i].x);
      My += sat_mag*(spin[i].y);
      Mz += sat_mag*(spin[i].z);
    }
    aveMx_output.cache.value=Mx*scaling_aveM;
    aveMx_output.cache.state_id=state.Id();
    aveMy_output.cache.value=My*scaling_aveM;
    aveMy_output.cache.state_id=state.Id();
    aveMz_output.cache.value=Mz*scaling_aveM;
    aveMz_output.cache.state_id=state.Id();
  } else {
    // Calculate components on a case-by-case basis
    if(aveMx_outputRequest) {
      aveMx_output.cache.state_id=0;
      OC_REAL8m Mx=0.0;
      for(OC_INDEX i=0;i<size;++i) {
        Mx += sMs[i]*(spin[i].x);
      }
      aveMx_output.cache.value=Mx*scaling_aveM;
      aveMx_output.cache.state_id=state.Id();
    }

    if(aveMy_outputRequest) {
      aveMy_output.cache.state_id=0;
      OC_REAL8m My=0.0;
      for(OC_INDEX i=0;i<size;++i) {
        My += sMs[i]*(spin[i].y);
      }
      aveMy_output.cache.value=My*scaling_aveM;
      aveMy_output.cache.state_id=state.Id();
    }

    if(aveMz_outputRequest) {
      aveMz_output.cache.state_id=0;
      OC_REAL8m Mz=0.0;
      for(OC_INDEX i=0;i<size;++i) {
        Mz += sMs[i]*(spin[i].z);
      }
      aveMz_output.cache.value=Mz*scaling_aveM;
      aveMz_output.cache.state_id=state.Id();
    }
  }
}

void Oxs_GPU_TimeDriver::Fill__maxSpinAng_output(const Oxs_SimState& state) {
  
  const OC_INDEX size = state.spin.Size();
  const OC_BOOL maxSpinAng_outputRequest =
    maxSpinAng_output.GetCacheRequestCount() > 0 &&
    maxSpinAng_output.cache.state_id != state.Id();
  const OC_BOOL stage_maxSpinAng_outputRequest =
    stage_maxSpinAng_output.GetCacheRequestCount() > 0 &&
    stage_maxSpinAng_output.cache.state_id != state.Id();
  const OC_BOOL run_maxSpinAng_outputRequest =
    run_maxSpinAng_output.GetCacheRequestCount() > 0 &&
    run_maxSpinAng_output.cache.state_id != state.Id();
    
  if (maxSpinAng_outputRequest || stage_maxSpinAng_outputRequest ||
      run_maxSpinAng_outputRequest) {
 
    maxSpinAng_output.cache.state_id =
    stage_maxSpinAng_output.cache.state_id =
    run_maxSpinAng_output.cache.state_id = 0;
    
    OC_REAL8m maxang,stage_maxang,run_maxang;
    maxang = stage_maxang = run_maxang = -1.0; // Safety init
    
    // computation
    ComputeMaxAng(state, size, maxang, stage_maxang, run_maxang);
    
    maxSpinAng_output.cache.value = maxang;
    stage_maxSpinAng_output.cache.value = stage_maxang;
    run_maxSpinAng_output.cache.value = run_maxang;
    
    maxSpinAng_output.cache.state_id =
    stage_maxSpinAng_output.cache.state_id =
    run_maxSpinAng_output.cache.state_id = state.Id();
  }
}

void Oxs_GPU_TimeDriver::ComputeMaxAng(
    const Oxs_SimState& state,
    const OC_INDEX &size,
    OC_REAL8m &maxang,
    OC_REAL8m &stage_maxang,
    OC_REAL8m &run_maxang) {

  if (host_struct.dev_dot != NULL && host_struct.dev_local_sum != NULL) {
    OC_REAL8m maxdot = maxDot(host_struct.dev_dot, host_struct.dev_local_sum, 
      size, BLK_SIZE, DEV_NUM);
    // Set max angle data
    const OC_REAL8m arg = 0.5*Oc_Sqrt(maxdot);
    maxang = (arg >= 1.0 ? 180.0 : asin(arg)*(360.0/PI));
  } else {
    maxang = state.mesh->MaxNeighborAngle(state.spin,*(state.Ms))*(180./PI);
  }

  OC_REAL8m dummy_value;
  String msa_name = MaxSpinAngleStateName();
  if(state.GetDerivedData(msa_name, dummy_value)) {
    // Ideally, energy values would never be computed more than once
    // for any one state, but in practice it seems inevitable that
    // such will occur on occasion.  For example, suppose a user
    // requests output on a state obtained by a stage crossing (as
    // opposed to a state obtained through a normal intrastage step);
    // a subsequent ::Step operation will re-compute the energies
    // because not all the information needed by the step transition
    // machinery is cached from an energy computation.  Even user
    // output requests on post ::Step states is problematic if some of
    // the requested output is not cached as part of the step
    // proceedings.  A warning is put into place below for debugging
    // purposes, but in general an error is raised only if results
    // from the recomputation are different than originally.
#ifndef NDEBUG
    static Oxs_WarningMessage maxangleset(3);
    maxangleset.Send(revision_info,OC_STRINGIFY(__LINE__),
                     "Programming inefficiency?"
                     " Oxs_GPU_UniformExchange max spin angle set twice.");
#endif
    // Max angle is computed by taking acos of the dot product
    // of neighboring spin vectors.  The relative error can be
    // quite large if the spins are nearly parallel.  The proper
    // error comparison is between the cos of the two values.
    // See NOTES VI, 6-Sep-2012, p71.
    OC_REAL8m diff = (dummy_value-maxang)*(PI/180.);
    OC_REAL8m sum  = (dummy_value+maxang)*(PI/180.);
    if(sum > PI ) sum = 2*PI - sum;
    if(fabs(diff*sum)>8*OC_REAL8_EPSILON) {
      char errbuf[1024];
      Oc_Snprintf(errbuf,sizeof(errbuf),
                  "Programming error:"
                  " Oxs_GPU_TimeDriver max spin angle set to"
                  " two different values;"
                  " orig val=%#.17g, new val=%#.17g",
                  dummy_value,maxang);
      throw Oxs_ExtError(this,errbuf);
    }
  } else {
    state.AddDerivedData(msa_name, maxang);
  }

  // Run and stage angle data depend on data from the previous state.
  // In the case that the energy (and hence max stage and run angle)
  // for the current state was computed previously, then the previous
  // state may have been dropped.  So, compute and save run and stage
  // angle data iff they are not already computed.

  // Check stage and run max angle data from previous state
  const Oxs_SimState* oldstate = NULL;
  stage_maxang = -1;
  run_maxang = -1;
  String smsa_name = StageMaxSpinAngleStateName();
  String rmsa_name = RunMaxSpinAngleStateName();
  if(state.previous_state_id &&
     (oldstate
      = director->FindExistingSimulationState(state.previous_state_id)) ) {
    if(oldstate->stage_number != state.stage_number) {
      stage_maxang = 0.0;
    } else {
      stage_maxang = stage_maxang_loc;
    }
    run_maxang = run_maxang_loc;
  }
  if(stage_maxang<maxang) stage_maxang = maxang;
  if(run_maxang<maxang)   run_maxang = maxang;

  stage_maxang_loc = stage_maxang;
  run_maxang_loc = run_maxang;
  
  // Stage max angle data
  if(!state.GetDerivedData(smsa_name,dummy_value)) {
    state.AddDerivedData(smsa_name,stage_maxang);
  }

  // Run max angle data
  if(!state.GetDerivedData(rmsa_name,dummy_value)) {
    state.AddDerivedData(rmsa_name,run_maxang);
  }
}