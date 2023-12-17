/* FILE: GPU_rungekuttaevolve.cc                 -*-Mode: c++-*-
 *
 * Concrete evolver class, using Runge-Kutta steps on GPU
 * Clipped from rungekuttaevolve.cc
 */

#include <float.h>
#include <string>

#include "nb.h"
#include "director.h"
#include "GPU_timedriver.h"
#include "simstate.h"
#include "rectangularmesh.h"   // For Zhang damping
#include "GPU_rungekuttaevolve.h"
#include "key.h"
#include "energy.h"             // Needed to make MSVC++ 5 happy

#include "GPU_evolver_kernel.h"
OC_USE_STRING;

// Oxs_Ext registration support
OXS_EXT_REGISTER(Oxs_GPU_RungeKuttaEvolve);

/* End includes */

// Constructor
Oxs_GPU_RungeKuttaEvolve::Oxs_GPU_RungeKuttaEvolve(
  const char* name,     // Child instance id
  Oxs_Director* newdtr, // App director
  const char* argstr)   // MIF input block parameters
  : Oxs_GPU_TimeEvolver(name, newdtr, argstr, host_struct),
    mesh_id(0),
    max_step_decrease(0.03125), max_step_increase_limit(4.0),
    max_step_increase_adj_ratio(1.9),
    reject_goal(0.05), reject_ratio(0.05),
    energy_state_id(0),next_timestep(0.),
    rkstep_ptr(NULL), allocated(false), dev_info(NULL), 
    dev_gamma(NULL), dev_alpha(NULL), dev_MValue_backup(NULL),
    dev_MValue_backup2(NULL), dev_dm_dt_backup(NULL), 
    dev_dm_dt_backup2(NULL), dev_dm_dt_backup4(NULL) {
  // Process arguments
  min_timestep=GetRealInitValue("min_timestep",0.);
  max_timestep=GetRealInitValue("max_timestep",1e-10);
  if(max_timestep<=0.0) {
    char buf[4096];
    Oc_Snprintf(buf,sizeof(buf),
                "Invalid parameter value:"
                " Specified max time step is %g (should be >0.)",
                static_cast<double>(max_timestep));
    throw Oxs_ExtError(this,buf);
  }

  allowed_error_rate = GetRealInitValue("error_rate",-1);
  if(allowed_error_rate>0.0) {
    allowed_error_rate *= PI*1e9/180.; // Convert from deg/ns to rad/s
  }
  allowed_absolute_step_error
    = GetRealInitValue("absolute_step_error",0.2);
  if(allowed_absolute_step_error>0.0) {
    allowed_absolute_step_error *= PI/180.; // Convert from deg to rad
  }
  allowed_relative_step_error
    = GetRealInitValue("relative_step_error",0.01);

  expected_energy_precision =
    GetRealInitValue("energy_precision",1e-10);

  reject_goal = GetRealInitValue("reject_goal",0.05);
  if(reject_goal<0.) {
    throw Oxs_ExtError(this,"Invalid initialization detected:"
         " \"reject_goal\" value must be non-negative.");
  }

  min_step_headroom = GetRealInitValue("min_step_headroom",0.33);
  if(min_step_headroom<0.) {
    throw Oxs_ExtError(this,"Invalid initialization detected:"
         " \"min_step_headroom\" value must be bigger than 0.");
  }

  max_step_headroom = GetRealInitValue("max_step_headroom",0.95);
  if(max_step_headroom<0.) {
    throw Oxs_ExtError(this,"Invalid initialization detected:"
         " \"max_step_headroom\" value must be bigger than 0.");
  }

  if(min_step_headroom>max_step_headroom) {
    throw Oxs_ExtError(this,"Invalid initialization detected:"
         " \"min_step_headroom\" value must not be larger than"
         " \"max_step_headroom\".");
  }


  if(HasInitValue("alpha")) {
    OXS_GET_INIT_EXT_OBJECT("alpha",Oxs_ScalarField,alpha_init);
  } else {
    alpha_init.SetAsOwner(dynamic_cast<Oxs_ScalarField *>
                          (MakeNew("Oxs_UniformScalarField",director,
                                   "value 0.5")));
  }

  // User may specify either gamma_G (Gilbert) or
  // gamma_LL (Landau-Lifshitz).
  gamma_style = GS_INVALID;
  if(HasInitValue("gamma_G") && HasInitValue("gamma_LL")) {
    throw Oxs_ExtError(this,"Invalid Specify block; "
                         "both gamma_G and gamma_LL specified.");
  } else if(HasInitValue("gamma_G")) {
    gamma_style = GS_G;
    OXS_GET_INIT_EXT_OBJECT("gamma_G",Oxs_ScalarField,gamma_init);
  } else if(HasInitValue("gamma_LL")) {
    gamma_style = GS_LL;
    OXS_GET_INIT_EXT_OBJECT("gamma_LL",Oxs_ScalarField,gamma_init);
  } else {
    gamma_style = GS_G;
    gamma_init.SetAsOwner(dynamic_cast<Oxs_ScalarField *>
                          (MakeNew("Oxs_UniformScalarField",director,
                                   "value -2.211e5")));
  }
  allow_signed_gamma = GetIntInitValue("allow_signed_gamma",0);
  do_precess = GetIntInitValue("do_precess",1);

  start_dm = GetRealInitValue("start_dm",0.01);
  start_dm *= PI/180.; // Convert from deg to rad

  start_dt = GetRealInitValue("start_dt",max_timestep/8.);

  if(start_dm<0. && start_dt<0.) {
    throw Oxs_ExtError(this,"Invalid initialization detected:"
                       " at least one of \"start_dm\" and \"start_dt\""
                       "  must be nonnegative.");
  }

  stage_init_step_control = SISC_AUTO;  // Safety
  String stage_start = GetStringInitValue("stage_start","auto");
  Oxs_ToLower(stage_start);
  if(stage_start.compare("start_conditions")==0) {
    stage_init_step_control = SISC_START_COND;
  } else if(stage_start.compare("continuous")==0) {
    stage_init_step_control = SISC_CONTINUOUS;
  } else if(stage_start.compare("auto")==0
            || stage_start.compare("automatic")==0) {
    stage_init_step_control = SISC_AUTO;
  } else {
    throw Oxs_ExtError(this,"Invalid initialization detected:"
                         " \"stage_start\" value must be one of"
                         " start_conditions, continuous, or auto.");
  }

  String method = GetStringInitValue("method","rkf54");
  Oxs_ToLower(method); // Do case insensitive match
  if(method.compare("rk2")==0) {
    rkstep_ptr = &Oxs_GPU_RungeKuttaEvolve::TakeRungeKuttaStep2;
  } else if(method.compare("rk2heun")==0) {
    rkstep_ptr = &Oxs_GPU_RungeKuttaEvolve::TakeRungeKuttaStep2Heun;
  } else if(method.compare("rk4")==0) {
    rkstep_ptr = &Oxs_GPU_RungeKuttaEvolve::TakeRungeKuttaStep4;
  } else if(method.compare("rkf54m")==0) {
    rkstep_ptr = &Oxs_GPU_RungeKuttaEvolve::TakeRungeKuttaFehlbergStep54M;
  } else if(method.compare("rkf54s")==0) {
    rkstep_ptr = &Oxs_GPU_RungeKuttaEvolve::TakeRungeKuttaFehlbergStep54S;
  } else if(method.compare("rkf54")==0) {
    rkstep_ptr = &Oxs_GPU_RungeKuttaEvolve::TakeRungeKuttaFehlbergStep54;
  } else {
    throw Oxs_ExtError(this,"Invalid initialization detected:"
                         " \"method\" value must be one of"
                         " rk2, rk2heun, rk4, rkf54, rkf54m, or rkf54s.");
  }

  // Setup outputs
  max_dm_dt_output.Setup(this,InstanceName(),"Max dm/dt","deg/ns",0,
     &Oxs_GPU_RungeKuttaEvolve::UpdateDerivedOutputs);
  dE_dt_output.Setup(this,InstanceName(),"dE/dt","J/s",0,
     &Oxs_GPU_RungeKuttaEvolve::UpdateDerivedOutputs);
  delta_E_output.Setup(this,InstanceName(),"Delta E","J",0,
     &Oxs_GPU_RungeKuttaEvolve::UpdateDerivedOutputs);
  dm_dt_output.Setup(this,InstanceName(),"dm/dt","rad/s",1,
     &Oxs_GPU_RungeKuttaEvolve::UpdateDerivedOutputs);
  mxH_output.Setup(this,InstanceName(),"mxH","A/m",1,
     &Oxs_GPU_RungeKuttaEvolve::UpdateDerivedOutputs);

  max_dm_dt_output.Register(director,-5);
  dE_dt_output.Register(director,-5);
  delta_E_output.Register(director,-5);
  dm_dt_output.Register(director,-5);
  mxH_output.Register(director,-5);

  // dm_dt and mxH output caches are used for intermediate storage,
  // so enable caching.
  // dm_dt_output.CacheRequestIncrement(1);
  // mxH_output.CacheRequestIncrement(1);

  VerifyAllInitArgsUsed();

  // Reserve space for temp_state; see Step() method below
  director->ReserveSimulationStateRequest(1);

#if REPORT_TIME_RKDEVEL
  timer.resize(10);
  timer_counts.resize(10);
#endif
}

OC_BOOL Oxs_GPU_RungeKuttaEvolve::Init() {
  Oxs_GPU_TimeEvolver::Init();

#if REPORT_TIME_RKDEVEL
  Oc_TimeVal cpu,wall;
  for(unsigned int ti=0;ti<timer.size();++ti) {
    timer[ti].GetTimes(cpu,wall);
    if(double(wall)>0.0) {
      fprintf(stderr,"               timer %2u ...   %7.2f cpu /%7.2f wall,"
              " (%s/%s)\n",
              ti,double(cpu),double(wall),InstanceName(),
              timer_counts[ti].name.c_str());
      if(timer_counts[ti].pass_count>0) {
        fprintf(stderr,"                \\---> passes = %d,"
                " bytes=%.2f MB, %.2f GB/sec\n",
                timer_counts[ti].pass_count,
                double(timer_counts[ti].bytes)/double(1024*1024),
                double(timer_counts[ti].bytes)
                /(double(1024*1024*1024)*double(wall)));
      }
    }
    timer[ti].Reset(); 
    timer_counts[ti].Reset();
  }
#endif // REPORT_TIME_RKDEVEL

  // Free scratch space allocated by previous problem (if any)
  vtmpA.Release();   vtmpB.Release();
  vtmpC.Release();   vtmpD.Release();

  // Free memory used by LLG gamma and alpha
  mesh_id = 0;
  alpha.Release();
  gamma.Release();

  max_step_increase = max_step_increase_limit;

  // Setup step_headroom and reject_ratio
  step_headroom = max_step_headroom;
  reject_ratio = reject_goal;

  energy_state_id=0;   // Mark as invalid state
  next_timestep=0.;    // Dummy value
  return 1;
}

OC_BOOL Oxs_GPU_RungeKuttaEvolve::InitGPU(const OC_INDEX &size) {
  cudaSetDevice(DEV_NUM);
  alloc_device(dev_info, 10, DEV_NUM, "dev_info");
  alloc_device(dev_gamma, size, DEV_NUM, "dev_gamma");
  alloc_device(dev_alpha, size, DEV_NUM, "dev_alpha");
  alloc_device(dev_MValue_backup, 3 * size, DEV_NUM, "dev_MValue_backup");
  alloc_device(dev_dm_dt_backup, 3 * size, DEV_NUM, "dev_dm_dt_backup");
  getFlatKernelSize(size, BLK_SIZE, grid_size, block_size);
  reduce_size = grid_size.x * grid_size.y * grid_size.z;
  
  return (allocated = true);
}

OC_BOOL Oxs_GPU_RungeKuttaEvolve::ReleaseGPU() {
  release_device(dev_info, DEV_NUM, "dev_info");
  release_device(dev_gamma, DEV_NUM, "dev_gamma");
  release_device(dev_alpha, DEV_NUM, "dev_alpha");
  release_device(dev_MValue_backup, DEV_NUM, "dev_MValue_backup");
  release_device(dev_MValue_backup2, DEV_NUM, "dev_MValue_backup2");
  release_device(dev_dm_dt_backup, DEV_NUM, "dev_dm_dt_backup");
  release_device(dev_dm_dt_backup2, DEV_NUM, "dev_dm_dt_backup2");
  release_device(dev_dm_dt_backup4, DEV_NUM, "dev_dm_dt_backup2");
  return true;
}

Oxs_GPU_RungeKuttaEvolve::~Oxs_GPU_RungeKuttaEvolve() {
  
  ReleaseGPU();
  
#if REPORT_TIME_RKDEVEL
  Oc_TimeVal cpu,wall;
  for(unsigned int ti=0;ti<timer.size();++ti) {
    timer[ti].GetTimes(cpu,wall);
    if(double(wall)>0.0) {
      fprintf(stderr,"               timer %2u ...   %7.2f cpu /%7.2f wall,"
              " (%s/%s)\n",
              ti,double(cpu),double(wall),InstanceName(),
              timer_counts[ti].name.c_str());
      if(timer_counts[ti].pass_count>0) {
        fprintf(stderr,"                \\---> passes = %d,"
                " bytes=%.2f MB, %.2f GB/sec\n",
                timer_counts[ti].pass_count,
                double(timer_counts[ti].bytes)/double(1024*1024),
                double(timer_counts[ti].bytes)
                /(double(1024*1024*1024)*double(wall)));
      }
    }
 }
#endif // REPORT_TIME_RKDEVEL
}

void Oxs_GPU_RungeKuttaEvolve::UpdateMeshArrays(const Oxs_Mesh* mesh) {
  mesh_id = 0; // Mark update in progress
  const OC_INDEX size = mesh->Size();
  OC_INDEX i;

  alpha_init->FillMeshValue(mesh,alpha);
  gamma_init->FillMeshValue(mesh,gamma);

  if(gamma_style == GS_G) { // Convert to LL form
    for(i=0;i<size;++i) {
      OC_REAL8m cell_alpha = alpha[i];
      gamma[i] /= (1+cell_alpha*cell_alpha);
    }
  }
  if(!allow_signed_gamma) {
    for(i=0;i<size;++i) gamma[i] = -1*fabs(gamma[i]);
  }

  if (allow_signed_gamma) {
    throw Oxs_ExtError(this, "signed gamma is unexpected for GPU implementation.");
  }
  FD_TYPE *tmpArray = new FD_TYPE[size];
  for (i = 0; i < size; i++) {
    tmpArray[i] = gamma[i];
  }
  memUpload_device(dev_gamma, tmpArray, size, DEV_NUM);
  for (i = 0; i < size; i++) {
    tmpArray[i] = -alpha[i];
  }
  memUpload_device(dev_alpha, tmpArray, size, DEV_NUM);
  
  if(tmpArray) {
    delete[] tmpArray;
  }
  
  mesh_id = mesh->Id();
}

OC_REAL8m Oxs_GPU_RungeKuttaEvolve::PositiveTimestepBound(OC_REAL8m max_dm_dt) { // Computes an estimate on the minimum time needed to
  // change the magnetization state, subject to floating
  // points limits.
#ifdef CHOOSESINGLE
  #define EPS OC_REAL4_EPSILON
  OC_REAL8m min_timestep = FLT_MAX/64.;
#elif defined(CHOOSEDOUBLE)
  #define EPS OC_REAL8_EPSILON
  OC_REAL8m min_timestep = DBL_MAX/64.;
#endif
  
  if(max_dm_dt>1 || EPS<min_timestep*max_dm_dt) {
    min_timestep = EPS/max_dm_dt;
    min_timestep *= 64;
    // A timestep of size min_timestep will be hopelessly lost
    // in roundoff error.  So increase a bit, based on an empirical
    // fudge factor.  This fudge factor can be tested by running a
    // problem with start_dm = 0.  If the evolver can't climb its
    // way out of the stepsize=0 hole, then this fudge factor is too
    // small.  So far, the most challenging examples have been
    // problems with small cells with nearly aligned spins, e.g., in
    // a remanent state with an external field is applied at t=0.
    // Damping ratio doesn't seem to have much effect, either way.
  } else {
    // Degenerate case: max_dm_dt_ must be exactly or very nearly
    // zero.  Punt.
    min_timestep = 1.0;
  }
#ifdef EPS
  #undef EPS
#endif
  return min_timestep;
}

void Oxs_GPU_RungeKuttaEvolve::Calculate_dm_dt
(const Oxs_SimState& state_,
 const Oxs_MeshValue<ThreeVector>& mxH_,
 OC_REAL8m pE_pt_,
 Oxs_MeshValue<ThreeVector>& dm_dt_,
 OC_REAL8m& max_dm_dt_,OC_REAL8m& dE_dt_,OC_REAL8m& min_timestep_export,
 const OC_BOOL &copyMemory) {
  // Imports: state, mxH_, pE_pt_
  // Exports: dm_dt_, max_dm_dt_, dE_dt_
  // NOTE: dm_dt_ is allowed, and in fact is encouraged,
  //   to be the same as mxH_.  In this case, mxH_ is
  //   overwritten by dm_dt on return.
  const Oxs_Mesh* mesh_ = state_.mesh;
  const Oxs_MeshValue<OC_REAL8m>& Ms_ = *(state_.Ms);
  const Oxs_MeshValue<ThreeVector>& spin_ = state_.spin;
  const OC_INDEX size = mesh_->Size(); // Assume import data are compatible

  // Fill out alpha and gamma meshvalue arrays, as necessary.
  if(mesh_id != mesh_->Id() || !gamma.CheckMesh(mesh_)
     || !alpha.CheckMesh(mesh_)) {
    UpdateMeshArrays(mesh_);
  }

  dm_dt_.AdjustSize(mesh_);  // For case &dm_dt_ != &mxH_

  dm_dt(grid_size, block_size, size, 0, 0, do_precess, host_struct, dev_gamma,
    dev_alpha);
  
  FD_TYPE *dev_max_dm_dt_sq = dev_info;
  FD_TYPE *dev_dE_dt_sum = dev_info + 1;
  collectDmDtStatistics(grid_size, block_size, size, reduce_size, BLK_SIZE,
    host_struct, dev_dE_dt_sum, dev_max_dm_dt_sq, dev_gamma, dev_alpha);
    
  //////////////////////////////CPU/////////////////////////////////
  // OC_REAL8m dE_dt_sum=0.0;
  // OC_REAL8m max_dm_dt_sq = 0.0;
  
  // // Canonical case: no Zhang damping
  // OC_INDEX i;
  // for(i=0;i<size;i++) { // ***ALREADY DEFINED ON GPU***
    // if(Ms_[i]==0.0) {
      // dm_dt_[i].Set(0,0,0);
    // } else {
      // OC_REAL8m coef1 = gamma[i];
      // OC_REAL8m coef2 = -1*alpha[i]*coef1;
      // if(!do_precess) coef1 = 0.0;

      // ThreeVector scratch1 = mxH_[i];
      // ThreeVector scratch2 = mxH_[i];
      // // Note: mxH may be same as dm_dt

      // scratch1 *= coef1;   // coef1 == 0 if do_precess if false
      // OC_REAL8m mxH_sq = scratch2.MagSq();
      // OC_REAL8m dm_dt_sq = mxH_sq*(coef1*coef1+coef2*coef2);
      // if(dm_dt_sq>max_dm_dt_sq) {
        // max_dm_dt_sq = dm_dt_sq;
      // }
      // dE_dt_sum += mxH_sq * Ms_[i] * mesh_->Volume(i) * coef2;
      // scratch2 ^= spin_[i]; // ((mxH)xm)
      // scratch2 *= coef2;  // -alpha.gamma((mxH)xm) = alpha.gamma(mx(mxH))
      // dm_dt_[i] = scratch1 + scratch2;
    // }
  // }
  //////////////////////////////CPU/////////////////////////////////
  
  if (copyMemory) {
    FD_TYPE host_dm_dt_info[2];

    memDownload_device(host_dm_dt_info, dev_info, 2, DEV_NUM);
    const OC_REAL8m max_dm_dt_sq = host_dm_dt_info[0];
    const OC_REAL8m dE_dt_sum = host_dm_dt_info[1];

    max_dm_dt_ = sqrt(max_dm_dt_sq);
    dE_dt_ = -1 * MU0 * dE_dt_sum + pE_pt_;
    /// The first term is (partial E/partial M)*dM/dt, the
    /// second term is (partial E/partial t)*dt/dt.  Note that,
    /// provided Ms_[i]>=0, that by constructions dE_dt_sum above
    /// is always non-negative, so dE_dt_ can only be made positive
    /// by positive pE_pt_.

    // Get bound on smallest stepsize that would actually
    // change spin new_max_dm_dt_index:
    min_timestep_export = PositiveTimestepBound(max_dm_dt_);
  }

  return;
}

void Oxs_GPU_RungeKuttaEvolve::CheckCache(const Oxs_SimState& cstate) {
  // Pull cached values out from cstate.
  // If cstate.Id() == energy_state_id, then cstate has been run
  // through either this method or UpdateDerivedOutputs.  Either
  // way, all derived state data should be stored in cstate,
  // except currently the "energy" mesh value array, which is
  // stored independently inside *this.  Eventually that should
  // probably be moved in some fashion into cstate too.
  if(energy_state_id != cstate.Id()) {
    // cached data out-of-date
    UpdateDerivedOutputs(cstate);
  }
  OC_BOOL cache_good = 1;
  OC_REAL8m max_dm_dt,dE_dt,delta_E,pE_pt;
  OC_REAL8m timestep_lower_bound;  // Smallest timestep that can actually
  /// change spin with max_dm_dt (due to OC_REAL8_EPSILON restrictions).
  /// The next timestep is based on the error from the last step.  If
  /// there is no last step (either because this is the first step,
  /// or because the last state handled by this routine is different
  /// from the incoming current_state), then timestep is calculated
  /// so that max_dm_dt * timestep = start_dm.

  cache_good &= cstate.GetDerivedData("Max dm/dt",max_dm_dt);
  cache_good &= cstate.GetDerivedData("dE/dt",dE_dt);
  cache_good &= cstate.GetDerivedData("Delta E",delta_E);
  cache_good &= cstate.GetDerivedData("pE/pt",pE_pt);
  cache_good &= cstate.GetDerivedData("Timestep lower bound",
                                      timestep_lower_bound);
  cache_good &= (energy_state_id == cstate.Id());
  // cache_good &= (dm_dt_output.cache.state_id == cstate.Id());

  if(!cache_good) {
    throw Oxs_ExtError(this,
       "Oxs_GPU_RungeKuttaEvolve::CheckCache: Invalid data cache.");
  }
}

void Oxs_GPU_RungeKuttaEvolve::AdjustState
(OC_REAL8m hstep,
 OC_REAL8m mstep,
 const Oxs_SimState& old_state,
 const Oxs_MeshValue<ThreeVector>& dm_dt,
 Oxs_SimState& new_state,
 OC_REAL8m& norm_error, 
 const FD_TYPE *dev_MValue_old, FD_TYPE *dev_MValue_new,
 const OC_BOOL &computeError) const {
  new_state.ClearDerivedData();

  if (dev_MValue_old == NULL || dev_MValue_new == NULL) {
      throw Oxs_ExtError(this,
                         "Oxs_GPU_RungeKuttaEvolve::AdjustState:"
                         " dev_MValue_old or dev_MValue_new is not initialized.");
  }
  
  const Oxs_MeshValue<ThreeVector>& old_spin = old_state.spin;
  // Oxs_MeshValue<ThreeVector>& new_spin = new_state.spin;
  
  // new_spin.AdjustSize(old_state.mesh);
  const OC_INDEX size = old_state.mesh->Size();

  // OC_REAL8m min_normsq = DBL_MAX;
  // OC_REAL8m max_normsq = 0.0;
  // ThreeVector tempspin;
  // OC_INDEX i;
  // for(i=0;i<size;++i) { // ***ALREADY DEFINED ON GPU***
    // tempspin = old_spin[i];
    // tempspin.Accum(mstep,dm_dt[i]);
    // OC_REAL8m magsq = tempspin.MakeUnit();
    // new_spin[i] = tempspin;
    // if(magsq<min_normsq) min_normsq=magsq;
    // if(magsq>max_normsq) max_normsq=magsq;
  // }
  FD_TYPE *dev_min_magsq = dev_info + 6;
  FD_TYPE *dev_max_magsq = dev_info + 7;
  adjustSpin(grid_size, block_size, size, reduce_size, BLK_SIZE, mstep, 
    host_struct, dev_MValue_old, dev_MValue_new, dev_min_magsq, dev_max_magsq);

  if (computeError) {
    FD_TYPE host_magsq_info[2];
    memDownload_device(host_magsq_info, dev_info + 6, 2, DEV_NUM);
    const OC_REAL8m min_normsq = host_magsq_info[0];
    const OC_REAL8m max_normsq = host_magsq_info[1];
    
    norm_error = OC_MAX(sqrt(max_normsq)-1.0,
                        1.0 - sqrt(min_normsq));
  }

  // Adjust time and iteration fields in new_state
  new_state.last_timestep=hstep;
  if(old_state.stage_number != new_state.stage_number) {
    // New stage
    new_state.stage_start_time = old_state.stage_start_time
                                + old_state.stage_elapsed_time;
    new_state.stage_elapsed_time = new_state.last_timestep;
  } else {
    new_state.stage_start_time = old_state.stage_start_time;
    new_state.stage_elapsed_time = old_state.stage_elapsed_time
                                  + new_state.last_timestep;
  }

  // Don't touch iteration counts. (?!)  The problem is that one call
  // to Oxs_GPU_RungeKuttaEvolve::Step() takes 2 half-steps, and it is the
  // result from these half-steps that are used as the export state.
  // If we increment the iteration count each time through here, then
  // the iteration count goes up by 2 for each call to Step().  So
  // instead, we leave iteration count at whatever value was filled
  // in by the Oxs_GPU_RungeKuttaEvolve::NegotiateTimeStep() method.
}

void Oxs_GPU_RungeKuttaEvolve::UpdateTimeFields
(const Oxs_SimState& cstate,
 Oxs_SimState& nstate,
 OC_REAL8m stepsize) const {
  nstate.last_timestep=stepsize;
  if(cstate.stage_number != nstate.stage_number) {
    // New stage
    nstate.stage_start_time = cstate.stage_start_time
                              + cstate.stage_elapsed_time;
    nstate.stage_elapsed_time = stepsize;
  } else {
    nstate.stage_start_time = cstate.stage_start_time;
    nstate.stage_elapsed_time = cstate.stage_elapsed_time
                                + stepsize;
  }
}

void Oxs_GPU_RungeKuttaEvolve::NegotiateTimeStep
(const Oxs_GPU_TimeDriver* driver,
 const Oxs_SimState&  cstate,
 Oxs_SimState& nstate,
 OC_REAL8m stepsize,
 OC_BOOL use_start_cond,
 OC_BOOL& force_step,
 OC_BOOL& driver_set_step) const { // This routine negotiates with driver over the proper step size.
  // As a side-effect, also initializes the nstate data structure,
  // where nstate is the "next state".

  // Pull needed cached values out from cstate.
  OC_REAL8m max_dm_dt;
  if(!cstate.GetDerivedData("Max dm/dt",max_dm_dt)) {
    throw Oxs_ExtError(this,
       "Oxs_GPU_RungeKuttaEvolve::NegotiateTimeStep: max_dm_dt not cached.");
  }
  OC_REAL8m timestep_lower_bound=0.;  // Smallest timestep that can actually
  /// change spin with max_dm_dt (due to OC_REAL8_EPSILON restrictions).
  if(!cstate.GetDerivedData("Timestep lower bound",
                            timestep_lower_bound)) {
    throw Oxs_ExtError(this,
       "Oxs_GPU_RungeKuttaEvolve::NegotiateTimeStep: "
       " timestep_lower_bound not cached.");
  }
  if(timestep_lower_bound<=0.0) {
    throw Oxs_ExtError(this,
       "Oxs_GPU_RungeKuttaEvolve::NegotiateTimeStep: "
       " cached timestep_lower_bound value not positive.");
  }

  // The next timestep is based on the error from the last step.  If
  // there is no last step (either because this is the first step,
  // or because the last state handled by this routine is different
  // from the incoming current_state), then timestep is calculated
  // so that max_dm_dt * timestep = start_dm or
  // timestep = start_dt.
  if(use_start_cond || stepsize<=0.0) {
    if(start_dm>=0.0) {
#ifdef CHOOSEDOUBLE
    if(start_dm < sqrt(DBL_MAX/4) * max_dm_dt) {
#endif
#ifdef CHOOSESINGLE
    if(start_dm < sqrt(FLT_MAX/4) * max_dm_dt) {
#endif
        stepsize = step_headroom * start_dm / max_dm_dt;
      } else {
#ifdef CHOOSEDOUBLE
      stepsize = sqrt(DBL_MAX/4);
#endif
#ifdef CHOOSESINGLE
      stepsize = sqrt(FLT_MAX/4);
#endif
      }
    }
    if(start_dt>=0.0) {
      if(start_dm<0.0 || stepsize>start_dt) {
        stepsize = start_dt;
      }
    }
  }

  // Insure step is not outside requested step bounds
  if(!use_start_cond && stepsize<timestep_lower_bound) {
    // Check for this before max_timestep, so max_timestep can
    // override.  On the one hand, if the timestep is too small to
    // move any spins, then taking the step just advances the
    // simulation time without changing the state; instead what would
    // be small accumulations are just lost.  On the other hand, if
    // the applied field is changing with time, then perhaps all that
    // is needed to get the magnetization moving is to advance the
    // simulation time.  In general, it is hard to tell which is
    // which, so we assume that the user knews what he was doing when
    // he set the max_timestep value (or accepted the default), and it
    // is up to him to notice if the simulation time is advancing
    // without any changes to the magnetization pattern.
    stepsize = timestep_lower_bound;
  }
  if(stepsize>max_timestep) stepsize = max_timestep;
  if(stepsize<min_timestep) stepsize = min_timestep;

  // Negotiate with driver over size of next step
  driver->FillStateMemberData(cstate,nstate);
  UpdateTimeFields(cstate,nstate,stepsize);

  // Update iteration count
  nstate.iteration_count = cstate.iteration_count + 1;
  nstate.stage_iteration_count = cstate.stage_iteration_count + 1;

  // Additional timestep control
  driver->FillStateSupplemental(nstate);

  // Check for forced step.
  // Note: The driver->FillStateSupplemental call may increase the
  //       timestep slightly to match a stage time termination
  //       criteria.  We should tweak the timestep size check
  //       to recognize that changes smaller than a few epsilon
  //       of the stage time are below our effective timescale
  //       resolution.
  force_step = 0;
#ifdef CHOOSESINGLE
  #define EPS OC_REAL4_EPSILON
#elif defined(CHOOSEDOUBLE)
  #define EPS OC_REAL8_EPSILON
#endif
  OC_REAL8m timestepcheck = nstate.last_timestep
                         - 4*EPS*nstate.stage_elapsed_time;
#ifdef EPS
  #undef EPS
#endif
  if(timestepcheck<=min_timestep || timestepcheck<=timestep_lower_bound) {
    // Either driver wants to force stepsize, or else step size can't
    // be reduced any further.
    force_step=1;
  }

  // Is driver responsible for stepsize?
  if(nstate.last_timestep == stepsize) driver_set_step = 0;
  else                                 driver_set_step = 1;
}

OC_BOOL Oxs_GPU_RungeKuttaEvolve::CheckError
(OC_REAL8m global_error_order,
 OC_REAL8m error,
 OC_REAL8m stepsize,
 OC_REAL8m reference_stepsize,
 OC_REAL8m max_dm_dt,
 OC_REAL8m& new_stepsize) { // Returns 1 if step is good, 0 if error is too large.
  // Export new_stepsize is set to suggested stepsize
  // for next step.
  //
  // new_stepsize is initially filled with a relative stepsize
  // adjustment ratio, e.g., 1.0 means no change in stepsize.
  // At the end of this routine this ratio is multiplied by
  // stepsize to get the actual absolute stepsize.
  //
  // The import stepsize is the size (in seconds) of the actual
  // step.  The new_stepsize is computed from this based on the
  // various error estimates.  An upper bound is placed on the
  // size of new_stepsize relative to the imports stepsize and
  // reference_stepsize.  reference_stepsize has no effect if
  // it is smaller than max_step_increase*stepsize.  It is
  // usually used only in the case where the stepsize was
  // artificially reduced by, for example, a stage stopping
  // criterion.
  //
  // NOTE: This routine assumes the local error order is
  //     global_error_order + 1.
  //
  // NOTE: A copy of this routine lives in eulerevolve.cc.  Updates
  //     should be copied there.
  OC_BOOL good_step = 1;
  OC_BOOL error_checked=0;

  if(allowed_relative_step_error>=0. || allowed_error_rate>=0.) {
    // Determine tighter rate bound.
    OC_REAL8m rate_error = 0.0;
    if(allowed_relative_step_error<0.) {
      rate_error = allowed_error_rate;
    } else if(allowed_error_rate<0.) {
      rate_error = allowed_relative_step_error * max_dm_dt;
    } else {
      rate_error = allowed_relative_step_error * max_dm_dt;
      if(rate_error>allowed_error_rate) {
        rate_error = allowed_error_rate;
      }
    }
    rate_error *= stepsize;

    // Rate check
    if(error>rate_error) {
      good_step = 0;
      new_stepsize = pow(rate_error/error,1.0/global_error_order);
    } else {
      #ifdef CHOOSEDOUBLE
        OC_REAL8m ratio = 0.125*DBL_MAX;
      #elif defined(CHOOSESINGLE)
        OC_REAL8m ratio = 0.125*FLT_MAX;
      #endif
      if(error>=1 || rate_error<ratio*error) {
        OC_REAL8m test_ratio = rate_error/error;
        if(test_ratio<ratio) ratio = test_ratio;
      }
      new_stepsize = pow(ratio,1.0/global_error_order);
    }
    error_checked = 1;
  }
  
  // Absolute error check
  if(allowed_absolute_step_error>=0.0) {
    OC_REAL8m test_stepsize = 0.0;
    OC_REAL8m local_error_order = global_error_order + 1.0;
    if(error>allowed_absolute_step_error) {
      good_step = 0;
      test_stepsize = pow(allowed_absolute_step_error/error,
                          1.0/local_error_order);
    } else {
      #ifdef CHOOSEDOUBLE
        OC_REAL8m ratio = 0.125*DBL_MAX;
      #elif defined(CHOOSESINGLE)
        OC_REAL8m ratio = 0.125*FLT_MAX;
      #endif
      if(error>=1 || allowed_absolute_step_error<ratio*error) {
        OC_REAL8m test_ratio = allowed_absolute_step_error/error;
        if(test_ratio<ratio) ratio = test_ratio;
      }
      test_stepsize = pow(ratio,1.0/local_error_order);
    }
    if(!error_checked || test_stepsize<new_stepsize) {
      new_stepsize = test_stepsize;
    }
    error_checked = 1;
  }
  if(error_checked) {
    new_stepsize *= step_headroom;
    if(new_stepsize<max_step_decrease) {
      new_stepsize = max_step_decrease*stepsize;
    } else {
      new_stepsize *= stepsize;
      OC_REAL8m step_bound = stepsize * max_step_increase;
      const OC_REAL8m refrat = 0.85;  // Ad hoc value
      if(stepsize<reference_stepsize*refrat) {
        step_bound = OC_MIN(step_bound,reference_stepsize);
      } else if(stepsize<reference_stepsize) {
        OC_REAL8m ref_bound = reference_stepsize + (max_step_increase-1)
          *(stepsize-reference_stepsize*refrat)/(1-refrat);
        /// If stepsize = reference_stepsize*refrat,
        ///     then ref_bound = reference_stepsize
        /// If stepsize = reference_stepsize,
        ///     then ref_bound = step_bound
        /// Otherwise, linear interpolation on stepsize.
        step_bound = OC_MIN(step_bound,ref_bound);
      }
      if(new_stepsize>step_bound) {
        new_stepsize = step_bound;
      }
    }
  } else {
    new_stepsize = stepsize;
  }
  return good_step;
}

void Oxs_GPU_RungeKuttaEvolve::TakeRungeKuttaStep2
(OC_REAL8m stepsize,
 Oxs_ConstKey<Oxs_SimState> current_state_key,
 const Oxs_MeshValue<ThreeVector>& current_dm_dt,
 Oxs_Key<Oxs_SimState>& next_state_key,
 OC_REAL8m& error_estimate,OC_REAL8m& global_error_order,
 OC_REAL8m& norm_error,
 OC_BOOL& new_energy_and_dmdt_computed) {
  // This routine performs a second order Runge-Kutta step, with
  // error estimation.  The form used is the "modified Euler"
  // method due to Collatz (1960).
  //
  // A general RK2 step involves
  //
  //  dm_dt1 = dm_dt(t1,m1)
  //  dm_dt2 = dm_dt(t1+a.h,m1+a.h.dm_dt1)
  //  m2 = m1 + h.( (1-1/(2a)).dm_dt1 + (1/(2a)).dm_dt2 )
  //
  // where 0<a<=1 is a free parameter.  Taking a=1/2 gives
  //
  //  dm_dt1 = dm_dt(t1,m1)
  //  dm_dt2 = dm_dt(t1+h/2,m1+dm_dt1.h/2)
  //  m2 = m1 + dm_dt2*h + O(h^3)
  //
  // This is the "modified Euler" method from Collatz (1960).
  // Setting a=1 yields
  //
  //  dm_dt1 = dm_dt(t1,m1)
  //  dm_dt2 = dm_dt(t1+h,m1+dm_dt1.h)
  //  m2 = m1 + (dm_dt1 + dm_dt2).h/2 + O(h^3)
  //
  // which is the method of Heun (1900).  For details see
  // J. Stoer and R. Bulirsch, "Introduction to Numerical
  // Analysis," Springer 1993, Section 7.2.1, p438.
  //
  // In the code below, we use the "modified Euler" approach,
  // i.e., select a=1/2.
  
  const Oxs_SimState* cstate = &(current_state_key.GetReadReference());

  OC_INDEX i;
  const OC_INDEX size = cstate->mesh->Size();

  backUpEnergy(host_struct.dev_energy_bak, host_struct.dev_energy, size);
  backUpEnergy(dev_MValue_backup, host_struct.dev_MValue, 3 * size);
  backUpEnergy(dev_dm_dt_backup, host_struct.dev_dm_dt, 3 * size);

  OC_REAL8m pE_pt, max_dm_dt, dE_dt, timestep_lower_bound, dummy_error;

  // Calculate dm_dt2
  AdjustState(stepsize/2, stepsize/2, *cstate, current_dm_dt,
              next_state_key.GetWriteReference(), dummy_error, 
              dev_MValue_backup, host_struct.dev_MValue, false);
  GPU_GetEnergyDensity(next_state_key.GetReadReference(), temp_energy,
                   &vtmpA, NULL, pE_pt);
  Calculate_dm_dt(next_state_key.GetReadReference(), vtmpA, pE_pt,
                  vtmpA, max_dm_dt, dE_dt, timestep_lower_bound, false); // Since
  // all the exports except dm_dt is not used, we don't copy GPU data here
  // host_struct.dev_dm_dt, instead of vtmpA, holds dm_dt2
  Oxs_SimState* nstate = &(next_state_key.GetWriteReference());
  nstate->ClearDerivedData();
  nstate->spin = cstate->spin;
  // nstate->spin.Accumulate(stepsize, vtmpA);
  accumulate(grid_size, block_size, size, 1.f, stepsize, host_struct.dev_dm_dt,
    dev_MValue_backup, host_struct.dev_MValue);
  // Tweak "last_timestep" field in next_state, and adjust other
  // time fields to protect against rounding errors.
  UpdateTimeFields(*cstate, *nstate, stepsize);

  // Normalize spins in nstate, and collect norm error info
  // Normalize m2, including norm error check
  FD_TYPE *dev_min_magsq = dev_info + 6;
  FD_TYPE *dev_max_magsq = dev_info + 7;
  makeUnitAndCollectMinMax(grid_size, block_size, size, reduce_size, BLK_SIZE,
    host_struct.dev_MValue, host_struct.dev_local_sum, dev_min_magsq, 
    dev_max_magsq);
  const Oxs_SimState* endstate
    = &(next_state_key.GetReadReference()); // Lock down
                      
  // To estimate error, compute dm_dt at end state.
  GPU_GetEnergyDensity(*endstate, temp_energy, &mxH_output.cache.value,
                   NULL, pE_pt);
  /// compute total_E /// computation of dev_dE and dev_var_dE is redundent, but
  /// good of reusing code
  FD_TYPE *dev_dE = dev_info + 3;
  FD_TYPE *dev_var_dE = dev_info + 4;
  FD_TYPE *dev_total_E = dev_info + 5;
  collectEnergyStatistics(grid_size, block_size, size, reduce_size, BLK_SIZE,
    host_struct, dev_dE, dev_var_dE, dev_total_E);
  backUpEnergy(host_struct.dev_dm_dt_bak, host_struct.dev_dm_dt, 3 * size);
  
  if((mxH_output.GetCacheRequestCount()>0
        && mxH_output.cache.state_id != endstate->Id())) {
    FD_TYPE *tmp_mxH = new FD_TYPE[size * 3];
    memDownload_device(tmp_mxH, host_struct.dev_torque, 3 * size, DEV_NUM);
    Oxs_MeshValue<ThreeVector>& mxH = mxH_output.cache.value;
    for(int i = 0; i < size; i++) {
      mxH[i] = ThreeVector(tmp_mxH[i], tmp_mxH[i+size], tmp_mxH[i+2*size]);
    }
    if(tmp_mxH) delete[] tmp_mxH;
    mxH_output.cache.state_id = endstate->Id();
  }
  // mxH_output.cache.state_id = endstate->Id();
  Calculate_dm_dt(*endstate, mxH_output.cache.value, pE_pt,
                  vtmpB, max_dm_dt, dE_dt, timestep_lower_bound, false); //Since 

  FD_TYPE *dev_max_error_sq = dev_info + 2;
  dmDtErrorStep2(grid_size, block_size, size, reduce_size, BLK_SIZE, 
    dev_dm_dt_backup, host_struct.dev_dm_dt_bak, host_struct.dev_dm_dt, 
    host_struct.dev_local_sum, dev_max_error_sq);
  
  FD_TYPE host_info[8];
  memDownload_device(host_info, dev_info, 8, DEV_NUM);  
  // host_info[0, 1] --> outputs from Calculate_dm_dt
  const OC_REAL8m max_dm_dt_sq = host_info[0];
  const OC_REAL8m dE_dt_sum = host_info[1];
  max_dm_dt = sqrt(max_dm_dt_sq);
  dE_dt = -1 * MU0 * dE_dt_sum + pE_pt;
  timestep_lower_bound = PositiveTimestepBound(max_dm_dt);
  // host_info[2] --> output from dmDtErrorStep2
  const OC_REAL8m max_err_sq = host_info[2];
  error_estimate = sqrt(max_err_sq) * stepsize;
  // host_info[3, 4, 5] --> outputs from energyStatistics
  energyStatistics[0] = host_info[3];
  energyStatistics[1] = host_info[4];
  energyStatistics[2] = host_info[5];
  const OC_REAL8m total_E = energyStatistics[2];
  // host_info[6, 7] --> outputs from makeUnitAndCollectMinMax
  const OC_REAL8m min_normsq = host_info[6];
  const OC_REAL8m max_normsq = host_info[7];
  norm_error = OC_MAX(sqrt(max_normsq)-1.0, 1.0 - sqrt(min_normsq));
    
  if(!endstate->AddDerivedData("Timestep lower bound",
                                timestep_lower_bound) ||
     !endstate->AddDerivedData("Max dm/dt",max_dm_dt) ||
     !endstate->AddDerivedData("pE/pt",pE_pt) ||
     !endstate->AddDerivedData("Total E",total_E) ||
     !endstate->AddDerivedData("dE/dt",dE_dt)) {
    throw Oxs_ExtError(this,
                         "Oxs_GPU_RungeKuttaEvolve::TakeRungeKuttaStep2:"
                         " Programming error; data cache already set.");
  }
  
  global_error_order = 2.0;

  // Move end dm_dt data into vtmpA, for use by calling routine.
  // Note that end energy is already in temp_energy, as per
  // contract.
  vtmpA.Swap(vtmpB);
  new_energy_and_dmdt_computed = 1;
}

void Oxs_GPU_RungeKuttaEvolve::TakeRungeKuttaStep2Heun
(OC_REAL8m stepsize,
 Oxs_ConstKey<Oxs_SimState> current_state_key,
 const Oxs_MeshValue<ThreeVector>& current_dm_dt,
 Oxs_Key<Oxs_SimState>& next_state_key,
 OC_REAL8m& error_estimate,OC_REAL8m& global_error_order,
 OC_REAL8m& norm_error,
 OC_BOOL& new_energy_and_dmdt_computed) {
  // This routine performs a second order Runge-Kutta step, with
  // error estimation.  The form used in this routine is the "Heun"
  // method.
  //
  // A general RK2 step involves
  //
  //  dm_dt1 = dm_dt(t1,m1)
  //  dm_dt2 = dm_dt(t1+a.h,m1+a.h.dm_dt1)
  //  m2 = m1 + h.( (1-1/(2a)).dm_dt1 + (1/(2a)).dm_dt2 )
  //
  // where 0<a<=1 is a free parameter.  Taking a=1/2 gives
  //
  //  dm_dt1 = dm_dt(t1,m1)
  //  dm_dt2 = dm_dt(t1+h/2,m1+dm_dt1.h/2)
  //  m2 = m1 + dm_dt2*h + O(h^3)
  //
  // This is the "modified Euler" method from Collatz (1960).
  // Setting a=1 yields
  //
  //  dm_dt1 = dm_dt(t1,m1)
  //  dm_dt2 = dm_dt(t1+h,m1+dm_dt1.h)
  //  m2 = m1 + (dm_dt1 + dm_dt2).h/2 + O(h^3)
  //
  // which is the method of Heun (1900).  For details see
  // J. Stoer and R. Bulirsch, "Introduction to Numerical
  // Analysis," Springer 1993, Section 7.2.1, p438.
  //
  // In the code below, we use the Heun approach,
  // i.e., select a=1.

  const Oxs_SimState* cstate = &(current_state_key.GetReadReference());

  OC_INDEX i;
  const OC_INDEX size = cstate->mesh->Size();

  backUpEnergy(host_struct.dev_energy_bak, host_struct.dev_energy, size);
  backUpEnergy(dev_MValue_backup, host_struct.dev_MValue, 3 * size);
  backUpEnergy(dev_dm_dt_backup, host_struct.dev_dm_dt, 3 * size);
  
  OC_REAL8m pE_pt,max_dm_dt,dE_dt,timestep_lower_bound,dummy_error;

  // Calculate dm_dt2
  AdjustState(stepsize,stepsize, *cstate, current_dm_dt,
              next_state_key.GetWriteReference(), dummy_error, 
              dev_MValue_backup, host_struct.dev_MValue, false);
  GPU_GetEnergyDensity(next_state_key.GetReadReference(), temp_energy,
                   &vtmpA, NULL, pE_pt);
  Calculate_dm_dt(next_state_key.GetReadReference(), vtmpA, pE_pt,
                  vtmpA, max_dm_dt, dE_dt, timestep_lower_bound, false);
  // vtmpA holds dm_dt2
  backUpEnergy(host_struct.dev_dm_dt_bak, host_struct.dev_dm_dt, 3 * size);

  // Form 0.5*(dm_dt1+dm_dt2)
  accumulate(grid_size, block_size, size, 0.5f, 0.5f, dev_dm_dt_backup,
    host_struct.dev_dm_dt, host_struct.dev_dm_dt);

  // Create new state
  Oxs_SimState* nstate = &(next_state_key.GetWriteReference());
  nstate->ClearDerivedData();
  nstate->spin = cstate->spin;
  // nstate->spin.Accumulate(stepsize,vtmpA);
  accumulate(grid_size, block_size, size, 1.f, stepsize, host_struct.dev_dm_dt,
    dev_MValue_backup, host_struct.dev_MValue);

  // Tweak "last_timestep" field in next_state, and adjust other
  // time fields to protect against rounding errors.
  UpdateTimeFields(*cstate,*nstate,stepsize);

  // Normalize spins in nstate, and collect norm error info
  // Normalize m2, including norm error check
  FD_TYPE *dev_min_magsq = dev_info + 6;
  FD_TYPE *dev_max_magsq = dev_info + 7;
  makeUnitAndCollectMinMax(grid_size, block_size, size, reduce_size, BLK_SIZE,
    host_struct.dev_MValue, host_struct.dev_local_sum, dev_min_magsq, 
    dev_max_magsq);
  const Oxs_SimState* endstate
    = &(next_state_key.GetReadReference()); // Lock down

  // To estimate error, compute dm_dt at end state.
  GPU_GetEnergyDensity(*endstate, temp_energy, &mxH_output.cache.value,
                   NULL, pE_pt);
  /// compute total_E
  FD_TYPE *dev_dE = dev_info + 3;
  FD_TYPE *dev_var_dE = dev_info + 4;
  FD_TYPE *dev_total_E = dev_info + 5;
  collectEnergyStatistics(grid_size, block_size, size, reduce_size, BLK_SIZE,
    host_struct, dev_dE, dev_var_dE, dev_total_E);
  
  if((mxH_output.GetCacheRequestCount()>0
        && mxH_output.cache.state_id != endstate->Id())) {
    FD_TYPE *tmp_mxH = new FD_TYPE[size * 3];
    memDownload_device(tmp_mxH, host_struct.dev_torque, 3 * size, DEV_NUM);
    Oxs_MeshValue<ThreeVector>& mxH = mxH_output.cache.value;
    for(int i = 0; i < size; i++) {
      mxH[i] = ThreeVector(tmp_mxH[i], tmp_mxH[i+size], tmp_mxH[i+2*size]);
    }
    if(tmp_mxH) delete[] tmp_mxH;
    mxH_output.cache.state_id = endstate->Id();
  }
  
  // mxH_output.cache.state_id=endstate->Id();
  Calculate_dm_dt(*endstate, mxH_output.cache.value, pE_pt,
                  vtmpB, max_dm_dt, dE_dt, timestep_lower_bound, false);

  // Best guess at error compares computed endpoint against what
  // we would get if we ran a Heun-type step with dm/dt at this
  // endpoint.
  // OC_REAL8m max_err_sq = 0.0;
  // for(i=0;i<size;++i) { // ***ALREADY DEFINED ON GPU***
    // ThreeVector tvec = vtmpC[i] - vtmpB[i];
    // OC_REAL8m err_sq = tvec.MagSq();
    // if(err_sq>max_err_sq) max_err_sq = err_sq;
  // }
  FD_TYPE *dev_max_error_sq = dev_info + 2;
  dmDtError(grid_size, block_size, size, reduce_size, BLK_SIZE, host_struct,
    dev_max_error_sq);
  
  FD_TYPE host_info[8];
  memDownload_device(host_info, dev_info, 8, DEV_NUM);  
  // host_info[0, 1] --> outputs from Calculate_dm_dt
  const OC_REAL8m max_dm_dt_sq = host_info[0];
  const OC_REAL8m dE_dt_sum = host_info[1];
  max_dm_dt = sqrt(max_dm_dt_sq);
  dE_dt = -1 * MU0 * dE_dt_sum + pE_pt;
  timestep_lower_bound = PositiveTimestepBound(max_dm_dt);
  // host_info[2] --> output from dmDtErrorStep2
  const OC_REAL8m max_err_sq = host_info[2];
  error_estimate = sqrt(max_err_sq) * stepsize / 2.;
  // host_info[3, 4, 5] --> outputs from energyStatistics
  energyStatistics[0] = host_info[3];
  energyStatistics[1] = host_info[4];
  energyStatistics[2] = host_info[5];
  const OC_REAL8m total_E = energyStatistics[2];
  // host_info[6, 7] --> outputs from makeUnitAndCollectMinMax
  const OC_REAL8m min_normsq = host_info[6];
  const OC_REAL8m max_normsq = host_info[7];
  norm_error = OC_MAX(sqrt(max_normsq)-1.0, 1.0 - sqrt(min_normsq));
    
  if(!endstate->AddDerivedData("Timestep lower bound",
                                timestep_lower_bound) ||
     !endstate->AddDerivedData("Max dm/dt",max_dm_dt) ||
     !endstate->AddDerivedData("pE/pt",pE_pt) ||
     !endstate->AddDerivedData("Total E",total_E) ||
     !endstate->AddDerivedData("dE/dt",dE_dt)) {
    throw Oxs_ExtError(this,
                         "Oxs_GPU_RungeKuttaEvolve::TakeRungeKuttaStep2Heun:"
                         " Programming error; data cache already set.");
  }
  
  global_error_order = 2.0;

  // Move end dm_dt data into vtmpA, for use by calling routine.
  // Note that end energy is already in temp_energy, as per
  // contract.
  vtmpA.Swap(vtmpB);
  new_energy_and_dmdt_computed = 1;
}

void Oxs_GPU_RungeKuttaEvolve::TakeRungeKuttaStep4
(OC_REAL8m stepsize,
 Oxs_ConstKey<Oxs_SimState> current_state_key,
 const Oxs_MeshValue<ThreeVector>& current_dm_dt,
 Oxs_Key<Oxs_SimState>& next_state_key,
 OC_REAL8m& error_estimate,OC_REAL8m& global_error_order,
 OC_REAL8m& norm_error,
 OC_BOOL& new_energy_and_dmdt_computed) {
  // This routine performs two successive "classical" Runge-Kutta
  // steps of size stepsize/2, and stores the resulting magnetization
  // state into the next_state export.  Additionally, a single
  // step of size stepsize is performed, and used for estimating
  // the error.  This error is returned in the export error_estimate;
  // This is the largest error detected cellwise, in radians.  The
  // export global_error_order is always set to 4 by this routine.
  // (The local error order is one better, i.e., 5.)  The norm_error
  // export is set to the cellwise maximum deviation from unit norm
  // across all the spins in the final state, before renormalization.

  // A single RK4 step involves
  //  dm_dt1 = dm_dt(t1,m1)
  //  dm_dt2 = dm_dt(t1+h/2,m1+dm_dt1*h/2)
  //  dm_dt3 = dm_dt(t1+h/2,m1+dm_dt2*h/2)
  //  dm_dt4 = dm_dt(t1+h,m1+dm_dt3*h)
  //  m2 = m1 + dm_dt1*h/6 + dm_dt2*h/3 + dm_dt3*h/3 + dm_dt4*h/6 + O(h^5)
  // To improve accuracy, for each step accumulate dm_dt?, where
  // ?=1,2,3,4, into m2 with proper weights, and add in m1 at the end.

  // To calculate dm_dt, we first fill next_state with the proper
  // spin data (e.g., m1+dm_dt2*h/2).  The utility routine AdjustState
  // is applied for this purpose.  Then next_state is locked down,
  // so that it will have a valid state_id, and GetEnergyDensity
  // is called in order to get mxH and related data.  This is then
  // passed to Calculate_dm_dt.  Note that dm_dt is not calculated
  // for the final state.  This is left for the calling routine,
  // which first examines the error estimates to decide whether
  // or not the step will be accepted.  If the step is rejected,
  // then the energy and dm_dt do not need to be calculated.

  // Scratch space usage:
  //   vtmpA is used to store, successively, dm_dt2, dm_dt3 and
  //      dm_dt4.  In calculating dm_dt?, vtmpA is first filled
  //      with mxH by the GetEnergyDensity() routine.
  //      Calculate_dm_dt() allows import mxH and export dm_dt
  //      to be the same MeshValue structure.
  //   vtmpB is used to accumulate the weighted sums of the dm_dt's,
  //      as explained above.  To effect a small efficiency gain,
  //      dm_dt1 is calculated directly into vtmpB, in the same
  //      manner as the other dm_dt's are filled into vtmpA.
  //   tempState is used to hold the middle state when calculating
  //      the two half steps, and the end state for the single
  //      full size step.

  // Any locks arising from temp_state_key will be released when
  // temp_state_key is destroyed on method exit.
  Oxs_SimState* nstate = &(next_state_key.GetWriteReference());
  const Oxs_SimState* cstate = &(current_state_key.GetReadReference());
  Oxs_Key<Oxs_SimState> temp_state_key;
  director->GetNewSimulationState(temp_state_key);
  Oxs_SimState* tstate = &(temp_state_key.GetWriteReference());
  nstate->CloneHeader(*tstate);

  OC_INDEX i;
  const OC_INDEX size = cstate->mesh->Size();
  
  if (!dev_dm_dt_backup2) {
    alloc_device(dev_dm_dt_backup2, 3 * size, DEV_NUM, "dev_dm_dt_backup2");
  }
  if (!dev_MValue_backup2) {
    alloc_device(dev_MValue_backup2, 3 * size, DEV_NUM, "dev_MValue_backup2");
  }
  backUpEnergy(host_struct.dev_energy_bak, host_struct.dev_energy, size);
  backUpEnergy(dev_MValue_backup, host_struct.dev_MValue, 3 * size);
  backUpEnergy(dev_dm_dt_backup, host_struct.dev_dm_dt, 3 * size);

  OC_REAL8m pE_pt,max_dm_dt,dE_dt,timestep_lower_bound;
  OC_REAL8m dummy_error;
  // Do first half step.  Because dm_dt1 is already calculated,
  // we fill dm_dt2 directly into vtmpB.
  AdjustState(stepsize/4, stepsize/4, *cstate, current_dm_dt,
              temp_state_key.GetWriteReference(), dummy_error, 
              dev_MValue_backup, host_struct.dev_MValue, false);
  GPU_GetEnergyDensity(temp_state_key.GetReadReference(), temp_energy,
                   &vtmpB, NULL, pE_pt);
  Calculate_dm_dt(temp_state_key.GetReadReference(), vtmpB, pE_pt,
                  vtmpB, max_dm_dt, dE_dt, timestep_lower_bound, false);
  backUpEnergy(host_struct.dev_dm_dt_bak, host_struct.dev_dm_dt, 3 * size);                
  // vtmpB currently holds dm_dt2
  
  AdjustState(stepsize/4, stepsize/4, *cstate, vtmpB,
              temp_state_key.GetWriteReference(), dummy_error, 
              dev_MValue_backup, host_struct.dev_MValue, false);
  GPU_GetEnergyDensity(temp_state_key.GetReadReference(), temp_energy,
                   &vtmpA, NULL, pE_pt);
  Calculate_dm_dt(temp_state_key.GetReadReference(), vtmpA, pE_pt,
                  vtmpA, max_dm_dt, dE_dt, timestep_lower_bound, false);
  accumulate(grid_size, block_size, size, 1.f, 1.f, host_struct.dev_dm_dt,
    host_struct.dev_dm_dt_bak, host_struct.dev_dm_dt_bak);
  // vtmpA holds dm_dt3, vtmpB holds dm_dt2 + dm_dt3.
  
  AdjustState(stepsize/2, stepsize/2, *cstate, vtmpA,
              temp_state_key.GetWriteReference(), dummy_error, 
              dev_MValue_backup, host_struct.dev_MValue, false);
  GPU_GetEnergyDensity(temp_state_key.GetReadReference(), temp_energy,
                   &vtmpA, NULL, pE_pt);
  Calculate_dm_dt(temp_state_key.GetReadReference(), vtmpA, pE_pt,
                  vtmpA, max_dm_dt, dE_dt, timestep_lower_bound, false);
  // vtmpA holds dm_dt4
  accumulate(grid_size, block_size, size, 1.f, 1.f, dev_dm_dt_backup,
    host_struct.dev_dm_dt, host_struct.dev_dm_dt);
  accumulate(grid_size, block_size, size, 1.f, 0.5f, host_struct.dev_dm_dt,
    host_struct.dev_dm_dt_bak, host_struct.dev_dm_dt_bak);
  tstate = &(temp_state_key.GetWriteReference());
  tstate->ClearDerivedData();
  tstate->spin = cstate->spin;
  accumulate(grid_size, block_size, size, 1.f, stepsize/6., 
    host_struct.dev_dm_dt_bak, dev_MValue_backup, host_struct.dev_MValue);
  // Note: state time index set to lasttime + stepsize/2
  // by dm_dt4 calculation above.  Note that "h" here is
  // stepsize/2, so the weights on dm_dt1, dm_dt2, dm_dt3 and
  // dn_dt4 are (stepsize/2)(1/6, 1/3, 1/3, 1/6), respectively.

  // Save vtmpB for error estimate
  // vtmpB.Swap(vtmpC);
  backUpEnergy(dev_dm_dt_backup2, host_struct.dev_dm_dt_bak, 3 * size); 
  // Normalize spins in tstate, and collect norm error info.
  FD_TYPE *dev_min_magsq = dev_info + 6;
  FD_TYPE *dev_max_magsq = dev_info + 7;
  makeUnitAndCollectMinMax(grid_size, block_size, size, reduce_size, BLK_SIZE,
    host_struct.dev_MValue, host_struct.dev_local_sum, dev_min_magsq, 
    dev_max_magsq);
    
  // At this point, temp_state holds the middle point.
  // Calculate dm_dt for this state, and store in vtmpB.
  tstate = NULL; // Disable non-const access
  const Oxs_SimState* midstate
    = &(temp_state_key.GetReadReference());
  GPU_GetEnergyDensity(*midstate, temp_energy, &vtmpB, NULL, pE_pt);
  Calculate_dm_dt(*midstate, vtmpB, pE_pt,
                  vtmpB, max_dm_dt, dE_dt, timestep_lower_bound, false);
  backUpEnergy(dev_MValue_backup2, host_struct.dev_MValue, 3 * size);
  backUpEnergy(host_struct.dev_dm_dt_bak, host_struct.dev_dm_dt, 3 * size);

  // Next do second half step.  Store end result in next_state
  AdjustState(stepsize/4, stepsize/4, *midstate, vtmpB,
              next_state_key.GetWriteReference(), dummy_error,
              dev_MValue_backup2, host_struct.dev_MValue, false);
  GPU_GetEnergyDensity(next_state_key.GetReadReference(), temp_energy,
                   &vtmpA, NULL, pE_pt);
  Calculate_dm_dt(next_state_key.GetReadReference(), vtmpA, pE_pt,
                  vtmpA, max_dm_dt, dE_dt, timestep_lower_bound, false);
  // vtmpB currently holds dm_dt1, vtmpA holds dm_dt2
  accumulate(grid_size, block_size, size, 0.5f, 1.f, host_struct.dev_dm_dt,
    host_struct.dev_dm_dt_bak, host_struct.dev_dm_dt_bak);
   
  AdjustState(stepsize/4, stepsize/4, *midstate, vtmpA,
              next_state_key.GetWriteReference(), dummy_error,
              dev_MValue_backup2, host_struct.dev_MValue, false);
  GPU_GetEnergyDensity(next_state_key.GetReadReference(), temp_energy,
                   &vtmpA, NULL, pE_pt);
  Calculate_dm_dt(next_state_key.GetReadReference(), vtmpA, pE_pt,
                  vtmpA, max_dm_dt, dE_dt, timestep_lower_bound, false);
  accumulate(grid_size, block_size, size, 1.f, 1.f, host_struct.dev_dm_dt,
    host_struct.dev_dm_dt_bak, host_struct.dev_dm_dt_bak);
  // vtmpA holds dm_dt3, vtmpB holds dm_dt1/2 + dm_dt2 + dm_dt3.
  AdjustState(stepsize/2, stepsize/2, *midstate, vtmpA,
              next_state_key.GetWriteReference(), dummy_error,
              dev_MValue_backup2, host_struct.dev_MValue, false);
  GPU_GetEnergyDensity(next_state_key.GetReadReference(), temp_energy,
                   &vtmpA, NULL, pE_pt);
  Calculate_dm_dt(next_state_key.GetReadReference(), vtmpA, pE_pt,
                  vtmpA, max_dm_dt, dE_dt, timestep_lower_bound, false);
  // vtmpA holds dm_dt4
  accumulate(grid_size, block_size, size, 1.f, 0.5f, host_struct.dev_dm_dt,
    host_struct.dev_dm_dt_bak, host_struct.dev_dm_dt_bak);
    
  nstate = &(next_state_key.GetWriteReference());
  nstate->ClearDerivedData();
  nstate->spin = midstate->spin;
  // nstate->spin.Accumulate(stepsize/6.,vtmpB);
  accumulate(grid_size, block_size, size, 1.f, stepsize/6., 
    host_struct.dev_dm_dt_bak, dev_MValue_backup2, host_struct.dev_MValue);
  midstate = NULL; // We're done using midstate

  // Combine vtmpB with results from first half-step.
  // This is used for error estimation.
  // vtmpC += vtmpB;
  accumulate(grid_size, block_size, size, 1.f, 1.f, host_struct.dev_dm_dt_bak,
    dev_dm_dt_backup2, dev_dm_dt_backup2);

  // Tweak "last_timestep" field in next_state, and adjust other
  // time fields to protect against rounding errors.
  UpdateTimeFields(*cstate,*nstate,stepsize);
  // Normalize spins in nstate, and collect norm error info.
  FD_TYPE *dev_min_magsq2 = dev_info + 8;
  FD_TYPE *dev_max_magsq2 = dev_info + 9;
  makeUnitAndCollectMinMax(grid_size, block_size, size, reduce_size, BLK_SIZE,
    host_struct.dev_MValue, host_struct.dev_local_sum, dev_min_magsq2, 
    dev_max_magsq2);

  nstate = NULL;
  next_state_key.GetReadReference(); // Lock down (safety)
  backUpEnergy(dev_MValue_backup2, host_struct.dev_MValue, 3 * size);
  backUpEnergy(host_struct.dev_dm_dt, dev_dm_dt_backup, 3 * size);

  // Repeat now for full step, storing end result into temp_state
  AdjustState(stepsize/2, stepsize/2, *cstate, current_dm_dt,
              temp_state_key.GetWriteReference(), dummy_error,
              dev_MValue_backup, host_struct.dev_MValue, false);
  GPU_GetEnergyDensity(temp_state_key.GetReadReference(), temp_energy,
                   &vtmpB, NULL, pE_pt);
  Calculate_dm_dt(temp_state_key.GetReadReference(), vtmpB, pE_pt,
                  vtmpB, max_dm_dt, dE_dt, timestep_lower_bound, false);
  backUpEnergy(host_struct.dev_dm_dt_bak, host_struct.dev_dm_dt, 3 * size);
  // vtmpB currently holds dm_dt2
  AdjustState(stepsize/2, stepsize/2, *cstate, vtmpB,
              temp_state_key.GetWriteReference(), dummy_error,
              dev_MValue_backup, host_struct.dev_MValue, false);
  GPU_GetEnergyDensity(temp_state_key.GetReadReference(), temp_energy,
                   &vtmpA, NULL, pE_pt);
  Calculate_dm_dt(temp_state_key.GetReadReference(), vtmpA, pE_pt,
                  vtmpA,max_dm_dt, dE_dt, timestep_lower_bound, false);
  // vtmpB += vtmpA;
  accumulate(grid_size, block_size, size, 1.f, 1.f, host_struct.dev_dm_dt,
    host_struct.dev_dm_dt_bak, host_struct.dev_dm_dt_bak);
  // vtmpA holds dm_dt3, vtmpB holds dm_dt2 + dm_dt3.
  AdjustState(stepsize,stepsize, *cstate, vtmpA,
              temp_state_key.GetWriteReference(), dummy_error,
              dev_MValue_backup, host_struct.dev_MValue, false);
  GPU_GetEnergyDensity(temp_state_key.GetReadReference(), temp_energy,
                   &vtmpA, NULL, pE_pt);
  Calculate_dm_dt(temp_state_key.GetReadReference(), vtmpA, pE_pt,
                  vtmpA, max_dm_dt, dE_dt, timestep_lower_bound, false);
  // vtmpA holds dm_dt4
  accumulate(grid_size, block_size, size, 1.f, 1.f, dev_dm_dt_backup,
    host_struct.dev_dm_dt, host_struct.dev_dm_dt);
  accumulate(grid_size, block_size, size, 1.f, 0.5f, host_struct.dev_dm_dt,
    host_struct.dev_dm_dt_bak, host_struct.dev_dm_dt_bak);

  backUpEnergy(host_struct.dev_MValue, dev_MValue_backup2, 3 * size);
  
  // Estimate error
  FD_TYPE *dev_max_error_sq = dev_info + 2;
  dmDtErrorStep4(grid_size, block_size, size, reduce_size, BLK_SIZE,
    host_struct.dev_dm_dt_bak, dev_dm_dt_backup2, host_struct.dev_local_sum,
    dev_max_error_sq);
  /// vtmpB hold 0.5*dm_dt1 + dm_dt2 + dm_dt3 + 0.5*dm_dt4,
  /// but successor state looks like
  ///    m2 = m1 + dm_dt1*h/6 + dm_dt2*h/3 + dm_dt3*h/3 + dm_dt4*h/6 + O(h^5)

  FD_TYPE host_info[10];
  memDownload_device(host_info, dev_info, 10, DEV_NUM);  
  // host_info[2] --> output from dmDtErrorStep4
  const OC_REAL8m max_error_sq = host_info[2];
  error_estimate = sqrt(max_error_sq) * stepsize / 3.;
  // host_info[6, 7, 8, 9] --> outputs from makeUnitAndCollectMinMax
  const OC_REAL8m min_normsq = min(host_info[6], host_info[8]);
  const OC_REAL8m max_normsq = max(host_info[7], host_info[9]);

  norm_error = OC_MAX(sqrt(max_normsq) - 1.0, 1.0 - sqrt(min_normsq));
    
  global_error_order = 4.0;
  new_energy_and_dmdt_computed = 0;
}

void Oxs_GPU_RungeKuttaEvolve::RungeKuttaFehlbergBase54
(RKF_SubType method,
 OC_REAL8m stepsize,
 Oxs_ConstKey<Oxs_SimState> current_state_key,
 const Oxs_MeshValue<ThreeVector>& current_dm_dt,
 Oxs_Key<Oxs_SimState>& next_state_key,
 OC_REAL8m& error_estimate,OC_REAL8m& global_error_order,
 OC_REAL8m& norm_error,
 OC_BOOL& new_energy_and_dmdt_computed) { 
  // Runge-Kutta-Fehlberg routine with combined 4th and 5th
  // order Runge-Kutta steps.  The difference between the
  // two results (4th vs. 5th) is used to estimate the error.
  // The largest detected error detected cellsize is returned
  // in export error_estimate.  The units on this are radians.
  // The export global_error_order is set to 4 by this routine.
  // (The local error order is one better, i.e., 5.)  The norm_error
  // export is set to the cellwise maximum deviation from unit norm
  // across all the spins in the final state, before renormalization.

#if REPORT_TIME_RKDEVEL
timer[1].Start(); /**/
timer[2].Start(); /**/
#endif // REPORT_TIME_RKDEVEL

  // The following coefficients appear in
  //
  //   J. R. Dormand and P. J. Prince, ``A family of embedded
  //   Runge-Kutta formulae,'' J. Comp. Appl. Math., 6, 19--26
  //   (1980).
  //
  // They are also listed in J. Stoer and R. Bulirsch's book,
  // ``Introduction to Numerical Analysis,'' Springer, 2nd edition,
  // Sec. 7.2.5, p 454, but there are a number of errors in the S&B
  // account; the reader is recommended to refer directly to the D&P
  // paper.  A follow-up paper,
  //
  //   J. R. Dormand and P. J. Prince, ``A reconsideration of some 
  //   embedded Runge-Kutta formulae,'' J. Comp. Appl. Math., 15,
  //   203--211 (1986)
  //
  // provides formulae with improved stability and higher order.
  // See also
  //
  //   J. H. Williamson, ``Low-Storage Runge-Kutta Schemes,''
  //   J. Comp. Phys., 35, 48--56 (1980).
  //
  // FORMULAE for RK5(4)7FM:
  //
  //     dm_dt1 = dm_dt(t1,m1)
  //     dm_dt2 = dm_dt(t1+ (1/5)*h, m1+h*k1);
  //     dm_dt3 = dm_dt(t1+(3/10)*h, m1+h*k2);
  //     dm_dt4 = dm_dt(t1+ (4/5)*h, m1+h*k3);
  //     dm_dt5 = dm_dt(t1+ (8/9)*h, m1+h*k4);
  //     dm_dt6 = dm_dt(t1+     1*h, m1+h*k5);
  //     dm_dt7 = dm_dt(t1+     1*h, m1+h*k6);
  //  where
  //     k1 = dm_dt1*1/5
  //     k2 = dm_dt1*3/40       + dm_dt2*9/40
  //     k3 = dm_dt1*44/45      - dm_dt2*56/15      + dm_dt3*32/9
  //     k4 = dm_dt1*19372/6561 - dm_dt2*25360/2187 + dm_dt3*64448/6561
  //               - dm_dt4*212/729
  //     k5 = dm_dt1*9017/3168  - dm_dt2*355/33     + dm_dt3*46732/5247
  //               + dm_dt4*49/176   - dm_dt5*5103/18656
  //     k6 = dm_dt1*35/384     + 0                 + dm_dt3*500/1113
  //               + dm_dt4*125/192  - dm_dt5*2187/6784  + dm_dt6*11/84
  // Then
  //     Da = dm_dt1*35/384     + 0 + dm_dt3*500/1113   + dm_dt4*125/192
  //              - dm_dt5*2187/6784      + dm_dt6*11/84
  //     Db = dm_dt1*5179/57600 + 0 + dm_dt3*7571/16695 + dm_dt4*393/640
  //              - dm_dt5*92097/339200   + dm_dt6*187/2100   + dm_dt7*1/40
  // and
  //     m2a = m1 + h*Da
  //     m2b = m1 + h*Db.
  //
  // where m2a is the 5th order estimate, which is the candidate
  // for the next state, and m2b is the 4th order estimate used
  // only for error estimation/stepsize control.  Note that the
  // 4th order estimate uses more dm/dt evaluations (7) than the
  // 5th order method (6).  This is intentional; the coefficients
  // are selected to minimize error (see D&P paper cited above).
  // The extra calculation costs are minimal, because the 7th dm_dt
  // evaluation is at the candidate next state, it is re-used
  // on the next step unless the step rejected.
  //
  // The error estimate is
  // 
  //     E = |m2b-m2a| = h*|Db-Da| = C*h^6
  //
  // where C is a constant that can be estimated in terms of E.
  // Note that we don't need to know C in order to adjust the
  // stepsize, because stepsize adjustment involves only the
  // ratio of the old stepsize to the new, so C drops out.

  // Scratch space usage:
  //  The import next_state is used for intermediate state
  // storage for all dm_dt computations.  The final computation
  // is for dm_dt7 at m1+h*k6 = m1+h*Da, which is the candidate
  // next state.  (Da=k6; see FSAL note below.)

  // Four temporary arrays, A-D, are used:
  //
  // Step \  Temp
  // Index \ Array:  A         B         C         D
  // ------+---------------------------------------------
  //   1   |      dm_dt2       -         -         -
  //   2   |      dm_dt2      k2         -         -
  //   3   |      dm_dt2     dm_dt3      -         -
  //   4   |      dm_dt2     dm_dt3     k3         -
  //   5   |      dm_dt2     dm_dt3    dm_dt4      -
  //   6   |      dm_dt2     dm_dt3    dm_dt4     k4
  //   7   |      dm_dt2     dm_dt3    dm_dt4    dm_dt5
  //   8   |        k5       dm_dt3    dm_dt4    dm_dt5
  //   9   |      dm_dt6     dm_dt3    dm_dt4    dm_dt5
  //  10   |      k6(3,6)    dD(3,6)   dm_dt4    dm_dt5
  //  11   |      dm_dt7     dD(3,6)   dm_dt4    dm_dt5
  //  12   |      dm_dt7       dD      dm_dt4    dm_dt5
  //
  // Here dD is Db-Da.  We don't need to compute Db directly.
  // The parenthesized numbers, e.g., k6(3,6) indicates
  // a partially formed value.  The total value k6 depends
  // upon dm_dt1, dm_dt3, dm_dt4, dm_dt5, and dm_dt6.  But
  // if we form k6 directly at step 11 in array A, then we
  // lose dm_dt6 which is needed to compute dD.  Rather than
  // use an additional array, we accumulate partial results
  // into arrays A and B for k6 and dD as indicated.
  //   Note that Da = k6.  Further, note that dm_dt7 is
  // dm_dt for the next state candidate.  (This is the 'F',
  // short for 'FSAL' ("first same as last"?) in the method
  // name, RK5(4)7FM.  The 'M' means minimized error norm,
  // 7 is the number of stages, and 5(4) is the main/subsidiary
  // integration formula order.  See the D&P 1986 paper for
  // details and additional references.)

  // Coefficient arrays, a, b, dc, defined by:
  //
  //   dm_dtN = dm_dt(t1+aN*h,m1+h*kN)
  //       kN = \sum_{M=1}^{M=N} dm_dtM*bNM
  //  Db - Da = \sum dm_dtM*dcM
  //
  OC_REAL8m a1,a2,a3,a4; // a5 and a6 are 1.0
  OC_REAL8m b11, b21, b22, b31, b32, b33, b41, b42, b43, b44,
    b51, b52, b53, b54, b55, b61, b63, b64, b65, b66; // b62 is 0.0
  OC_REAL8m dc1, dc3, dc4, dc5, dc6, dc7;  // c[k] = b6k, and c^[2]=c[2]=0.0,
  /// where c are the coeffs for Da, c^ for Db, and dcM = c^[M]-c[M].

  switch(method) {
  case RK547FC:
    /////////////////////////////////////////////////////////////////
    // Coefficients for Dormand & Prince RK5(4)7FC
    a1 = OC_REAL8m(1.)/OC_REAL8m(5.);
    a2 = OC_REAL8m(3.)/OC_REAL8m(10.);
    a3 = OC_REAL8m(6.)/OC_REAL8m(13.);
    a4 = OC_REAL8m(2.)/OC_REAL8m(3.);
    // a5 and a6 are 1.0

    b11 =      OC_REAL8m(1.)/OC_REAL8m(5.);
  
    b21 =      OC_REAL8m(3.)/OC_REAL8m(40.);
    b22 =      OC_REAL8m(9.)/OC_REAL8m(40.);
  
    b31 =    OC_REAL8m(264.)/OC_REAL8m(2197.);
    b32 =    OC_REAL8m(-90.)/OC_REAL8m(2197.);
    b33 =    OC_REAL8m(840.)/OC_REAL8m(2197.);
  
    b41 =    OC_REAL8m(932.)/OC_REAL8m(3645.);
    b42 =    OC_REAL8m(-14.)/OC_REAL8m(27.);
    b43 =   OC_REAL8m(3256.)/OC_REAL8m(5103.);
    b44 =   OC_REAL8m(7436.)/OC_REAL8m(25515.);
  
    b51 =   OC_REAL8m(-367.)/OC_REAL8m(513.);
    b52 =     OC_REAL8m(30.)/OC_REAL8m(19.);
    b53 =   OC_REAL8m(9940.)/OC_REAL8m(5643.);
    b54 = OC_REAL8m(-29575.)/OC_REAL8m(8208.);
    b55 =   OC_REAL8m(6615.)/OC_REAL8m(3344.);
  
    b61 =     OC_REAL8m(35.)/OC_REAL8m(432.);
    b63 =   OC_REAL8m(8500.)/OC_REAL8m(14553.);
    b64 = OC_REAL8m(-28561.)/OC_REAL8m(84672.);
    b65 =    OC_REAL8m(405.)/OC_REAL8m(704.);
    b66 =     OC_REAL8m(19.)/OC_REAL8m(196.);
    // b62 is 0.0

    // Coefs for error calculation (c^[k] - c[k]).
    // Note that c[k] = b6k, and c^[2]=c[2]=0.0
    dc1 =     OC_REAL8m(11.)/OC_REAL8m(108.)    - b61;
    dc3 =   OC_REAL8m(6250.)/OC_REAL8m(14553.)  - b63;
    dc4 =  OC_REAL8m(-2197.)/OC_REAL8m(21168.)  - b64;
    dc5 =     OC_REAL8m(81.)/OC_REAL8m(176.)    - b65;
    dc6 =    OC_REAL8m(171.)/OC_REAL8m(1960.)   - b66;
    dc7 =      OC_REAL8m(1.)/OC_REAL8m(40.);
    break;

  case RK547FM:
    /////////////////////////////////////////////////////////////////
    // Coefficients for Dormand & Prince RK5(4)7FM
    a1 = OC_REAL8m(1.)/OC_REAL8m(5.);
    a2 = OC_REAL8m(3.)/OC_REAL8m(10.);
    a3 = OC_REAL8m(4.)/OC_REAL8m(5.);
    a4 = OC_REAL8m(8.)/OC_REAL8m(9.);
    // a5 and a6 are 1.0

    b11 =      OC_REAL8m(1.)/OC_REAL8m(5.);
  
    b21 =      OC_REAL8m(3.)/OC_REAL8m(40.);
    b22 =      OC_REAL8m(9.)/OC_REAL8m(40.);
  
    b31 =     OC_REAL8m(44.)/OC_REAL8m(45.);
    b32 =    OC_REAL8m(-56.)/OC_REAL8m(15.);
    b33 =     OC_REAL8m(32.)/OC_REAL8m(9.);
  
    b41 =  OC_REAL8m(19372.)/OC_REAL8m(6561.);
    b42 = OC_REAL8m(-25360.)/OC_REAL8m(2187.);
    b43 =  OC_REAL8m(64448.)/OC_REAL8m(6561.);
    b44 =   OC_REAL8m(-212.)/OC_REAL8m(729.);
  
    b51 =   OC_REAL8m(9017.)/OC_REAL8m(3168.);
    b52 =   OC_REAL8m(-355.)/OC_REAL8m(33.);
    b53 =  OC_REAL8m(46732.)/OC_REAL8m(5247.);
    b54 =     OC_REAL8m(49.)/OC_REAL8m(176.);
    b55 =  OC_REAL8m(-5103.)/OC_REAL8m(18656.);
  
    b61 =     OC_REAL8m(35.)/OC_REAL8m(384.);
    b63 =    OC_REAL8m(500.)/OC_REAL8m(1113.);
    b64 =    OC_REAL8m(125.)/OC_REAL8m(192.);
    b65 =  OC_REAL8m(-2187.)/OC_REAL8m(6784.);
    b66 =     OC_REAL8m(11.)/OC_REAL8m(84.);
    // b62 is 0.0

    // Coefs for error calculation (c^[k] - c[k]).
    // Note that c[k] = b6k, and c^[2]=c[2]=0.0
    dc1 =   OC_REAL8m(5179.)/OC_REAL8m(57600.)  - b61;
    dc3 =   OC_REAL8m(7571.)/OC_REAL8m(16695.)  - b63;
    dc4 =    OC_REAL8m(393.)/OC_REAL8m(640.)    - b64;
    dc5 = OC_REAL8m(-92097.)/OC_REAL8m(339200.) - b65;
    dc6 =    OC_REAL8m(187.)/OC_REAL8m(2100.)   - b66;
    dc7 =      OC_REAL8m(1.)/OC_REAL8m(40.);
    break;
  case RK547FS:
    /////////////////////////////////////////////////////////////////
    // Coefficients for Dormand & Prince RK5(4)7FS
    a1 = OC_REAL8m(2.)/OC_REAL8m(9.);
    a2 = OC_REAL8m(1.)/OC_REAL8m(3.);
    a3 = OC_REAL8m(5.)/OC_REAL8m(9.);
    a4 = OC_REAL8m(2.)/OC_REAL8m(3.);
    // a5 and a6 are 1.0

    b11 =      OC_REAL8m(2.)/OC_REAL8m(9.);
  
    b21 =      OC_REAL8m(1.)/OC_REAL8m(12.);
    b22 =      OC_REAL8m(1.)/OC_REAL8m(4.);
  
    b31 =     OC_REAL8m(55.)/OC_REAL8m(324.);
    b32 =    OC_REAL8m(-25.)/OC_REAL8m(108.);
    b33 =     OC_REAL8m(50.)/OC_REAL8m(81.);
  
    b41 =     OC_REAL8m(83.)/OC_REAL8m(330.);
    b42 =    OC_REAL8m(-13.)/OC_REAL8m(22.);
    b43 =     OC_REAL8m(61.)/OC_REAL8m(66.);
    b44 =      OC_REAL8m(9.)/OC_REAL8m(110.);
  
    b51 =    OC_REAL8m(-19.)/OC_REAL8m(28.);
    b52 =      OC_REAL8m(9.)/OC_REAL8m(4.);
    b53 =      OC_REAL8m(1.)/OC_REAL8m(7.);
    b54 =    OC_REAL8m(-27.)/OC_REAL8m(7.);
    b55 =     OC_REAL8m(22.)/OC_REAL8m(7.);
  
    b61 =     OC_REAL8m(19.)/OC_REAL8m(200.);
    b63 =      OC_REAL8m(3.)/OC_REAL8m(5.);
    b64 =   OC_REAL8m(-243.)/OC_REAL8m(400.);
    b65 =     OC_REAL8m(33.)/OC_REAL8m(40.);
    b66 =      OC_REAL8m(7.)/OC_REAL8m(80.);
    // b62 is 0.0

    // Coefs for error calculation (c^[k] - c[k]).
    // Note that c[k] = b6k, and c^[2]=c[2]=0.0
    dc1 =    OC_REAL8m(431.)/OC_REAL8m(5000.)  - b61;
    dc3 =    OC_REAL8m(333.)/OC_REAL8m(500.)   - b63;
    dc4 =  OC_REAL8m(-7857.)/OC_REAL8m(10000.) - b64;
    dc5 =    OC_REAL8m(957.)/OC_REAL8m(1000.)  - b65;
    dc6 =    OC_REAL8m(193.)/OC_REAL8m(2000.)  - b66;
    dc7 =     OC_REAL8m(-1.)/OC_REAL8m(50.);
    break;
  default:
    throw Oxs_ExtError(this,
                 "Oxs_GPU_RungeKuttaEvolve::RungeKuttaFehlbergBase54:"
                 " Programming error; Invalid sub-type.");
  }

#ifndef NDEBUG
  // COEFFICIENT CHECKS ////////////////////////////////////////
  // Try to catch some simple typing errors.  Oc_Nop calls below force
  // evaluation in order shown; otherwise, some compiler optimizations
  // reorder sums into form with less accuracy.
#define EPS (8*OC_REAL8_EPSILON)  // 6*OC_REAL8_EPSILON should be enough,
  /// but include a little slack compilers with bad numeric taste.
  if(fabs(b11-a1)>EPS ||
     fabs(b21+b22-a2)>EPS ||
     fabs(b31+b32+b33-a3)>EPS ||
     fabs(b41+b42+b43+b44-a4)>EPS ||
     fabs(Oc_Nop(b51+b52+b53) -1.0 + Oc_Nop(b54+b55))>EPS ||
     fabs(b61+b63+b64+b65+b66-1.0)>EPS) {
    char buf[512];
    Oc_Snprintf(buf,sizeof(buf),
                "Coefficient check failed:\n"
                "1: %g\n2: %g\n3: %g\n4: %g\n5: %g\n6: %g",
                static_cast<double>(b11-a1),
                static_cast<double>(b21+b22-a2),
                static_cast<double>(b31+b32+b33-a3),
                static_cast<double>(b41+b42+b43+b44-a4),
                static_cast<double>(b51+b52+b53+b54+b55-1.0),
                static_cast<double>(b61+b63+b64+b65+b66-1.0));
    throw Oxs_ExtError(this,buf);
  }
#endif // NDEBUG

  const Oxs_SimState* cstate = &(current_state_key.GetReadReference());
  OC_REAL8m pE_pt, max_dm_dt, dE_dt, timestep_lower_bound;
  OC_REAL8m dummy_error;

  const OC_INDEX size = cstate->mesh->Size();
  
  if (!dev_dm_dt_backup2) {
    alloc_device(dev_dm_dt_backup2, 3 * size, DEV_NUM, "dev_dm_dt_backup2");
  }
  if (!dev_MValue_backup2) {
    alloc_device(dev_MValue_backup2, 3 * size, DEV_NUM, "dev_MValue_backup2");
  }
  FD_TYPE* &dev_dm_dt_backup3 = dev_MValue_backup2;
  if (!dev_dm_dt_backup4) {
    alloc_device(dev_dm_dt_backup4, 3 * size, DEV_NUM, "dev_dm_dt_backup4");
  }
  
  backUpEnergy(host_struct.dev_energy_bak, host_struct.dev_energy, size);
  backUpEnergy(dev_MValue_backup, host_struct.dev_MValue, 3 * size);
  backUpEnergy(dev_dm_dt_backup, host_struct.dev_dm_dt, 3 * size);
  
#if REPORT_TIME_RKDEVEL
timer[2].Stop(); /**/
++timer_counts[2].pass_count;
 timer_counts[2].name = "RKFB54 setup";
#endif // REPORT_TIME_RKDEVEL
  // Step 1
  AdjustState(stepsize * a1, stepsize * b11, *cstate, current_dm_dt,
              next_state_key.GetWriteReference(), dummy_error, 
              dev_MValue_backup, host_struct.dev_MValue, false);
  GPU_GetEnergyDensity(next_state_key.GetReadReference(), temp_energy,
                   &vtmpA, NULL, pE_pt);
  Calculate_dm_dt(next_state_key.GetReadReference(), vtmpA, pE_pt,
                  vtmpA, max_dm_dt,dE_dt, timestep_lower_bound, false);

  // Step 2
  vtmpB = current_dm_dt;
  // vtmpB *= b21;
  // vtmpB.Accumulate(b22, vtmpA);
  backUpEnergy(host_struct.dev_dm_dt_bak, host_struct.dev_dm_dt, 3 * size);
  accumulate(grid_size, block_size, size, b21, b22, host_struct.dev_dm_dt,
    dev_dm_dt_backup, host_struct.dev_dm_dt);
  AdjustState(stepsize * a2, stepsize, *cstate, vtmpB,
              next_state_key.GetWriteReference(), dummy_error, 
              dev_MValue_backup, host_struct.dev_MValue, false);

  // Step 3
  GPU_GetEnergyDensity(next_state_key.GetReadReference(), temp_energy,
                   &vtmpB, NULL, pE_pt);
  Calculate_dm_dt(next_state_key.GetReadReference(), vtmpB, pE_pt,
                  vtmpB, max_dm_dt, dE_dt, timestep_lower_bound, false);
// vtmpA --> host_struct.dev_dm_dt_bak, vtmpB --> host_struct.dev_dm_dt
  // Step 4
  vtmpC = current_dm_dt;
  // vtmpC *= b31;
  // vtmpC.Accumulate(b32, vtmpA);
  accumulate(grid_size, block_size, size, b31, b32, host_struct.dev_dm_dt_bak,
    dev_dm_dt_backup, dev_dm_dt_backup2);
  // vtmpC.Accumulate(b33, vtmpB);
  backUpEnergy(dev_dm_dt_backup3, host_struct.dev_dm_dt, 3 * size);
  accumulate(grid_size, block_size, size, 1.0, b33, dev_dm_dt_backup3,
    dev_dm_dt_backup2, host_struct.dev_dm_dt);
// vtmpA --> host_struct.dev_dm_dt_bak, vtmpB --> dev_dm_dt_backup3
// vtmpC --> host_struct.dev_dm_dt, current_dm_dt --> dev_dm_dt_backup
  AdjustState(stepsize * a3, stepsize, *cstate, vtmpC,
              next_state_key.GetWriteReference(), dummy_error, 
              dev_MValue_backup, host_struct.dev_MValue, false);
  // Step 5
  GPU_GetEnergyDensity(next_state_key.GetReadReference(), temp_energy,
                   &vtmpC, NULL, pE_pt);
  Calculate_dm_dt(next_state_key.GetReadReference(), vtmpC, pE_pt,
                  vtmpC, max_dm_dt, dE_dt, timestep_lower_bound, false);

  // Step 6
  vtmpD = current_dm_dt;
  // vtmpD *= b41;
  // vtmpD.Accumulate(b42, vtmpA);
  backUpEnergy(dev_dm_dt_backup2, host_struct.dev_dm_dt, 3 * size);
  accumulate(grid_size, block_size, size, b41, b42, host_struct.dev_dm_dt_bak,
    dev_dm_dt_backup, host_struct.dev_dm_dt);
  // vtmpD.Accumulate(b43, vtmpB);
  accumulate(grid_size, block_size, size, 1.0, b43, dev_dm_dt_backup3,
    host_struct.dev_dm_dt, host_struct.dev_dm_dt);
  // vtmpD.Accumulate(b44, vtmpC);
  accumulate(grid_size, block_size, size, 1.0, b44, dev_dm_dt_backup2,
    host_struct.dev_dm_dt, host_struct.dev_dm_dt);
// vtmpA --> host_struct.dev_dm_dt_bak, vtmpB --> dev_dm_dt_backup3
// vtmpC --> dev_dm_dt_backup2, current_dm_dt --> dev_dm_dt_backup
// vtmpD --> host_struct.dev_dm_dt
  AdjustState(stepsize * a4, stepsize, *cstate, vtmpD,
              next_state_key.GetWriteReference(), dummy_error, 
              dev_MValue_backup, host_struct.dev_MValue, false);

  // Step 7
  GPU_GetEnergyDensity(next_state_key.GetReadReference(), temp_energy,
                   &vtmpD, NULL, pE_pt);
  Calculate_dm_dt(next_state_key.GetReadReference(), vtmpD, pE_pt,
                  vtmpD, max_dm_dt, dE_dt, timestep_lower_bound, false);
  // Array holdings: A=dm_dt2   B=dm_dt3   C=dm_dt4   D=dm_dt5

  // Step 8
  // vtmpA *= b52;
  // vtmpA.Accumulate(b51, current_dm_dt);
  accumulate(grid_size, block_size, size, b51, b52, host_struct.dev_dm_dt_bak,
    dev_dm_dt_backup, host_struct.dev_dm_dt_bak);
  // vtmpA.Accumulate(b53, vtmpB);
  accumulate(grid_size, block_size, size, 1.0, b53, dev_dm_dt_backup3,
    host_struct.dev_dm_dt_bak, host_struct.dev_dm_dt_bak);
  // vtmpA.Accumulate(b54, vtmpC);
  accumulate(grid_size, block_size, size, 1.0, b54, dev_dm_dt_backup2,
    host_struct.dev_dm_dt_bak, host_struct.dev_dm_dt_bak);
  // vtmpA.Accumulate(b55, vtmpD);
  accumulate(grid_size, block_size, size, 1.0, b55, host_struct.dev_dm_dt,
    host_struct.dev_dm_dt_bak, host_struct.dev_dm_dt_bak);
  // swap dev_dm_dt and dev_dm_dt_bak pointer before updating dev_dm_dt
  FD_TYPE *dev_dm_dt_tmp = host_struct.dev_dm_dt_bak;
  host_struct.dev_dm_dt_bak = host_struct.dev_dm_dt;
  host_struct.dev_dm_dt = dev_dm_dt_tmp;
// vtmpA --> host_struct.dev_dm_dt, vtmpB --> dev_dm_dt_backup3
// vtmpC --> dev_dm_dt_backup2, current_dm_dt --> dev_dm_dt_backup
// vtmpD --> host_struct.dev_dm_dt_bak
  AdjustState(stepsize, stepsize, *cstate, vtmpA,
              next_state_key.GetWriteReference(), dummy_error, 
              dev_MValue_backup, host_struct.dev_MValue, false); // a5=1.0

  // Step 9
  GPU_GetEnergyDensity(next_state_key.GetReadReference(), temp_energy,
                   &vtmpA, NULL, pE_pt);
  Calculate_dm_dt(next_state_key.GetReadReference(), vtmpA, pE_pt,
                  vtmpA,max_dm_dt, dE_dt, timestep_lower_bound, false);
  // Array holdings: A=dm_dt6   B=dm_dt3   C=dm_dt4   D=dm_dt5

  // Step 10
  // OC_INDEX i;
  // for(i=0;i<size;i++) { // ***CAN BE DEFINED ON GPU***
    // ThreeVector dm_dt3 = vtmpB[i];
    // ThreeVector dm_dt6 = vtmpA[i];
    // vtmpA[i] = b63 * dm_dt3  +  b66 * dm_dt6;  // k6(3,6)
    // vtmpB[i] = dc3 * dm_dt3  +  dc6 * dm_dt6;  // dD(3,6)
  // }
  backUpEnergy(dev_dm_dt_backup4, host_struct.dev_dm_dt, 3 * size);
  accumulate(grid_size, block_size, size, b63, b66, host_struct.dev_dm_dt,
    dev_dm_dt_backup3, host_struct.dev_dm_dt);
  accumulate(grid_size, block_size, size, dc3, dc6, dev_dm_dt_backup4,
    dev_dm_dt_backup3, dev_dm_dt_backup3);
  // Array holdings: A=k6(3,6)   B=dD(3,6)   C=dm_dt4   D=dm_dt5

  // Step 11
  // vtmpA.Accumulate(b61, current_dm_dt);   // Note: b62=0.0
  accumulate(grid_size, block_size, size, b61, 1.0, host_struct.dev_dm_dt,
    dev_dm_dt_backup, host_struct.dev_dm_dt);
  // vtmpA.Accumulate(b64, vtmpC);
  accumulate(grid_size, block_size, size, b64, 1.0, host_struct.dev_dm_dt,
    dev_dm_dt_backup2, host_struct.dev_dm_dt);
  // vtmpA.Accumulate(b65, vtmpD);
  accumulate(grid_size, block_size, size, b65, 1.0, host_struct.dev_dm_dt,
    host_struct.dev_dm_dt_bak, host_struct.dev_dm_dt);
  AdjustState(stepsize, stepsize, *cstate, vtmpA,
              next_state_key.GetWriteReference(), norm_error, 
              dev_MValue_backup, host_struct.dev_MValue, false); // a6=1.0
  const Oxs_SimState& endstate
    = next_state_key.GetReadReference(); // Candidate next state
  // OC_REAL8m total_E;
  GPU_GetEnergyDensity(endstate, temp_energy, &mxH_output.cache.value,
                   NULL, pE_pt); //, total_E
  // mxH_output.cache.state_id=endstate.Id();///////////////
  
  // compute total_E
  FD_TYPE *dev_dE = dev_info + 3;
  FD_TYPE *dev_var_dE = dev_info + 4;
  FD_TYPE *dev_total_E = dev_info + 5;
  collectEnergyStatistics(grid_size, block_size, size, reduce_size, BLK_SIZE,
    host_struct, dev_dE, dev_var_dE, dev_total_E);
  
  if((mxH_output.GetCacheRequestCount() > 0
        && mxH_output.cache.state_id != endstate.Id())) {
    FD_TYPE *tmp_mxH = new FD_TYPE[size * 3];
    memDownload_device(tmp_mxH, host_struct.dev_torque, 3 * size, DEV_NUM);
    Oxs_MeshValue<ThreeVector>& mxH = mxH_output.cache.value;
    for(int i = 0; i < size; i++) {
      mxH[i] = ThreeVector(tmp_mxH[i], tmp_mxH[i+size], tmp_mxH[i+2*size]);
    }
    if(tmp_mxH) delete[] tmp_mxH;
    mxH_output.cache.state_id = endstate.Id();
  }
  
  Calculate_dm_dt(endstate, mxH_output.cache.value, pE_pt,
                  vtmpA, max_dm_dt, dE_dt, timestep_lower_bound, false);
  // Array holdings: A=dm_dt7   B=dD(3,6)   C=dm_dt4   D=dm_dt5

  // Step 12
  // OC_REAL8m max_dD_sq=0.0;
  // vtmpB.Accumulate(dc1, current_dm_dt);
  accumulate(grid_size, block_size, size, 1.0, dc1, dev_dm_dt_backup,
    dev_dm_dt_backup3, dev_dm_dt_backup3);
  // vtmpB.Accumulate(dc4, vtmpC);
  accumulate(grid_size, block_size, size, 1.0, dc4, dev_dm_dt_backup2,
    dev_dm_dt_backup3, dev_dm_dt_backup3);
  // vtmpB.Accumulate(dc5, vtmpD);
  accumulate(grid_size, block_size, size, 1.0, dc5, host_struct.dev_dm_dt_bak,
    dev_dm_dt_backup3, dev_dm_dt_backup3);
  // vtmpB.Accumulate(dc7, vtmpA);
  accumulate(grid_size, block_size, size, 1.0, dc7, host_struct.dev_dm_dt,
    dev_dm_dt_backup3, dev_dm_dt_backup3);
  // Array holdings: A=dm_dt7   B=dD   C=dm_dt4   D=dm_dt5
// vtmpA --> host_struct.dev_dm_dt, vtmpB --> dev_dm_dt_backup3
// vtmpC --> dev_dm_dt_backup2, current_dm_dt --> dev_dm_dt_backup
// vtmpD --> host_struct.dev_dm_dt_bak
  // next_state holds candidate next state, normalized and
  // with proper time field settings; see Step 11.  Note that
  // Step 11 also set norm_error.

  // Error estimate is max|m2a-m2b| = h*max|dD|
  // for(i=0;i<size;i++) { // ***CAN BE DEFINED ON GPU***
    // OC_REAL8m magsq = vtmpB[i].MagSq();
    // if(magsq>max_dD_sq) max_dD_sq = magsq;
  // }
  memPurge_device(dev_dm_dt_backup2, 3 * size, DEV_NUM);
  FD_TYPE *dev_max_dD_sq = dev_info + 2;
  maxDiff(grid_size, block_size, size, reduce_size, BLK_SIZE, dev_dm_dt_backup3,
    dev_dm_dt_backup2, host_struct.dev_local_sum, dev_max_dD_sq);
  
  FD_TYPE host_info[8];
  memDownload_device(host_info, dev_info, 8, DEV_NUM);  
  // host_info[0, 1] --> outputs from Calculate_dm_dt
  const OC_REAL8m max_dm_dt_sq = host_info[0];
  const OC_REAL8m dE_dt_sum = host_info[1];
  max_dm_dt = sqrt(max_dm_dt_sq);
  dE_dt = -1 * MU0 * dE_dt_sum + pE_pt;
  timestep_lower_bound = PositiveTimestepBound(max_dm_dt);
  // host_info[2] --> output from maxDiff
  const OC_REAL8m max_dD_sq = host_info[2];
  error_estimate = stepsize * sqrt(max_dD_sq);
  // host_info[3, 4, 5] --> outputs from energyStatistics
  energyStatistics[0] = host_info[3];
  energyStatistics[1] = host_info[4];
  energyStatistics[2] = host_info[5];
  const OC_REAL8m total_E = energyStatistics[2];
  // host_info[6, 7] --> outputs from AdjustState
  const OC_REAL8m min_normsq = host_info[6];
  const OC_REAL8m max_normsq = host_info[7];
    
  norm_error = OC_MAX(sqrt(max_normsq)-1.0, 1.0 - sqrt(min_normsq));
    
  if(!endstate.AddDerivedData("Timestep lower bound",
                                timestep_lower_bound) ||
     !endstate.AddDerivedData("Max dm/dt",max_dm_dt) ||
     !endstate.AddDerivedData("pE/pt",pE_pt) ||
     !endstate.AddDerivedData("Total E",total_E) ||
     !endstate.AddDerivedData("dE/dt",dE_dt)) {
    throw Oxs_ExtError(this,
                 "Oxs_GPU_RungeKuttaEvolve::RungeKuttaFehlbergBase54:"
                 " Programming error; data cache already set.");
  }

  global_error_order = 5.0;

  new_energy_and_dmdt_computed = 1;

#if REPORT_TIME_RKDEVEL
timer[1].Stop(); /**/
++timer_counts[1].pass_count;
 timer_counts[1].name = "RKFB54 total";
#endif // REPORT_TIME_RKDEVEL

}

void Oxs_GPU_RungeKuttaEvolve::TakeRungeKuttaFehlbergStep54
(OC_REAL8m stepsize,
 Oxs_ConstKey<Oxs_SimState> current_state_key,
 const Oxs_MeshValue<ThreeVector>& current_dm_dt,
 Oxs_Key<Oxs_SimState>& next_state_key,
 OC_REAL8m& error_estimate,OC_REAL8m& global_error_order,
 OC_REAL8m& norm_error,
 OC_BOOL& new_energy_and_dmdt_computed) {
  RungeKuttaFehlbergBase54(RK547FC,stepsize,
     current_state_key,current_dm_dt,next_state_key,
     error_estimate,global_error_order,norm_error,
     new_energy_and_dmdt_computed);
}

void Oxs_GPU_RungeKuttaEvolve::TakeRungeKuttaFehlbergStep54M
(OC_REAL8m stepsize,
 Oxs_ConstKey<Oxs_SimState> current_state_key,
 const Oxs_MeshValue<ThreeVector>& current_dm_dt,
 Oxs_Key<Oxs_SimState>& next_state_key,
 OC_REAL8m& error_estimate,OC_REAL8m& global_error_order,
 OC_REAL8m& norm_error,
 OC_BOOL& new_energy_and_dmdt_computed) {
  RungeKuttaFehlbergBase54(RK547FM,stepsize,
     current_state_key,current_dm_dt,next_state_key,
     error_estimate,global_error_order,norm_error,
     new_energy_and_dmdt_computed);
}

void Oxs_GPU_RungeKuttaEvolve::TakeRungeKuttaFehlbergStep54S
(OC_REAL8m stepsize,
 Oxs_ConstKey<Oxs_SimState> current_state_key,
 const Oxs_MeshValue<ThreeVector>& current_dm_dt,
 Oxs_Key<Oxs_SimState>& next_state_key,
 OC_REAL8m& error_estimate,OC_REAL8m& global_error_order,
 OC_REAL8m& norm_error,
 OC_BOOL& new_energy_and_dmdt_computed) {
  RungeKuttaFehlbergBase54(RK547FS,stepsize,
     current_state_key,current_dm_dt,next_state_key,
     error_estimate,global_error_order,norm_error,
     new_energy_and_dmdt_computed);
}

OC_REAL8m Oxs_GPU_RungeKuttaEvolve::MaxDiff
(const Oxs_MeshValue<ThreeVector>& vecA,
 const Oxs_MeshValue<ThreeVector>& vecB) {
  throw Oxs_ExtError(this, "MaxDiff GPU is not ready");
  
  OC_INDEX size = vecA.Size();
  if(vecB.Size()!=size) {
    throw Oxs_ExtError(this,
                 "Oxs_GPU_RungeKuttaEvolve::MaxDiff:"
                 " Import MeshValues incompatible (different lengths).");
  }
  OC_REAL8m max_magsq = 0.0;
  for(OC_INDEX i=0;i<size;i++) { // ***CAN BE EASILIY DEFINED ON GPU, but check where it is used, maybe already defined***
    ThreeVector vtemp = vecB[i] - vecA[i];
    OC_REAL8m magsq = vtemp.MagSq();
    if(magsq>max_magsq) max_magsq = magsq;
  }
  return sqrt(max_magsq);
}

void Oxs_GPU_RungeKuttaEvolve::AdjustStepHeadroom(OC_INT4m step_reject) { // step_reject should be 0 or 1, reflecting whether the current
  // step was rejected or not.  This routine updates reject_ratio
  // and adjusts step_headroom appropriately.

  // First adjust reject_ratio, weighing mostly the last
  // thirty or so results.
  reject_ratio = (31*reject_ratio + step_reject)/32.;

  // Adjust step_headroom
  if(reject_ratio>reject_goal && step_reject>0) {
    // Reject ratio too high and getting worse
    step_headroom *= 0.925;
  }
  if(reject_ratio<reject_goal && step_reject<1) {
    // Reject ratio too small and getting smaller
    step_headroom *= 1.075;
  }

  if(step_headroom>max_step_headroom) step_headroom=max_step_headroom;
  if(step_headroom<min_step_headroom) step_headroom=min_step_headroom;
}

////////////////////////////////////////////////////////////////////////
/// Oxs_GPU_RungeKuttaEvolve::ComputeEnergyChange  /////////////////////////
///    non-threaded version      /////////////////////////
////////////////////////////////////////////////////////////////////////
void Oxs_GPU_RungeKuttaEvolve::ComputeEnergyChange
(const Oxs_Mesh* mesh,
 const Oxs_MeshValue<OC_REAL8m>& current_energy,
 const Oxs_MeshValue<OC_REAL8m>& candidate_energy,
 OC_REAL8m& dE,OC_REAL8m& var_dE,OC_REAL8m& total_E) { // Computes cellwise difference between energies, and variance.
  
  // Export total_E is "current" energy (used for stepsize control).
  // Nb_Xpfloat dE_xp      = 0.0;
  // Nb_Xpfloat var_dE_xp  = 0.0;
  // Nb_Xpfloat total_E_xp = 0.0;
  // const OC_INDEX size = mesh->Size();
  // for(OC_INDEX i=0;i<size;++i) { // ***ALREADY DEFINED ON GPU***
    // OC_REAL8m vol = mesh->Volume(i);
    // OC_REAL8m e = vol*current_energy[i];
    // OC_REAL8m new_e = vol*candidate_energy[i];
    // total_E_xp += e;
    // dE_xp += new_e - e;
    // var_dE_xp += new_e*new_e + e*e;
  // }
  // total_E = total_E_xp.GetValue();
  // dE      = dE_xp.GetValue();
  // var_dE  = var_dE_xp.GetValue();
  const OC_INDEX size = mesh->Size();
  FD_TYPE *dev_dE = dev_info + 3; //dev_E_info;
  FD_TYPE *dev_var_dE = dev_info + 4;
  FD_TYPE *dev_total_E = dev_info + 5;
  collectEnergyStatistics(grid_size, block_size, size, reduce_size, BLK_SIZE,
    host_struct, dev_dE, dev_var_dE, dev_total_E);
    
  FD_TYPE host_E_change_info[6];
  memDownload_device(host_E_change_info, dev_info, 6, DEV_NUM);
  dE      = host_E_change_info[3];
  var_dE  = host_E_change_info[4];
  total_E = host_E_change_info[5];
}

OC_BOOL Oxs_GPU_RungeKuttaEvolve::InitNewStage
(const Oxs_GPU_TimeDriver* /* driver */,
 Oxs_ConstKey<Oxs_SimState> state,
 Oxs_ConstKey<Oxs_SimState> prevstate) {
  // Update derived data in state.
  const Oxs_SimState& cstate = state.GetReadReference();
  const Oxs_SimState* pstate_ptr = prevstate.GetPtr();
  UpdateDerivedOutputs(cstate,pstate_ptr);

  // Note 1: state is a copy-by-value import, so its read lock
  //         will be released on exit.
  // Note 2: pstate_ptr will be NULL if prevstate has
  //         "INVALID" status.

  return 1;
}

OC_BOOL Oxs_GPU_RungeKuttaEvolve::Step(const Oxs_GPU_TimeDriver* driver,
                      Oxs_ConstKey<Oxs_SimState> current_state_key,
                      const Oxs_DriverStepInfo& step_info,
                      Oxs_Key<Oxs_SimState>& next_state_key,
                      DEVSTRUCT &host_struct_copy) {
#if REPORT_TIME
  steponlytime.Start();
#endif // REPORT_TIME

  const OC_REAL8m bad_energy_cut_ratio = 0.75;
  const OC_REAL8m bad_energy_step_increase = 1.3;

  const OC_REAL8m previous_next_timestep = next_timestep;

  const Oxs_SimState& cstate = current_state_key.GetReadReference();

  if (!allocated) {
    InitGPU(cstate.spin.Size());
  }
  
  host_struct_copy = host_struct;

  CheckCache(cstate);
#ifdef CHOOSESINGLE
  #define EPS OC_REAL4_EPSILON
#elif defined(CHOOSEDOUBLE)
  #define EPS OC_REAL8_EPSILON
#endif
  // Note if start_dm or start_dt is being used
  OC_BOOL start_cond_active=0;
  if(next_timestep<=0.0 ||
     (cstate.stage_iteration_count<1
      && step_info.current_attempt_count==0)) {
    if(cstate.stage_number==0
       || stage_init_step_control == SISC_START_COND) {
      start_cond_active = 1;
    } else if(stage_init_step_control == SISC_CONTINUOUS) {
      start_cond_active = 0;  // Safety
    } else if(stage_init_step_control == SISC_AUTO) {
      // Automatic detection based on energy values across
      // stage boundary.
      OC_REAL8m total_E,E_diff;
      if(cstate.GetDerivedData("Total E",total_E) &&
         cstate.GetDerivedData("Delta E",E_diff)  &&
         fabs(E_diff) <= 256*EPS*fabs(total_E) ) {
        // The factor of 256 in the preceding line is a fudge factor,
        // selected with no particular justification.
        start_cond_active = 0;  // Continuous case
      } else {
        start_cond_active = 1;  // Assume discontinuous
      }
    } else {
      throw Oxs_ExtError(this,
           "Oxs_GPU_RungeKuttaEvolve::Step; Programming error:"
           " unrecognized stage_init_step_control value");
    }
  }
#ifdef EPS
  #undef EPS
#endif
  // Negotiate timestep, and also initialize both next_state and
  // temp_state structures.
  Oxs_SimState* work_state = &(next_state_key.GetWriteReference());
  OC_BOOL force_step=0,driver_set_step=0;
  NegotiateTimeStep(driver,cstate,*work_state,next_timestep,
                    start_cond_active,force_step,driver_set_step);
  OC_REAL8m stepsize = work_state->last_timestep;
  work_state=NULL; // Safety: disable pointer

  // Step
  OC_REAL8m error_estimate,norm_error;
  OC_REAL8m global_error_order;
  OC_BOOL new_energy_and_dmdt_computed;
  OC_BOOL reject_step=0;
  (this->*rkstep_ptr)(stepsize,current_state_key,
                      dm_dt_output.cache.value,
                      next_state_key,
                      error_estimate,global_error_order,norm_error,
                      new_energy_and_dmdt_computed);
  const Oxs_SimState& nstate = next_state_key.GetReadReference();
  driver->FillStateDerivedData(cstate,nstate);

  OC_REAL8m max_dm_dt;
  cstate.GetDerivedData("Max dm/dt",max_dm_dt);
  OC_REAL8m reference_stepsize = stepsize;
  if(driver_set_step) reference_stepsize = previous_next_timestep;
  OC_BOOL good_step = CheckError(global_error_order,error_estimate,
                              stepsize,reference_stepsize,
                              max_dm_dt,next_timestep);
  OC_REAL8m timestep_grit = PositiveTimestepBound(max_dm_dt);
  /// Note: Might want to use average or larger of max_dm_dt
  /// and new_max_dm_dt (computed below.)

  if(!good_step && !force_step) {
    // Bad step; The only justfication to do energy and dm_dt
    // computation would be to get an energy-based stepsize
    // adjustment estimate, which we only need to try if 
    // next_timestep is larger than cut applied by energy
    // rejection code (i.e., bad_energy_cut_ratio).
    if(next_timestep<=stepsize*bad_energy_cut_ratio) {
      AdjustStepHeadroom(1);
#if REPORT_TIME
      steponlytime.Stop();
#endif // REPORT_TIME
      return 0; // Don't bother with energy calculation
    }
    reject_step=1; // Otherwise, mark step rejected and see what energy
    /// info suggests for next stepsize
  }

  if(start_cond_active && !force_step) {
    if(start_dm>=0.0) {
      // Check that no spin has moved by more than start_dm
        
      FD_TYPE *dev_max_error_sq = dev_info + 2;
      maxDiff(grid_size, block_size, cstate.spin.Size(), reduce_size, BLK_SIZE, 
        dev_MValue_backup, host_struct.dev_MValue, host_struct.dev_local_sum, 
        dev_max_error_sq);
      FD_TYPE cpu_max_error_sq[1];
      memDownload_device(cpu_max_error_sq, dev_max_error_sq, 1, DEV_NUM);
      OC_REAL8m max_magsq = cpu_max_error_sq[0];
      OC_REAL8m diff = sqrt(max_magsq);
      // OC_REAL8m diff = MaxDiff(cstate.spin,nstate.spin);
      if(diff>start_dm) {
        next_timestep = step_headroom * stepsize * (start_dm/diff);
        if(next_timestep<=stepsize*bad_energy_cut_ratio) {
          AdjustStepHeadroom(1);
#if REPORT_TIME
          steponlytime.Stop();
#endif // REPORT_TIME
          return 0; // Don't bother with energy calculation
        }
        reject_step=1; // Otherwise, mark step rejected and see what energy
        /// info suggests for next stepsize
      }
    }
  }

#ifdef OLDE_CODE
  if(norm_error>0.0005) {
    fprintf(stderr,
            "Iteration %u passed error check; norm_error=%8.5f\n",
            nstate.iteration_count,norm_error);
  } /**/
#endif // OLDE_CODE

  // Energy timestep control:
  //   The relationship between energy error and stepsize appears to be
  // highly nonlinear, so that estimating appropriate stepsize from energy
  // increase is difficult.  Perhaps it is possible to include energy
  // interpolation into RK step routines, but for the present we just
  // reduce the step by a fixed ratio if we detect energy increase beyond
  // that which can be attributed to numerical errors.  Of course, this
  // doesn't take into account the expected energy decrease (which depends
  // on the damping ratio alpha), which is another reason to try to build
  // it into the high order RK step routines.
  OC_REAL8m pE_pt,new_pE_pt=0.;
  cstate.GetDerivedData("pE/pt",pE_pt);
  OC_REAL8m dE, var_dE, total_E;
  if(new_energy_and_dmdt_computed) {
    nstate.GetDerivedData("pE/pt",new_pE_pt);
    dE = energyStatistics[0];
    var_dE = energyStatistics[1];
    total_E = energyStatistics[2];
  } else {
    OC_REAL8m new_total_E;
    int size = cstate.spin.Size();
    GPU_GetEnergyDensity(nstate, temp_energy, &mxH_output.cache.value, NULL,
      new_pE_pt);
      
    ComputeEnergyChange(nstate.mesh, energy, temp_energy, dE, var_dE, total_E);
    new_total_E = total_E;
    if((mxH_output.GetCacheRequestCount()>0
          && mxH_output.cache.state_id != nstate.Id())) {
      FD_TYPE *tmp_mxH = new FD_TYPE[size * 3];
      memDownload_device(tmp_mxH, host_struct.dev_torque, 3 * size, DEV_NUM);
      Oxs_MeshValue<ThreeVector>& mxH = mxH_output.cache.value;
      for(int i = 0; i < size; i++) {
        mxH[i] = ThreeVector(tmp_mxH[i], tmp_mxH[i+size], tmp_mxH[i+2*size]);
      }
      if(tmp_mxH) delete[] tmp_mxH;
      mxH_output.cache.state_id = nstate.Id();
    }
    // mxH_output.cache.state_id=nstate.Id();
    if(!nstate.AddDerivedData("pE/pt",new_pE_pt)) {
      throw Oxs_ExtError(this,
           "Oxs_GPU_RungeKuttaEvolve::Step:"
           " Programming error; data cache (pE/pt) already set.");
    }
    if(!nstate.AddDerivedData("Total E",new_total_E)) {
      throw Oxs_ExtError(this,
           "Oxs_GPU_RungeKuttaEvolve::Step:"
           " Programming error; data cache (Total E) already set.");
    }
  }

#if REPORT_TIME_RKDEVEL
timer[0].Start(); /**/
#endif // REPORT_TIME_RKDEVEL

  if(!nstate.AddDerivedData("Delta E",dE)) {
    throw Oxs_ExtError(this,
         "Oxs_GPU_RungeKuttaEvolve::Step:"
         " Programming error; data cache (Delta E) already set.");
  }
#if REPORT_TIME_RKDEVEL
timer[0].Stop(); /**/
++timer_counts[0].pass_count;
 timer_counts[0].bytes += (nstate.mesh->Size())*(2*sizeof(OC_REAL8m));
 timer_counts[0].name = "ComputeEnergyChange";
#endif // REPORT_TIME_RKDEVEL

  if(expected_energy_precision>=0.) {
    var_dE *= expected_energy_precision * expected_energy_precision;
    /// Variance, assuming error in each energy[i] term is independent,
    /// uniformly distributed, 0-mean, with range
    ///        +/- expected_energy_precision*energy[i].
    /// It would probably be better to get an error estimate directly
    /// from each energy term.
    OC_REAL8m E_numerror = OC_MAX(fabs(total_E)*expected_energy_precision,
                               2*sqrt(var_dE));
    OC_REAL8m pE_pt_max = OC_MAX(pE_pt,new_pE_pt); // Might want to
    /// change this from a constant to a linear function in timestep.

    OC_REAL8m reject_dE = 2*E_numerror + pE_pt_max * stepsize;
    if(dE>reject_dE) {
      OC_REAL8m teststep = bad_energy_cut_ratio*stepsize;
      if(teststep<next_timestep) {
        next_timestep=teststep;
        max_step_increase = bad_energy_step_increase;
        // Damp the enthusiasm of the RK stepper routine for
        // stepsize growth, for a step or two.
      }
      if(!force_step) reject_step=1;
    }
  }

  // Guarantee that next_timestep is large enough to move
  // at least one spin by an amount perceptible to the
  // floating point representation.
  if(next_timestep<timestep_grit) next_timestep = timestep_grit;

  if(!force_step && reject_step) {
    AdjustStepHeadroom(1);
#if REPORT_TIME
    steponlytime.Stop();
#endif // REPORT_TIME
    return 0;
  }

  // Otherwise, we are accepting the new step.

  // Calculate dm_dt at new point, and fill in cache.
  if(new_energy_and_dmdt_computed) {
    // dm_dt_output.cache.value.Swap(vtmpA);
  } else {
    OC_REAL8m new_max_dm_dt,new_dE_dt,new_timestep_lower_bound;
    // if(mxH_output.cache.state_id != nstate.Id()) { // Safety
      // throw Oxs_ExtError(this,
           // "Oxs_GPU_RungeKuttaEvolve::Step:"
           // " Programming error; mxH_output cache improperly filled.");
    // }
    // Calculate_dm_dt(nstate,
                    // mxH_output.cache.value,new_pE_pt,
                    // dm_dt_output.cache.value,new_max_dm_dt,
                    // new_dE_dt,new_timestep_lower_bound);
    Calculate_dm_dt(nstate, mxH_output.cache.value, new_pE_pt,
                    dm_dt_output.cache.value, new_max_dm_dt, new_dE_dt, 
                    new_timestep_lower_bound, true); //Since 
    // all the exports are used, we transfer GPU memory inside the subroutine, 
    // but it can be combined outside, if necessary
    if(!nstate.AddDerivedData("Timestep lower bound",
                              new_timestep_lower_bound) ||
       !nstate.AddDerivedData("Max dm/dt",new_max_dm_dt) ||
       !nstate.AddDerivedData("dE/dt",new_dE_dt)) {
      throw Oxs_ExtError(this,
                           "Oxs_GPU_RungeKuttaEvolve::Step:"
                           " Programming error; data cache already set.");
    }
  }
  if((dm_dt_output.GetCacheRequestCount()>0
      && dm_dt_output.cache.state_id != nstate.Id())) {

    int size = nstate.spin.Size();
    Oxs_MeshValue<ThreeVector>& dm_dt = dm_dt_output.cache.value;
    FD_TYPE *tmp_dm_dt = new FD_TYPE[size * 3];
    memDownload_device(tmp_dm_dt, host_struct.dev_dm_dt, size * 3, DEV_NUM);
    for(int i=0; i<size; i++){
      dm_dt[i] = ThreeVector(tmp_dm_dt[i], tmp_dm_dt[i+size], tmp_dm_dt[i+2*size]);
    }
    if(tmp_dm_dt) delete[] tmp_dm_dt;
    dm_dt_output.cache.state_id = nstate.Id();
  }
  // dm_dt_output.cache.state_id = nstate.Id();

  energy.Swap(temp_energy);
  energy_state_id = nstate.Id();

  AdjustStepHeadroom(0);
  if(!force_step && max_step_increase<max_step_increase_limit) {
    max_step_increase *= max_step_increase_adj_ratio;
  }
  if(max_step_increase>max_step_increase_limit) {
    max_step_increase = max_step_increase_limit;
  }

#if REPORT_TIME
  steponlytime.Stop();
#endif // REPORT_TIME
  return 1; // Accept step
}

void Oxs_GPU_RungeKuttaEvolve::UpdateDerivedOutputs(const Oxs_SimState& state,
    const Oxs_SimState* prevstate_ptr) {
  // This routine fills all the Oxs_RungeKuttaEvolve Oxs_ScalarOutput's to
  // the appropriate value based on the import "state", and any of
  // Oxs_VectorOutput's that have CacheRequest enabled are filled.
  // It also makes sure all the expected WOO objects in state are
  // filled.
  
  if (!allocated) { // NO THIS IN GPU_Euler
    InitGPU(state.spin.Size());
  }

  max_dm_dt_output.cache.state_id
    = dE_dt_output.cache.state_id
    = delta_E_output.cache.state_id
    = 0;  // Mark change in progress

  int size = state.mesh->Size();
  OC_REAL8m dummy_value;
  if(!state.GetDerivedData("Max dm/dt",max_dm_dt_output.cache.value) ||
     !state.GetDerivedData("dE/dt",dE_dt_output.cache.value) ||
     !state.GetDerivedData("Delta E",delta_E_output.cache.value) ||
     !state.GetDerivedData("pE/pt",dummy_value) ||
     !state.GetDerivedData("Total E",dummy_value) ||
     !state.GetDerivedData("Timestep lower bound",dummy_value)) {

    // Missing at least some data, so calculate from scratch

    // Calculate H and mxH outputs
    Oxs_MeshValue<ThreeVector>& mxH = mxH_output.cache.value;
    OC_REAL8m pE_pt, total_E;
    // GetEnergyDensity(state, energy, &mxH, NULL, pE_pt, total_E);
    GPU_GetEnergyDensity(state, energy, &mxH, NULL, pE_pt);
    energy_state_id = state.Id();
    /// compute total_E /// computation of dev_dE and dev_var_dE is redundent, but
    /// good of reusing code
    FD_TYPE *dev_dE = dev_info + 3;
    FD_TYPE *dev_var_dE = dev_info + 4;
    FD_TYPE *dev_total_E = dev_info + 5;
    collectEnergyStatistics(grid_size, block_size, size, reduce_size, BLK_SIZE,
      host_struct, dev_dE, dev_var_dE, dev_total_E);
    FD_TYPE cpu_total_E[1];
    memDownload_device(cpu_total_E, dev_info + 5, 1, DEV_NUM);
    total_E = cpu_total_E[0];
    /// compute total_E ///
    // mxH_output.cache.state_id = state.Id();
    if(!state.GetDerivedData("pE/pt", dummy_value)) {
      state.AddDerivedData("pE/pt", pE_pt);
    }
    if(!state.GetDerivedData("Total E", dummy_value)) {
      state.AddDerivedData("Total E", total_E);
    }

    // Calculate dm/dt, Max dm/dt and dE/dt
    Oxs_MeshValue<ThreeVector>& dm_dt = dm_dt_output.cache.value;
    // dm_dt_output.cache.state_id=0;
    OC_REAL8m timestep_lower_bound;
    Calculate_dm_dt(state,mxH,pE_pt,dm_dt,
                    max_dm_dt_output.cache.value,
                    dE_dt_output.cache.value,timestep_lower_bound, true);
    // dm_dt_output.cache.state_id=state.Id();

    if(!state.GetDerivedData("Max dm/dt",dummy_value)) {
      state.AddDerivedData("Max dm/dt",max_dm_dt_output.cache.value);
    }

    if(!state.GetDerivedData("dE/dt",dummy_value)) {
      state.AddDerivedData("dE/dt",dE_dt_output.cache.value);
    }

    if(!state.GetDerivedData("Timestep lower bound",dummy_value)) {
      state.AddDerivedData("Timestep lower bound",
                           timestep_lower_bound);
    }

    if(!state.GetDerivedData("Delta E",dummy_value)) {
      if(state.previous_state_id == 0) {
        // No previous state
        dummy_value = 0.0;
      } else if(prevstate_ptr!=NULL
                && state.previous_state_id == prevstate_ptr->Id()) {
        OC_REAL8m old_E;
        if(!prevstate_ptr->GetDerivedData("Total E",old_E)) {
          throw Oxs_ExtError(this,
                             "Oxs_GPU_RungeKuttaEvolve::UpdateDerivedOutputs:"
                             " \"Total E\" not set in previous state.");
        }
        dummy_value = total_E - old_E; // This is less accurate than adding
        /// up the cell-by-cell differences (as is used in the main code),
        /// but w/o the entire energy map for prevstate this is the best
        /// we can do.
      } else {
        throw Oxs_ExtError(this,
           "Oxs_GPU_RungeKuttaEvolve::UpdateDerivedOutputs:"
           " Can't derive Delta E from single state.");
      }
      state.AddDerivedData("Delta E",dummy_value);
    }
    delta_E_output.cache.value=dummy_value;

  }

  if((dm_dt_output.GetCacheRequestCount()>0
      && dm_dt_output.cache.state_id != state.Id()) ||
     (mxH_output.GetCacheRequestCount()>0
      && mxH_output.cache.state_id != state.Id())) {
        
    Oxs_MeshValue<ThreeVector>& mxH = mxH_output.cache.value;
    Oxs_MeshValue<ThreeVector>& dm_dt = dm_dt_output.cache.value;
    if (energy_state_id != state.Id()) {
      OC_REAL8m pE_pt;
      GPU_GetEnergyDensity(state, energy, &mxH, NULL, pE_pt);
      energy_state_id=state.Id();
      
      // update dm_dt
      if((dm_dt_output.GetCacheRequestCount()>0
          && dm_dt_output.cache.state_id != state.Id())) {
        // dm_dt_output.cache.state_id=0;
        OC_REAL8m timestep_lower_bound;
        Calculate_dm_dt(state, mxH, pE_pt, dm_dt,
                    max_dm_dt_output.cache.value,
                    dE_dt_output.cache.value, timestep_lower_bound, false);
      }
    }

    if((dm_dt_output.GetCacheRequestCount()>0
          && dm_dt_output.cache.state_id != state.Id())) {

      FD_TYPE *tmp_dm_dt = new FD_TYPE[size * 3];
      memDownload_device(tmp_dm_dt, host_struct.dev_dm_dt, size * 3, DEV_NUM);
      for(int i=0; i<size; i++){
        dm_dt[i] = ThreeVector(tmp_dm_dt[i], tmp_dm_dt[i+size], tmp_dm_dt[i+2*size]);
      }
      if(tmp_dm_dt) delete[] tmp_dm_dt;
      dm_dt_output.cache.state_id = state.Id();
    }
    
    if((mxH_output.GetCacheRequestCount()>0
        && mxH_output.cache.state_id != state.Id())) {
        
      FD_TYPE *tmp_mxH = new FD_TYPE[size * 3];
      memDownload_device(tmp_mxH, host_struct.dev_torque, 3 * size, DEV_NUM);
      for(int i=0; i<size; i++){
        mxH[i] = ThreeVector(tmp_mxH[i], tmp_mxH[i+size], tmp_mxH[i+2*size]);
      }
      if(tmp_mxH) delete[] tmp_mxH;
      mxH_output.cache.state_id = state.Id();
    }
  }

  max_dm_dt_output.cache.value*=(180e-9/PI);
  /// Convert from radians/second to deg/ns

  max_dm_dt_output.cache.state_id
    = dE_dt_output.cache.state_id
    = delta_E_output.cache.state_id
    = state.Id();
}
