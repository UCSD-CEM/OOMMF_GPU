/* FILE: tuw_cvodeevolve.h                 -*-Mode: c++-*-
 *
 * Concrete evolver class, using cvode package
 *
 */

#ifndef _TUW_CVODEEVOLVE
#define _TUW_CVODEEVOLVE

#include <stdio.h>  // for FILE* logfile.

#include "director.h"
#include "ext.h"
#include "key.h"
#include "meshvalue.h"
#include "simstate.h"
#include "output.h"
#include "threevector.h"
#include "timedriver.h"
#include "timeevolver.h"

/* End includes */

// cvode stuff
#include "cvode/cvode.h"

typedef realtype real; 
typedef int integer;

class Tuw_CvodeEvolve;  // Forward reference

// We use void*'s for the second and third argument because some C++
// compilers don't like pointers to anonymous structures.  In the
// tuw_cvodeevolve.cc file there is a wrapper to this function
// that has C-linkage and N_Vector's for the second and third
// argument.
int GetDmDt(real t,void* y, void* ydot,void* datablock);

struct Tuw_CvodeEvolveDataBlock {
public:
  Tuw_CvodeEvolve* evolveptr;
  Tcl_Interp* mif_interp;
  Oxs_Output* state_energy_func;
  Oxs_Key<Oxs_SimState>* state_key; // Varies between cvode calls
  Oxs_MeshValue<OC_REAL8m> energy; // scratch space
  Oxs_MeshValue<ThreeVector> mxH; // scratch space
  OC_BOOL do_precess;  // If false, then do pure damping
  OC_REAL8m gamma;  // Landau-Lifschitz gyromagnetic ratio
  OC_REAL8m alpha;  // Landau-Lifschitz damping coef
  OC_REAL8m max_dm_dt; // GetDmDt export, in rad/sec
  OC_REAL8m total_energy; // Energy for current state.
  Tuw_CvodeEvolveDataBlock() : evolveptr(NULL), mif_interp(NULL),
                          state_energy_func(NULL), state_key(NULL),
                          do_precess(0), gamma(0.), alpha(0.),
                          max_dm_dt(0.), total_energy(0.) {}
  ~Tuw_CvodeEvolveDataBlock() {}
  /// DataBlock does not "own" (i.e., is not responsible for
  /// creating or deleting) any of the pointed to memory structures.
};

class Tuw_CvodeEvolve:public Oxs_TimeEvolver {
  friend int GetDmDt(real,void*,void*,void*);
  /// RHS function passed to the cvode library.
private:
  enum ODESolverStyle { NONSTIFF, STIFF };
  ODESolverStyle style;

  string logfilename;
  FILE* logfile;

  // Base step size control parameters
  OC_REAL8m min_timestep;           // Seconds
  OC_REAL8m max_timestep;           // Seconds
  OC_REAL8m initial_timestep;       // Seconds

  // The following evolution constants are uniform for now.  These
  // should be changed to arrays in the future.
  OC_REAL8m allowed_error; // deg/ns

  // Data cached from last state
  OC_UINT4m state_id;
  OC_UINT4m stage_number; // Needed to detect stage changes, so
                      /// cvode driver can be re-initialized.
  OC_REAL8m stepsize; // Seconds.  First guess at new stepsize
                 /// is last stepsize.

  // Memory for cvode
  Tuw_CvodeEvolveDataBlock datablock;
  OC_BOOL cvode_initialized;
  OC_BOOL cvode_reset_request;
  OC_UINT4m cvode_reset_count;
  OC_UINT4m successful_cvode_count;
  void* cvode_mem;
  N_Vector yout;
  real reltol,abstol;
  OC_REAL8m renormtol;

  // Routine for exact reports.
  void ComputeEnergyAndMaxDmDt(const Oxs_SimState& state,
                               OC_REAL8m& max_dm_dt,
                               OC_REAL8m& total_energy);

  // Outputs
  OC_UINT4m exact_report_period;
  void UpdateDerivedOutputs(const Oxs_SimState&);
  Oxs_ScalarOutput<Tuw_CvodeEvolve> max_dm_dt_output;
  Oxs_ScalarOutput<Tuw_CvodeEvolve> cvode_reset_count_output;
  Oxs_ScalarOutput<Tuw_CvodeEvolve> Delta_E_output;
  // add counter of successful CVode calls
  Oxs_ScalarOutput<Tuw_CvodeEvolve> success_cvode;

public:
  virtual const char* ClassName() const; // ClassName() is
  /// automatically generated by the OXS_EXT_REGISTER macro.
  virtual OC_BOOL Init();
  Tuw_CvodeEvolve(const char* name,     // Child instance id
             Oxs_Director* newdtr, // App director
            const char* argstr);  // MIF input block parameters
  virtual ~Tuw_CvodeEvolve();

  virtual  OC_BOOL
  Step(const Oxs_TimeDriver* driver,
       Oxs_ConstKey<Oxs_SimState> current_state,
       const Oxs_DriverStepInfo& step_info,
       Oxs_Key<Oxs_SimState>& next_state);
  // Returns true if step was successful, false if
  // unable to step as requested.
};

#endif // _TUW_CVODEEVOLVE
