/* FILE: GPU_energy.h                 -*-Mode: c++-*-
 *
 * Abstract energy class, derived from Oxs_Energy class. The declaration
 * an implementation of the Oxs_GPU_Energy child class 
 * Oxs_GPU_ChunkEnergy, is in separate GPU_chunkenergy.h and 
 * GPU_chunkenergy.cc files.
 *
 * Note: The friend function Oxs_GPU_ComputeEnergies() is declared in the
 * GUP_energy.h header (since its interface only references the base
 * Oxs_GPU_Energy class), but the implementation is in GPU_chunkenergy.cc
 * (because the implementation includes accesses to the Oxs_GPU_ChunkEnergy
 * API).
 */

#ifndef _OXS_GPU_ENERGY
#define _OXS_GPU_ENERGY

#include "energy.h"

#include "GPU_devstruct.h"
/* End includes */

////////////////////////////////////////////////////////////////////////
class Oxs_GPU_Energy:public Oxs_Energy {
  friend void Oxs_GPU_ComputeEnergies(const Oxs_SimState& state,
                         Oxs_ComputeEnergyData& oced,
                         const vector<Oxs_Energy*>& energies,
                         Oxs_ComputeEnergyExtraData& oceed,
						 DEVSTRUCT& host_struct);
  // Note: The declaration of this friend function is farther down
  // in this file, but the implementation of Oxs_ComputeEnergies
  // is in the file chunkenergy.cc.
private:
  // Track count of number of times GetEnergy() has been
  // called in current problem run.
  OC_UINT4m calc_count;
#ifdef EXPORT_CALC_COUNT
  // Make calc_count available for output.
  Oxs_ScalarOutput<Oxs_GPU_Energy> calc_count_output;
  void FillCalcCountOutput(const Oxs_SimState&);
#endif // EXPORT_CALC_COUNT

#if REPORT_TIME
  // energytime records time (cpu and wall) spent in the GetEnergyData
  // member function.  The results are written to stderr when this
  // object is destroyed or re-initialized.
  // Note: Timing errors may occur if an exception is thrown from inside
  //       GetEnergyData, because there is currently no attempt made to
  //       catch such exceptions and stop the stopwatch.  This could be
  //       done, but it probably isn't necessary for a facility which
  //       is intended only for development purposes.
protected:
  Nb_StopWatch energytime;
private:
#endif // REPORT_TIME

  void SetupOutputs(); // Utility routine called by constructors.

  // Expressly disable default constructor, copy constructor and
  // assignment operator by declaring them without defining them.
  Oxs_GPU_Energy();
  Oxs_GPU_Energy(const Oxs_GPU_Energy&);
  Oxs_GPU_Energy& operator=(const Oxs_GPU_Energy&);

protected:

  Oxs_GPU_Energy(const char* name,      // Child instance id
             Oxs_Director* newdtr); // App director
  Oxs_GPU_Energy(const char* name,
             Oxs_Director* newdtr,
			 const char* argstr); 
  void UpdateStandardOutputs(const Oxs_SimState&);
  Oxs_ScalarOutput<Oxs_GPU_Energy> energy_sum_output;
  Oxs_VectorFieldOutput<Oxs_GPU_Energy> field_output;
  Oxs_ScalarFieldOutput<Oxs_GPU_Energy> energy_density_output;

   virtual void GetEnergy(const Oxs_SimState& state,
			  Oxs_EnergyData& oed) const {};
  virtual void GPU_GetEnergy(const Oxs_SimState& state,
			 Oxs_EnergyData& oed, DEVSTRUCT& host_struct, 
			 unsigned int flag_outputH,
			 unsigned int flag_outputE, 
       unsigned int flag_outputSumE,
       const OC_BOOL &flag_accum) const {};
  mutable DEVSTRUCT host_struct_copy;//this is problemsome, because the first initial step output is still incorrect
  mutable OC_BOOL initialized;
  virtual void GPU_ComputeEnergy(const Oxs_SimState& state,
							 Oxs_ComputeEnergyData& oced,
							 DEVSTRUCT& host_struct,
               const OC_BOOL &flag_accum) const;
               
  OC_BOOL InitGPU(const Oxs_SimState& state);
					
public:

  virtual ~Oxs_GPU_Energy();
  
  // Default problem initializer routine.  This sets up default
  // output objects, so any child that redefines this function
  // should embed a call to this Init() inside the child
  // specific version.
  virtual OC_BOOL Init();

  // For development:
  OC_UINT4m GetEnergyEvalCount() const { return calc_count; }
};

////////////////////////////////////////////////////////////////////////
// Oxs_GPU_ComputeEnergies compute sums of energies, fields, and/or
// torques for all energies in "energies" import on GPU.  
//   This function is declared here because the interface only requires
// knowledge of the base Oxs_Energy class; however, the implementation
// requires detailed knowledge of the Oxs_GPU_ChunkEnergy class as well,
// so the implementation of this class is in the file GPU_chunkenergy.cc.

void Oxs_GPU_ComputeEnergies(const Oxs_SimState& state,
                         Oxs_ComputeEnergyData& oced,
                         const vector<Oxs_Energy*>& energies,
                         Oxs_ComputeEnergyExtraData& oceed,
						 DEVSTRUCT& host_struct);
#endif // _OXS_GPU_ENERGY
