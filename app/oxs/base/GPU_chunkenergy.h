/* FILE: GPU_chunkenergy.h              -*-Mode: c++-*-
 *
 * Abstract GPU chunk energy class, derived from Oxs_ChunkEnergy class.  Children
 * of the Oxs_GPU_ChunkEnergy class include an interface allowing
 * computation on only a specified subrange of the elements (cells) in
 * the state mesh.
 *
 * Note: The friend function Oxs_GPU_ComputeEnergies() is declared in the
 * energy.h header (since its interface only references the base
 * Oxs_Energy class), but the implementation is in GPU_chunkenergy.cc
 * (because the implementation includes accesses to the Oxs_GPU_ChunkEnergy
 * API).
 */

#ifndef _OXS_GPU_CHUNKENERGY
#define _OXS_GPU_CHUNKENERGY

#include <vector>

#include "GPU_energy.h"
#include "chunkenergy.h"
#include "ext.h"
#include "simstate.h"
#include "key.h"
#include "mesh.h"
#include "meshvalue.h"
#include "outputderiv.h"
#include "util.h"

/* End includes */

////////////////////////////////////////////////////////////////////////
// Oxs_GPU_ChunkEnergy class: child class of Oxs_GPU_Energy that supports an
// additional ComputeEnergy interface --- one that allows computation
// on GPU.
class Oxs_GPU_ComputeEnergiesChunkThread;
class Oxs_GPU_ChunkEnergy : public Oxs_GPU_Energy {
  friend void Oxs_GPU_ComputeEnergies(const Oxs_SimState&,
                                  Oxs_ComputeEnergyData&,
                                  const vector<Oxs_Energy*>&,
                                  Oxs_ComputeEnergyExtraData& oceed,
								  DEVSTRUCT& dev_struct);
  friend class Oxs_GPU_ComputeEnergiesChunkThread;
private:
  // Expressly disable default constructor, copy constructor and
  // assignment operator by declaring them without defining them.
  Oxs_GPU_ChunkEnergy();
  Oxs_GPU_ChunkEnergy(const Oxs_GPU_ChunkEnergy&);
  Oxs_GPU_ChunkEnergy& operator=(const Oxs_GPU_ChunkEnergy&);

#if REPORT_TIME
  static Nb_StopWatch chunktime;  // Records time spent computing

  void ReportTime();
#else
  void ReportTime() {}
#endif // REPORT_TIME


protected:
  Oxs_GPU_ChunkEnergy(const char* name,Oxs_Director* newdtr)
    : Oxs_GPU_Energy(name,newdtr) {}
  Oxs_GPU_ChunkEnergy(const char* name,Oxs_Director* newdtr,
                  const char* argstr)
    : Oxs_GPU_Energy(name,newdtr,argstr) {}

  // For a given state, ComputeEnergyChunk (see below) performs the
  // energy/field/torque computation on a subrange of nodes across the
  // mesh.  Before each series of ComputeEnergyChunk calls for a given
  // state, ComputeEnergyChunkInitialize is run in thread 0 to perform
  // any non-threadable, termwise initialization.  For example, memory
  // allocation based on mesh size, or calls into the Tcl interpreter,
  // could be done here.
  //   Similarly, ComputeEnergyChunkFinalize is called at the end of
  // ComputeEnergyChunk processing for a given state.  It is also run in
  // thread 0.  This routine can be used to collate termwise data, such
  // as term-specific output.
  //   Note that as with ComputeEnergyChunk, the *Initialize and
  // *Finalize member functions are const, so only local and mutable
  // data may be modified.
  //   Note also that the default implementation for both *Initialize
  // and *Finalize are NOPs.
  virtual void ComputeEnergyChunkInitialize
  (const Oxs_SimState& /* state */,
  Oxs_ComputeEnergyDataThreaded& /* ocedt */,
  Oxs_ComputeEnergyDataThreadedAux& /* ocedtaux */,
  int /* number_of_threads */) const {}

  virtual void ComputeEnergyChunkFinalize
  (const Oxs_SimState& /* state */,
  const Oxs_ComputeEnergyDataThreaded& /* ocedt */,
  const Oxs_ComputeEnergyDataThreadedAux& /* ocedtaux */,
  int /* number_of_threads */) const {}
  
  virtual void
  GPU_ComputeEnergyChunk(const Oxs_SimState& state,
                     Oxs_ComputeEnergyDataThreaded& ocedt,
                     Oxs_ComputeEnergyDataThreadedAux& ocedtaux,
                     OC_INDEX node_start,OC_INDEX node_stop,
                     int threadnumber, DEVSTRUCT& dev_struct, 
					 OC_INDEX flag_mxHxm) const = 0;
  
  void GPU_ComputeEnergyAlt(const Oxs_SimState& state,
                        Oxs_ComputeEnergyData& oced,
						DEVSTRUCT& dev_struct) const {
#ifdef GPU_DEBUG2
  FILE *debugInfo = fopen("location.txt", "a");
  fprintf(debugInfo, "entering GPU_ComputeEnergyAlt in %s...\n", __FILE__);
  fclose(debugInfo);
#endif	
#if REPORT_TIME
    chunktime.Start();
#endif
    Oxs_ComputeEnergyDataThreaded ocedt(oced);
    Oxs_ComputeEnergyDataThreadedAux ocedtaux;

    ComputeEnergyChunkInitialize(state,ocedt,ocedtaux,1);
	GPU_ComputeEnergyChunk(state,ocedt,ocedtaux,0,state.mesh->Size(),0,
											dev_struct, 0);
    // "Main" thread is presumed; thread_number for main thread is 0.
    ComputeEnergyChunkFinalize(state,ocedt,ocedtaux,1);

    oced.energy_sum = ocedtaux.energy_total_accum;
    oced.pE_pt = ocedtaux.pE_pt_accum;
#if REPORT_TIME
    chunktime.Stop();
#endif
  }

public:
  virtual OC_BOOL Init() { ReportTime(); return Oxs_GPU_Energy::Init(); }
  virtual ~Oxs_GPU_ChunkEnergy() { ReportTime(); }

};

#endif // _OXS_GPU_CHUNKENERGY
