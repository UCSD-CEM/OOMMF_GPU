/* FILE: GPU_chunkenergy.cc                 -*-Mode: c++-*-
 *
 * Abstract GPU chunk energy class, derived from Oxs_ChunkEnergy class.  This
 * file also contains the implementation of the Oxs_GPU_ComputeEnergies()
 * friend function.
 */

#include <assert.h>
#include <string>

#include "GPU_chunkenergy.h"
#include "energy.h"
#include "mesh.h"

OC_USE_STRING;

/* End includes */

struct Oxs_GPU_ComputeEnergies_ChunkStruct {
public:
  Oxs_GPU_ChunkEnergy* energy;
  Oxs_ComputeEnergyDataThreaded ocedt;
  Oxs_ComputeEnergyDataThreadedAux ocedtaux;
  Oxs_GPU_ComputeEnergies_ChunkStruct()
    : energy(0) {}
};

class Oxs_GPU_ComputeEnergiesChunkThread : public Oxs_ThreadRunObj {
public:
  static Oxs_JobControl<ThreeVector> job_basket;
  /// job_basket is static, so only one "set" of this class is allowed.

  const Oxs_SimState* state;
  vector<Oxs_GPU_ComputeEnergies_ChunkStruct> energy_terms;

  Oxs_MeshValue<ThreeVector>* mxH;
  Oxs_MeshValue<ThreeVector>* mxH_accum;
  Oxs_MeshValue<ThreeVector>* mxHxm;
  const vector<OC_INDEX>* fixed_spins;
  OC_REAL8m max_mxH;

  OC_INDEX cache_blocksize;

  OC_BOOL accums_initialized;
  
   // GPU supported member
  OC_INDEX flag_mxHxm;

  Oxs_GPU_ComputeEnergiesChunkThread()
    : state(0),
      mxH(0),mxH_accum(0),
      mxHxm(0), fixed_spins(0),
      max_mxH(0.0),
      cache_blocksize(0), accums_initialized(0),
	  flag_mxHxm(0) {}

  void Cmd(int threadnumber, void* data) {};
  void GPU_Cmd(int threadnumber, void* data, DEVSTRUCT& dev_struct);

  static void Init(int thread_count,
                   const Oxs_StripedArray<ThreeVector>* arrblock) {
    job_basket.Init(thread_count,arrblock);
  }
  
  // Note: Default copy constructor and assignment operator,
  // and destructor.
};

Oxs_JobControl<ThreeVector> Oxs_GPU_ComputeEnergiesChunkThread::job_basket;

void
Oxs_GPU_ComputeEnergiesChunkThread::GPU_Cmd
(int threadnumber,
 void* /* data */, DEVSTRUCT& dev_struct)
{
#ifdef GPU_DEBUG
FILE *mylocation = fopen ("location.txt","a");
fprintf(mylocation,"entering GPU_Cmd in %s...\n", __FILE__);
fclose(mylocation);
#endif
  OC_REAL8m max_mxH_sq = 0.0;
  const Oxs_MeshValue<ThreeVector>& spin = state->spin;
  const Oxs_MeshValue<OC_REAL8m>& Ms = *(state->Ms);

  // In chunk post-processing segment, the torque at fixed spins
  // is forced to zero.  Variable i_fixed holds the latest working
  // position in the fixed_spins array between chunks.
  OC_INDEX i_fixed = 0;
  OC_INDEX i_fixed_total = 0;
  if(fixed_spins) i_fixed_total = fixed_spins->size();

  while(1) {
    // Claim a chunk
    OC_INDEX index_start,index_stop;
    job_basket.GetJob(threadnumber,index_start,index_stop);

    if(index_start>=index_stop) break;

	OC_INDEX icache_start=index_start;
	
      OC_INDEX icache_stop = icache_start + cache_blocksize;
      if(icache_stop>index_stop) icache_stop = index_stop;

      // Process chunk
      OC_UINT4m energy_item = 0;
      for(vector<Oxs_GPU_ComputeEnergies_ChunkStruct>::iterator eit
            = energy_terms.begin();
          eit != energy_terms.end() ; ++eit, ++energy_item) {

        // Set up some refs for convenience
        Oxs_GPU_ChunkEnergy& eterm = *(eit->energy);
        Oxs_ComputeEnergyDataThreaded& ocedt = eit->ocedt;
        Oxs_ComputeEnergyDataThreadedAux& ocedtaux = eit->ocedtaux;
#if REPORT_TIME
# if 0  // Individual chunk times currently meaningless,
        //and may slow code due to mutex blocks.
        ocedtaux.energytime.Start();
# endif 
#endif // REPORT_TIME
        if(!accums_initialized && energy_item==0) {
          // Note: Each thread has its own copy of the ocedt and
          // ocedtaux data, so we can tweak these as desired without
          // stepping on other threads

          // Move each accum pointer to corresponding non-accum member
          // for initialization.
          assert(ocedt.mxH == 0);
          Oxs_MeshValue<OC_REAL8m>* energy_accum_save = ocedt.energy_accum;
          if(ocedt.energy == 0) ocedt.energy = ocedt.energy_accum;
          ocedt.energy_accum = 0;

          Oxs_MeshValue<ThreeVector>* H_accum_save = ocedt.H_accum;
          if(ocedt.H == 0)      ocedt.H      = ocedt.H_accum;
          ocedt.H_accum      = 0;

          // Note: ocedt.mxH should always be zero, but check here anyway
          // for easier code maintenance.
          Oxs_MeshValue<ThreeVector>* mxH_accum_save = ocedt.mxH_accum;
          if(ocedt.mxH == 0)    ocedt.mxH    = ocedt.mxH_accum;
          ocedt.mxH_accum    = 0;

		  if(mxHxm)	flag_mxHxm = 1;
		  else	flag_mxHxm = 0;
		  eterm.GPU_ComputeEnergyChunk(*state,ocedt,ocedtaux,
                                   icache_start,icache_stop,
                                   threadnumber, dev_struct, flag_mxHxm);
		  
          // Copy data as necessary
          if(energy_accum_save) {
            if(ocedt.energy != energy_accum_save) {
              for(OC_INDEX i=icache_start;i<icache_stop;++i) {
                (*energy_accum_save)[i] = (*(ocedt.energy))[i];
              }
            } else {
              ocedt.energy = 0;
            }
          }
          ocedt.energy_accum = energy_accum_save;

          if(H_accum_save) {
            if(ocedt.H != H_accum_save) {
              for(OC_INDEX i=icache_start;i<icache_stop;++i) {
                (*H_accum_save)[i] = (*(ocedt.H))[i];
              }
            } else {
              ocedt.H = 0;
            }
          }
          ocedt.H_accum = H_accum_save;

          if(mxH_accum_save) {
            if(ocedt.mxH != mxH_accum_save) {
              // This branch should never run
              abort();
            } else {
              ocedt.mxH = 0;
            }
          }
          ocedt.mxH_accum = mxH_accum_save;

        } else {
          // Standard processing: accum elements already initialized.
          eterm.GPU_ComputeEnergyChunk(*state,ocedt,ocedtaux,
                                   icache_start,icache_stop,
                                   threadnumber, dev_struct, flag_mxHxm);
        }

#if REPORT_TIME
# if 0  // Individual chunk times currently meaningless,
        //and may slow code due to mutex blocks.
        ocedtaux.energytime.Stop();
# endif
#endif // REPORT_TIME
      }// process chunk

      // Post-processing, for this energy term and chunk.

      // Zero torque on fixed spins.  This code assumes that, 1) the
      // fixed_spins list is sorted in increasing order, and 2) the
      // chunk indices come in strictly monotonically increasing
      // order.
      // NB: Outside this loop, "i_fixed" stores the search start
      // location for the next chunk.
      while(i_fixed < i_fixed_total) {
        OC_INDEX index = (*fixed_spins)[i_fixed];
        if(index <  icache_start) { ++i_fixed; continue; }
        if(index >= icache_stop) break;
        if(mxH)       (*mxH)[index].Set(0.,0.,0.);
        if(mxH_accum) (*mxH_accum)[index].Set(0.,0.,0.);
        ++i_fixed;
      }

  }//while(1)

  max_mxH = sqrt(max_mxH_sq);
}

#if REPORT_TIME
Nb_StopWatch Oxs_GPU_ChunkEnergy::chunktime;

void Oxs_GPU_ChunkEnergy::ReportTime()
{
  Oc_TimeVal cpu,wall;
  chunktime.GetTimes(cpu,wall);
  if(double(wall)>0.0) {
    fprintf(stderr,"GetEnergy time (secs)%7.2f cpu /%7.2f wall,"
            " ChunkEnergies total (%u evals)\n",
            double(cpu),double(wall),GetEnergyEvalCount());
    chunktime.Reset();  // Only print once (per run).
  }
}
#endif // REPORT_TIME

void Oxs_GPU_ComputeEnergies
(const Oxs_SimState& state,
 Oxs_ComputeEnergyData& oced,
 const vector<Oxs_Energy*>& energies,
 Oxs_ComputeEnergyExtraData& oceed,
 DEVSTRUCT& dev_struct)
{ // Compute sums of energies, fields, and/or torques on GPU
  // for all energies in "energies" import.
  
  if(state.Id()==0) {
    String msg = String("Programming error:"
                        " Invalid (unlocked) state detected"
                        " in Oxs_ComputeEnergies");
    throw Oxs_ExtError(msg);
  }

  /* Ideally, since we maintain the computation on GPU,
     We don't need to maintain the following pointers... */
  if(oced.scratch_energy==NULL || oced.scratch_H==NULL) {
    // Bad input
    String msg = String("Oxs_ComputeEnergyData object in function"
                        " Oxs_ComputeEnergies"
                        " contains NULL scratch pointers.");
    throw Oxs_ExtError(msg);
  }

  if(oced.energy != NULL || oced.H != NULL || oced.mxH != NULL) {
    String msg = String("Programming error in function"
                        " Oxs_ComputeEnergies:"
                        " non-NULL energy, H, and/or mxH imports.");
    throw Oxs_ExtError(msg);
  }

  const int thread_count = Oc_GetMaxThreadCount();

  if(oced.energy_accum) {
    oced.energy_accum->AdjustSize(state.mesh);
  }
  if(oced.H_accum) {
    oced.H_accum->AdjustSize(state.mesh);
  }
  if(oced.mxH_accum) {
    oced.mxH_accum->AdjustSize(state.mesh);
  }
  if(oceed.mxHxm) {
    oceed.mxHxm->AdjustSize(state.mesh);
  }
  oced.energy_sum = 0.0;
  oced.pE_pt = 0.0;
  oceed.max_mxH = 0.0;


  if(energies.size() == 0) {
    // No energies.  Zero requested outputs and return.
    OC_INDEX size = state.mesh->Size();
    if(oced.energy_accum) {
      for(OC_INDEX i=0; i<size; ++i) {
        (*(oced.energy_accum))[i] = 0.0;
      }
    }
    if(oced.H_accum) {
      for(OC_INDEX i=0; i<size; ++i) {
        (*(oced.H_accum))[i] = ThreeVector(0.0,0.0,0.0);
      }
    }
    if(oced.mxH_accum) {
      for(OC_INDEX i=0; i<size; ++i) {
        (*(oced.mxH_accum))[i] = ThreeVector(0.0,0.0,0.0);
      }
    }
    if(oceed.mxHxm) {
      for(OC_INDEX i=0; i<size; ++i) {
        (*(oceed.mxHxm))[i] = ThreeVector(0.0,0.0,0.0);
      }
    }
    return;
  }

  if(oced.mxH_accum==0 && oceed.mxHxm!=0) {
    // Hack mxHxm into mxH_accum.  We can identify this situation
    // by checking mxH_accum == mxHxm, and undo at the end.  Also
    // The Oxs_ComputeEnergiesChunkThread objects know about this
    // and respond appropriately.
    oced.mxH_accum = oceed.mxHxm;
  }
  
  vector<Oxs_GPU_ComputeEnergies_ChunkStruct> chunk;
  vector<Oxs_GPU_Energy*> nonchunk;

  // Initialize those parts of ChunkStruct that are independent
  // of any particular energy term.
  Oxs_GPU_ComputeEnergies_ChunkStruct foo;
  foo.ocedt.state_id = state.Id();
  foo.ocedt.scratch_energy = oced.scratch_energy;
  foo.ocedt.scratch_H      = oced.scratch_H;
  foo.ocedt.energy_accum   = oced.energy_accum;
  foo.ocedt.H_accum        = oced.H_accum;
  foo.ocedt.mxH_accum      = oced.mxH_accum;
  for(vector<Oxs_Energy*>::const_iterator it = energies.begin();
      it != energies.end() ; ++it ) {
    Oxs_GPU_ChunkEnergy* ceptr =
      dynamic_cast<Oxs_GPU_ChunkEnergy*>(*it);
    if(ceptr != NULL) {
      // Set up and initialize chunk energy structures
      foo.energy = ceptr;
      if(ceptr->energy_density_output.GetCacheRequestCount()>0) {
        ceptr->energy_density_output.cache.state_id=0;
        foo.ocedt.energy = &(ceptr->energy_density_output.cache.value);
        foo.ocedt.energy->AdjustSize(state.mesh);
      }
      if(ceptr->field_output.GetCacheRequestCount()>0) {
        ceptr->field_output.cache.state_id=0;
        foo.ocedt.H = &(ceptr->field_output.cache.value);
        foo.ocedt.H->AdjustSize(state.mesh);
      }
      chunk.push_back(foo);
    } else {
      nonchunk.push_back(dynamic_cast<Oxs_GPU_Energy*>(*it));
    }
  }

  OC_BOOL accums_initialized = 0;

  // Non-chunk energies //////////////////////////////////////
  for(vector<Oxs_GPU_Energy*>::const_iterator ncit = nonchunk.begin();
      ncit != nonchunk.end() ; ++ncit ) {
    Oxs_GPU_Energy& eterm = *(*ncit);

#if REPORT_TIME
    eterm.energytime.Start();
#endif // REPORT_TIME
    
	Oxs_ComputeEnergyData term_oced(state);
    term_oced.scratch_energy = oced.scratch_energy;
    term_oced.scratch_H      = oced.scratch_H;
    term_oced.energy_accum = oced.energy_accum;
    term_oced.H_accum      = oced.H_accum;
    term_oced.mxH_accum    = oced.mxH_accum;

    if(eterm.energy_density_output.GetCacheRequestCount()>0) {
      eterm.energy_density_output.cache.state_id=0;
      term_oced.energy = &(eterm.energy_density_output.cache.value);
      term_oced.energy->AdjustSize(state.mesh);
    }

    if(eterm.field_output.GetCacheRequestCount()>0) {
      eterm.field_output.cache.state_id=0;
      term_oced.H = &(eterm.field_output.cache.value);
      term_oced.H->AdjustSize(state.mesh);
    }

    if(!accums_initialized) {

      // Initialize by filling
      term_oced.energy_accum = 0;
      term_oced.H_accum = 0;
      term_oced.mxH_accum = 0;
      if(term_oced.energy == 0) term_oced.energy = oced.energy_accum;
      if(term_oced.H == 0)      term_oced.H      = oced.H_accum;
      if(term_oced.mxH == 0)    term_oced.mxH    = oced.mxH_accum;
    }
	

    ++(eterm.calc_count);

	eterm.GPU_ComputeEnergy(state, term_oced, dev_struct, true);

	if(eterm.field_output.GetCacheRequestCount()>0) {
      eterm.field_output.cache.state_id=state.Id();
    }
    if(eterm.energy_density_output.GetCacheRequestCount()>0) {
      eterm.energy_density_output.cache.state_id=state.Id();
    }
    if(eterm.energy_sum_output.GetCacheRequestCount()>0) {

      eterm.energy_sum_output.cache.value=term_oced.energy_sum;
      eterm.energy_sum_output.cache.state_id=state.Id();
    }

    if(!accums_initialized) {

      // If output buffer spaced was used instead of accum space, then
      // copy from output buffer to accum space.  This hurts from a
      // memory bandwidth perspective, but is rather hard to avoid.
      // (Options: Do accum initialization in chunk-energy branch,
      // but that hurts with respect to mxHxm and max |mxH| computations.
      // Or one could have the ComputeEnergy class fill more than one
      // array with the non-accum output (say, via a parameter that
      // says to set to accum rather than add to accum), but that is
      // rather awkward.  Instead, we assume that if the user wants
      // high speed then he won't enable term energy or H outputs.)
      if(oced.energy_accum && term_oced.energy != oced.energy_accum) {
 
        *(oced.energy_accum) = *(term_oced.energy);
      }
      if(oced.H_accum      && term_oced.H      != oced.H_accum) {
        *(oced.H_accum) = *(term_oced.H);
      }
      if(oced.mxH_accum    && term_oced.mxH    != oced.mxH_accum) {
        *(oced.mxH_accum) = *(term_oced.mxH);
      }
      accums_initialized = 1;
    }
 
    oced.pE_pt += term_oced.pE_pt;
	
#if REPORT_TIME
    eterm.energytime.Stop();
#endif // REPORT_TIME
  }


  // Chunk energies ///////////////////////////////////////////

#if REPORT_TIME
  Oxs_GPU_ChunkEnergy::chunktime.Start();
#endif

  // Compute cache_blocksize
  const OC_INDEX meshsize = state.mesh->Size();
  const OC_INDEX cache_size = 1024 * 1024;  // Should come from
  /// platform file or perhaps sysconf().

  const OC_INDEX recsize = sizeof(ThreeVector) + sizeof(OC_REAL8m);
  /// May want to query individual energies for this.

#define FUDGE 8
  OC_INDEX tcblocksize = (cache_size>FUDGE*recsize ?
                        cache_size/(FUDGE*recsize) : 1);
  if(thread_count*tcblocksize>meshsize) {
    tcblocksize = meshsize/thread_count;
  }
  if(0 == tcblocksize) {
    tcblocksize = 1;    // Safety
  } else if(0 != tcblocksize%16) {
    tcblocksize += 16 - (tcblocksize%16);  // Make multiple of 16
  }
  const OC_INDEX cache_blocksize = tcblocksize;
  
  // Thread control
  static Oxs_ThreadTree threadtree;

  Oxs_GPU_ComputeEnergiesChunkThread::Init(thread_count,
                                       state.spin.GetArrayBlock());

  vector<Oxs_GPU_ComputeEnergiesChunkThread> chunk_thread;
  chunk_thread.resize(thread_count);
  chunk_thread[0].state     = &state;
  chunk_thread[0].energy_terms = chunk; // Make copies.
  chunk_thread[0].mxH       = oced.mxH;
  chunk_thread[0].mxH_accum = oced.mxH_accum;
  chunk_thread[0].mxHxm     = oceed.mxHxm;
  chunk_thread[0].fixed_spins = oceed.fixed_spin_list;
  chunk_thread[0].cache_blocksize = cache_blocksize;
  chunk_thread[0].accums_initialized = accums_initialized;

  // Initialize chunk energy computations
  for(vector<Oxs_GPU_ComputeEnergies_ChunkStruct>::iterator it
        = chunk.begin(); it != chunk.end() ; ++it ) {
    Oxs_GPU_ChunkEnergy& eterm = *(it->energy);  // For code clarity
    Oxs_ComputeEnergyDataThreaded& ocedt = it->ocedt;
    Oxs_ComputeEnergyDataThreadedAux& ocedtaux = it->ocedtaux;
    eterm.ComputeEnergyChunkInitialize(state,ocedt,ocedtaux,
                                       thread_count);			
  }

  chunk_thread[0].GPU_Cmd(0, NULL, dev_struct);
  
  
  // Note: If chunk.size()>0, then we are guaranteed that accums are
  // initialized.  If accums_initialized is ever needed someplace
  // downstream, then uncomment the following line:
  // if(chunk.size()>0) accums_initialized = 1;
  // Finalize chunk energy computations
  for(OC_INDEX ei=0;static_cast<size_t>(ei)<chunk.size();++ei) {

    Oxs_GPU_ChunkEnergy& eterm = *(chunk[ei].energy);  // Convenience
    const Oxs_ComputeEnergyDataThreaded& ocedt = chunk[ei].ocedt;
    const Oxs_ComputeEnergyDataThreadedAux& ocedtaux = chunk[ei].ocedtaux;

    eterm.ComputeEnergyChunkFinalize(state,ocedt,ocedtaux,
                                     thread_count);

    ++(eterm.calc_count);

    // For each energy term, loop though all threads and sum
    // energy and pE_pt contributions.
    OC_REAL8m pE_pt_term = chunk[ei].ocedtaux.pE_pt_accum;
    for(int ithread=0;ithread<thread_count;++ithread) {
      pE_pt_term
        += chunk_thread[ithread].energy_terms[ei].ocedtaux.pE_pt_accum;
    }
    oced.pE_pt += pE_pt_term;

    OC_REAL8m energy_term = chunk[ei].ocedtaux.energy_total_accum;
    for(int ithread=0;ithread<thread_count;++ithread) {
      energy_term
        += chunk_thread[ithread].energy_terms[ei].ocedtaux.energy_total_accum;
    }
    oced.energy_sum += energy_term;

    if(eterm.energy_sum_output.GetCacheRequestCount()>0) {
      eterm.energy_sum_output.cache.value=energy_term;
      eterm.energy_sum_output.cache.state_id=state.Id();
    }

    if(eterm.field_output.GetCacheRequestCount()>0) {
      eterm.field_output.cache.state_id=state.Id();
    }

    if(eterm.energy_density_output.GetCacheRequestCount()>0) {
      eterm.energy_density_output.cache.state_id=state.Id();
    }

#if REPORT_TIME
    Nb_StopWatch bar;
    bar.ThreadAccum(chunk[ei].ocedtaux.energytime);
    for(int ithread=0;ithread<thread_count;++ithread) {
      bar.ThreadAccum
        (chunk_thread[ithread].energy_terms[ei].ocedtaux.energytime);

    }
    eterm.energytime.Accum(bar);
#endif // REPORT_TIME
  }

  if(oceed.mxHxm!=0 && oced.mxH_accum == oceed.mxHxm) {
    // Undo mxHxm hack
    oced.mxH_accum = 0;
  }

  oceed.max_mxH = 0.0;
  for(vector<Oxs_GPU_ComputeEnergiesChunkThread>::const_iterator cect
        = chunk_thread.begin(); cect != chunk_thread.end() ; ++cect ) {
    if(cect->max_mxH > oceed.max_mxH) oceed.max_mxH = cect->max_mxH;
  }

#if REPORT_TIME
  Oxs_GPU_ChunkEnergy::chunktime.Stop();
#endif
}
