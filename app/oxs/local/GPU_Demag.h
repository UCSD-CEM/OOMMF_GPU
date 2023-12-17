/* FILE: GPU_Demag.h            -*-Mode: c++-*-
 *
 * Average H demag field across rectangular cells.  This is a modified
 * version of the simpledemag class, which uses symmetries in the
 * interaction coefficients to reduce memory usage.
 *
 * This code uses the Oxs_FFT3v classes to perform direct FFTs of the
 * import magnetization ThreeVectors.  This GPU_Demag class is a
 * drop-in replacement for an older GPU_Demag class that used the
 * scalar Oxs_FFT class.  That older class has been renamed
 * Oxs_DemagOld, and is contained in the demagold.* files.
 *
 * There are two .cc files providing definitions for routines in the
 * header file: demag.cc and demag-threaded.cc.  The first is provides
 * a non-threaded implementation of the routines, the second a
 * threaded version.  Exactly one of the .cc files is compiled into
 * the oxs executable, depending on the setting of the compiler macro
 * OOMMF_THREADS.
 *
 * This version of code does not support threaded CPU computation yet.
 */

#ifndef _GPU_DEMAG
#define _GPU_DEMAG


#define GPU_CPU_TRANS

#include <iostream>
#include "oc.h"  // Includes OOMMF_THREADS macro in ocport.h
#include "GPU_energy.h"
#include "fft3v.h"
#include "key.h"
#include "mesh.h"
#include "meshvalue.h"
#include "simstate.h"
#include "threevector.h"
#include "rectangularmesh.h"

#include "cufft.h"
#include "GPU_devstruct.h"

/* End includes */

class GPU_Demag
  : public Oxs_GPU_Energy, public Oxs_EnergyPreconditionerSupport {
	  //PRECEONDITIONER NOT SUPPORTED
  friend class Oxs_FFTLocker;
  friend class _Oxs_DemagFFTxThread;
  friend class _Oxs_DemagiFFTxDotThread;
  friend class _Oxs_DemagFFTyThread;
  friend class _Oxs_DemagFFTyConvolveThread;
  friend class _Oxs_DemagFFTzConvolveThread;
  friend class _Oxs_DemagFFTyzConvolveThread;

private:
#if REPORT_TIME
  mutable Nb_StopWatch inittime;

  mutable Nb_StopWatch fftforwardtime;
  mutable Nb_StopWatch fftxforwardtime;
  mutable Nb_StopWatch fftyforwardtime;

  mutable Nb_StopWatch fftinversetime;
  mutable Nb_StopWatch fftxinversetime;
  mutable Nb_StopWatch fftyinversetime;

  mutable Nb_StopWatch convtime;
  mutable Nb_StopWatch dottime;
  
  mutable Nb_StopWatch prectime;
  mutable Nb_StopWatch preptime;
  mutable Nb_StopWatch memcpytime;
  mutable Nb_StopWatch adsizetime;
  mutable Nb_StopWatch memsettime;
  mutable Nb_StopWatch tmptime;

  enum { dvltimer_number = 10 };
  mutable Nb_StopWatch dvltimer[dvltimer_number];
#endif // REPORT_TIME

  mutable OC_INDEX rdimx; // Natural size of real data
  mutable OC_INDEX rdimy; // Digital Mars compiler wants these as separate
  mutable OC_INDEX rdimz; //    statements, because of "mutable" keyword.
  mutable OC_INDEX rsize;
  mutable OC_INDEX cdimx; // Full size of complex data
  mutable OC_INDEX cdimy;
  mutable OC_INDEX cdimz;
  mutable OC_INDEX csize;
  // 2*(cdimx-1)>=rdimx, cdimy>=rdimy, cdimz>=rdimz
  // cdimx-1 and cdim[yz] should be powers of 2.
  mutable OC_INDEX adimx; // Dimensions of A## storage (see below).
  mutable OC_INDEX adimy;
  mutable OC_INDEX adimz;
  mutable OC_INDEX asize;
  mutable OC_INDEX fdimx; //full fft size
  mutable OC_INDEX fdimxy; //full fft size
  mutable OC_INDEX fdimy;
  mutable OC_INDEX fdimz;
  mutable OC_INDEX fsize;
  
  //*******THIS SHOULD BE ALWAYS PUT INTO 0 SINCE PERIODIC IS NOT SUPPORTED**
  mutable int xperiodic;  // If 0, then not periodic.  Otherwise,
  mutable int yperiodic;  // periodic in indicated direction.
  mutable int zperiodic;

  mutable OC_UINT4m mesh_id;

  // The A## arrays hold demag coefficients, transformed into
  // frequency domain.  These are held long term.  Due to
  // symmetries, only the first octant needs to be saved.
  //   The Hxfrm array is temporary space used primarily to hold
  // the transform image of the computed field.  It is also
  // used as intermediary storage for the A## values, which
  // for convenience are computed at full size before being
  // transfered to the 1/8-sized A## storage arrays.
  //   All of these arrays are actually arrays of complex-valued
  // three vectors, but are handled as simple REAL arrays.
  //mutable A_coefs* A;
  mutable FD_TYPE* G;

//#if !OOMMF_THREADS
  mutable OXS_FFT_REAL_TYPE *Hxfrm;
//#else
//  // In the threaded code, the memory pointed to by Hxfrm
//  // is managed by an Oxs_StripedArray object
//  //mutable Oxs_StripedArray<OXS_FFT_REAL_TYPE> Hxfrm_base;
//  mutable Oxs_StripedArray<OXS_FFT_REAL_TYPE> Hxfrm_base_yz;
//  /// Hxfrm_base_yz is used with the FFTyz+embedded convolution code
//#endif

  OC_REAL8m asymptotic_radius;
  /// If >=0, then radius beyond which demag coefficients A_coefs
  /// are computed using asymptotic (dipolar and higher) approximation
  /// instead of Newell's analytic formulae.

  //mutable OXS_FFT_REAL_TYPE *Mtemp;  // Temporary space to hold
  /// Ms[]*m[].  The plan is to make this space unnecessary
  /// by introducing FFT functions that can take Ms as input
  /// and do the multiplication on the fly.

  // Object to perform FFTs.  All transforms are the same size, so we
  // only need one Oxs_FFT3DThreeVector object.  (Note: A
  // multi-threaded version of this code might want to have a separate
  // Oxs_FFT3DThreeVector for each thread, so that the roots-of-unity
  // arrays in each thread are independent.  This may improve memory
  // access on multi-processor machines where each processor has its
  // own memory.)
   //   The embed_convolution boolean specifies whether or not to embed
  // the convolution (matrix-vector multiply in FFT transform space)
  // computation inside the fftz computation.  If true then
  // embed_block_size is the number of z-axis FFTs to perform about
  // each "convolution" (really matrix-vector multiply) computation.
  // These are set inside the FillCoefficientArrays member function,
  // and used inside GetEnergy.

  mutable Oxs_FFT1DThreeVector fftx;
  mutable Oxs_FFTStrided ffty;
  mutable Oxs_FFTStrided fftz;	//EMBED_CONVOLUTION SETTING SHOULD BE SYNCRONIZED WITH OTHER ORIGINAL FILES
  mutable OC_BOOL embed_convolution; // Note: Always true in threaded version
  
  /*************EMBED_CONVOLUTION SETTING SHOULD BE SYNCRONIZED WITH OTHER ORIGINAL FILES */
  mutable OC_INDEX embed_block_size;
  mutable OC_INDEX embed_yzblock_size;//EMBED_CONVOLUTION SETTING SHOULD BE SYNCRONIZED WITH OTHER ORIGINAL FILES

  //*************MAY NOT NEED cache_size IN THE GPU CODE************
  OC_INDEX cache_size; // Cache size in bytes.  Used to select
                       // embed_block_size.

  //*************NOT CLEAR WHY WE NEET THIS*************************
  OC_INT4m zero_self_demag;
  /// If zero_self_demag is true, then diag(1/3,1/3,1/3) is subtracted
  /// from the self-demag term.  In particular, for cubic cells this
  /// makes the self-demag field zero.  This will change the value
  /// computed for the demag energy by a constant amount, but since the
  /// demag field is changed by a multiple of m, the torque and
  /// therefore the magnetization dynamics are unaffected.

  void FillCoefficientArrays(const Oxs_Mesh* mesh, 
							DEVSTRUCT& dev_struct) const;

  void ReleaseMemory() const;
  void ReInitializeDevMemory(DEVSTRUCT& dev_struct) const;
  
  //GPU supported arrays and functions
  void AllocDevMemory(DEVSTRUCT& dev_struct) const ;

 mutable cufftHandle plan_fwd;
 mutable cufftHandle plan_bwd;
 mutable dim3 Knl_Blk_rsize;
 mutable dim3 Knl_Grid_rsize;
 mutable dim3 Knl_Blk_csize;
 mutable dim3 Knl_Grid_csize;
 mutable FD_TYPE* tmp_spin;
 mutable FD_TYPE* tmp_Ms;
 mutable FD_TYPE* tmp_field;
 mutable FD_TYPE* tmp_energy;
  
 mutable OC_BOOL cufftPlanCreated;
 mutable OC_BOOL cufftPlanWorkAreaSet;
 mutable int maxGridSize;
 mutable FD_TYPE maxTotalThreads;
 mutable FD_TYPE* dev_MValue;
 mutable FD_TYPE* dev_Ms;
 mutable FD_TYPE* dev_Energy;
 mutable FD_TYPE* dev_GreenFunc_k;
 mutable FD_TYPE* dev_Field;
 mutable FD_CPLX_TYPE* dev_Mtemp;
 mutable FD_TYPE* dev_Torque;
 mutable FD_TYPE* dev_field_loc;
 mutable FD_TYPE* dev_energy_loc;
 mutable FD_TYPE* dev_tmp;
 mutable FD_TYPE* dev_volume;
   int _dev_num;
#ifdef GPU_DEBUG
	mutable FILE* locate;
#endif
#ifdef GPU_TIME
	mutable FILE* gputime;
	mutable cudaEvent_t start, stop;
	mutable float elapsedTime;
#endif
protected:
  virtual void GetEnergy(const Oxs_SimState& state,
			 Oxs_EnergyData& oed) const{};
  virtual void GPU_GetEnergy(const Oxs_SimState& state,
			 Oxs_EnergyData& oed, DEVSTRUCT& dev_struct,
			 unsigned int flag_outputH,
			 unsigned int flag_outputE, unsigned int flag_outputSumE,
       const OC_BOOL &flag_accum) const;

public:
  virtual const char* ClassName() const; // ClassName() is
  /// automatically generated by the OXS_EXT_REGISTER macro.
  GPU_Demag(const char* name,     // Child instance id
	    Oxs_Director* newdtr, // App director
	    const char* argstr);  // MIF input block parameters
  virtual ~GPU_Demag();
  virtual OC_BOOL Init();
  void ReleaseDevMemory() const;
  
  //**********THIS SUBROUTINE IS NOT SUPPORTED IN THIS VERSION***
  virtual OC_INT4m IncrementPreconditioner(PreconditionerData& pcd) {
    throw Oxs_ExtError(this, "preconditioner is not supported by GPU libraries yet");
  }
};


#endif // _GPU_Demag
