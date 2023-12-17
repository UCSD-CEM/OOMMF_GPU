/* FILE: demag.cc            -*-Mode: c++-*-
 *
 * Average H demag field across rectangular cells.  This is a modified
 * version of the simpledemag class, which uses symmetries in the
 * interaction coefficients to reduce memory usage.
 *
 * The formulae used are reduced forms of equations in A. J. Newell,
 * W. Williams, and D. J. Dunlop, "A Generalization of the Demagnetizing
 * Tensor for Nonuniform Magnetization," Journal of Geophysical Research
 * - Solid Earth 98, 9551-9555 (1993).
 *
 * This code uses the Oxs_FFT3v classes to perform direct FFTs of the
 * import magnetization ThreeVectors.  This GPU_Demag class is a
 * drop-in replacement for an older GPU_Demag class that used the
 * scalar Oxs_FFT class.  That older class has been renamed
 * Oxs_DemagOld, and is contained in the demagold.* files.
 *
 * NOTE: This is the non-threaded implementation of the routines
 *       declared in demag.h.  This version is included iff the
 *       compiler macro OOMMF_THREADS is 0.  For the threaded
 *       version of this code, see demag-threaded.cc.
 */

#include "GPU_Demag.h"  // Includes definition of OOMMF_THREADS macro
#include "GPU_Demag_kernel.h"
#include "demagcoef.h" // Used by both single-threaded code, and
/// also common single/multi-threaded code at bottom of this file.

#include <assert.h>
#include <string>
#include <sstream>

#include "ext.h"
#include "nb.h"
#include "director.h"
#include "key.h"
#include "mesh.h"
#include "meshvalue.h"
#include "simstate.h"
#include "threevector.h"
#include "energy.h"             // Needed to make MSVC++ 5 happy

#include "rectangularmesh.h"

#include "GPU_helper.h"

OC_USE_STRING;

/* End includes */

// Oxs_Ext registration support
OXS_EXT_REGISTER(GPU_Demag);

#ifndef VERBOSE_DEBUG
# define VERBOSE_DEBUG 0
#endif

// Size of threevector.  This macro is defined for code legibility
// and maintenance; it should always be "3".
#define ODTV_VECSIZE 3

// Size of complex value, in real units.  This macro is defined for code
// legibility and maintenance; it should always be "2".
#define ODTV_COMPLEXSIZE 2

// Constructor
GPU_Demag::GPU_Demag(
  const char* name,     // Child instance id
  Oxs_Director* newdtr, // App director
  const char* argstr)   // MIF input block parameters
  : Oxs_GPU_Energy(name,newdtr,argstr),
    rdimx(0),rdimy(0),rdimz(0), rsize(0),
	cdimx(0),cdimy(0),cdimz(0), csize(0),
    adimx(0),adimy(0),adimz(0), asize(0),
	fdimx(0),fdimy(0),fdimz(0), fsize(0),
    xperiodic(0),yperiodic(0),zperiodic(0),
    mesh_id(0), tmp_spin(0), tmp_field(0), tmp_energy(0),
	tmp_Ms(0), G(0),asymptotic_radius(-1),//Mtemp(0),,Hxfrm(0)
    embed_convolution(0),embed_block_size(0),
	dev_MValue(0), dev_Ms(0), dev_Field(0),
	dev_Energy(0), dev_GreenFunc_k(0),
	dev_Mtemp(0), dev_Torque(0), dev_field_loc(0),
	dev_energy_loc(0), cufftPlanCreated(false),
  cufftPlanWorkAreaSet(false) {
  _dev_num = DEV_NUM;
  asymptotic_radius = GetRealInitValue("asymptotic_radius",32.0);
  /// Units of (dx*dy*dz)^(1/3) (geometric mean of cell dimensions).
  /// Value of -1 disables use of asymptotic approximation on
  /// non-periodic grids.  For periodic grids zero or negative values
  /// for asymptotic_radius are reset to half the width of the
  /// simulation window in the periodic dimenions.  This may be
  /// counterintuitive, so it might be better to disallow or modify
  /// the behavior in the periodic setting.

  cache_size = 1024*GetIntInitValue("cache_size_KB",1024);
  /// Cache size in KB.  Default is 1 MB.  Code wants bytes, so multiply
  /// user input by 1024.  cache_size is used to set embed_block_size in
  /// FillCoefficientArrays member function.

  zero_self_demag = GetIntInitValue("zero_self_demag",0);
  /// If true, then diag(1/3,1/3,1/3) is subtracted from the self-demag
  /// term.  In particular, for cubic cells this makes the self-demag
  /// field zero.  This will change the value computed for the demag
  /// energy by a constant amount, but since the demag field is changed
  /// by a multiple of m, the torque and therefore the magnetization
  /// dynamics are unaffected.

  VerifyAllInitArgsUsed();
  
  fetchInfo_device(maxGridSize, maxTotalThreads, DEV_NUM);
}

GPU_Demag::~GPU_Demag() {
#if REPORT_TIME
  Oc_TimeVal cpu,wall;
  FILE* gputimeWall;

  double cpuDemag = 0.0;
  double wallDemag = 0.0;
  double totalDemag = 0.0;
  gputimeWall = fopen ("gputime_wall.txt","a");

  inittime.GetTimes(cpu,wall);
  if(double(wall)>0.0) {
    fprintf(gputimeWall,"      subtime ...   init%7.2f cpu /%7.2f wall,"
            " (%.1000s)\n",
            double(cpu),double(wall),InstanceName());
	totalDemag +=  double(wall);
  }
  
  memcpytime.GetTimes(cpu,wall);
  if(double(wall)>0.0) {
    fprintf(gputimeWall,"      subtime ...   memcpy%7.2f cpu /%7.2f wall,"
            " (%.1000s)\n",
            double(cpu),double(wall),InstanceName());
	cpuDemag +=  double(cpu);
	wallDemag +=  double(wall);
  }
  
  prectime.GetTimes(cpu,wall);
  if(double(wall)>0.0) {
    fprintf(gputimeWall,"      subtime ...   prec%7.2f cpu /%7.2f wall,"
            " (%.1000s)\n",
            double(cpu),double(wall),InstanceName());
	cpuDemag +=  double(cpu);
	wallDemag +=  double(wall);
  }
  
  
  memsettime.GetTimes(cpu,wall);
  if(double(wall)>0.0) {
    fprintf(gputimeWall,"      subtime ...   memset%7.2f cpu /%7.2f wall,"
            " (%.1000s)\n",
            double(cpu),double(wall),InstanceName());
	cpuDemag +=  double(cpu);
	wallDemag +=  double(wall);
  }
  
  tmptime.GetTimes(cpu,wall);
  if(double(wall)>0.0) {
    fprintf(gputimeWall,"      subtime ...   tmp%7.2f cpu /%7.2f wall,"
            " (%.1000s)\n",
            double(cpu),double(wall),InstanceName());
	cpuDemag +=  double(cpu);
	wallDemag +=  double(wall);
  }
  
  
  preptime.GetTimes(cpu,wall);
  if(double(wall)>0.0) {
    fprintf(gputimeWall,"      subtime ...   prep%7.2f cpu /%7.2f wall,"
            " (%.1000s)\n",
            double(cpu),double(wall),InstanceName());
	 cpuDemag +=  double(cpu);
	 wallDemag +=  double(wall);
  }
 
  
  adsizetime.GetTimes(cpu,wall);
  if(double(wall)>0.0) {
    fprintf(gputimeWall,"      subtime ...   adsize%7.2f cpu /%7.2f wall,"
            " (%.1000s)\n",
            double(cpu),double(wall),InstanceName());
	 cpuDemag +=  double(cpu);
	 wallDemag +=  double(wall);
  }
 

  fftforwardtime.GetTimes(cpu,wall);
  if(double(wall)>0.0) {
    fprintf(gputimeWall,"      subtime ...  f-fft%7.2f cpu /%7.2f wall,"
            " (%.1000s)\n",
            double(cpu),double(wall),InstanceName());
	cpuDemag +=  double(cpu);
	wallDemag +=  double(wall);
  }
  
  fftinversetime.GetTimes(cpu,wall);
  if(double(wall)>0.0) {
    fprintf(gputimeWall,"      subtime ...  i-fft%7.2f cpu /%7.2f wall,"
            " (%.1000s)\n",
            double(cpu),double(wall),InstanceName());
	cpuDemag +=  double(cpu);
	wallDemag +=  double(wall);
  }


  convtime.GetTimes(cpu,wall);
  if(double(wall)>0.0) {
    fprintf(gputimeWall,"      subtime ...   conv%7.2f cpu /%7.2f wall,"
            " (%.1000s)\n",
            double(cpu),double(wall),InstanceName());
	cpuDemag +=  double(cpu);
	wallDemag +=  double(wall);
  }

  dottime.GetTimes(cpu,wall);
  if(double(wall)>0.0) {
    fprintf(gputimeWall,"      subtime ...    dot%7.2f cpu /%7.2f wall,"
            " (%.1000s)\n",
            double(cpu),double(wall),InstanceName());
	cpuDemag +=  double(cpu);
	wallDemag +=  double(wall);
  }

	totalDemag += wallDemag;
    fprintf(gputimeWall,"      Demag ...    %7.3f cpu/ %7.3f wall/ total %7.3f, (%.1000s)\n",
		cpuDemag, wallDemag, totalDemag, InstanceName());
	fclose(gputimeWall);
#endif // REPORT_TIME
  ReleaseMemory();
}

OC_BOOL GPU_Demag::Init()
{
#if REPORT_TIME
  Oc_TimeVal cpu,wall;

  inittime.GetTimes(cpu,wall);
  if(double(wall)>0.0) {
    fprintf(stderr,"      subtime ...   init%7.2f cpu /%7.2f wall,"
            " (%.1000s)\n",
            double(cpu),double(wall),InstanceName());
  }
  
  preptime.GetTimes(cpu,wall);
  if(double(wall)>0.0) {
    fprintf(stderr,"      subtime ...   prep%7.2f cpu /%7.2f wall,"
            " (%.1000s)\n",
            double(cpu),double(wall),InstanceName());
  }
  
  memcpytime.GetTimes(cpu,wall);
  if(double(wall)>0.0) {
    fprintf(stderr,"      subtime ...   memcpy%7.2f cpu /%7.2f wall,"
            " (%.1000s)\n",
            double(cpu),double(wall),InstanceName());
  }
  
  prectime.GetTimes(cpu,wall);
  if(double(wall)>0.0) {
    fprintf(stderr,"      subtime ...   prec%7.2f cpu /%7.2f wall,"
            " (%.1000s)\n",
            double(cpu),double(wall),InstanceName());
  }
  
  memsettime.GetTimes(cpu,wall);
  if(double(wall)>0.0) {
    fprintf(stderr,"      subtime ...   memset%7.2f cpu /%7.2f wall,"
            " (%.1000s)\n",
            double(cpu),double(wall),InstanceName());
  }
  
  tmptime.GetTimes(cpu,wall);
  if(double(wall)>0.0) {
    fprintf(stderr,"      subtime ...   tmp%7.2f cpu /%7.2f wall,"
            " (%.1000s)\n",
            double(cpu),double(wall),InstanceName());
  }
  
  adsizetime.GetTimes(cpu,wall);
  if(double(wall)>0.0) {
    fprintf(stderr,"      subtime ...   adsize%7.2f cpu /%7.2f wall,"
            " (%.1000s)\n",
            double(cpu),double(wall),InstanceName());
  }

  fftforwardtime.GetTimes(cpu,wall);
  if(double(wall)>0.0) {
    fprintf(stderr,"      subtime ...  f-fft%7.2f cpu /%7.2f wall,"
            " (%.1000s)\n",
            double(cpu),double(wall),InstanceName());
  }
  fftinversetime.GetTimes(cpu,wall);
  if(double(wall)>0.0) {
    fprintf(stderr,"      subtime ...  i-fft%7.2f cpu /%7.2f wall,"
            " (%.1000s)\n",
            double(cpu),double(wall),InstanceName());
  }

  fftxforwardtime.GetTimes(cpu,wall);
  if(double(wall)>0.0) {
    fprintf(stderr,"      subtime ... f-fftx%7.2f cpu /%7.2f wall,"
            " (%.1000s)\n",
            double(cpu),double(wall),InstanceName());
  }
  fftxinversetime.GetTimes(cpu,wall);
  if(double(wall)>0.0) {
    fprintf(stderr,"      subtime ... i-fftx%7.2f cpu /%7.2f wall,"
            " (%.1000s)\n",
            double(cpu),double(wall),InstanceName());
  }

  fftyforwardtime.GetTimes(cpu,wall);
  if(double(wall)>0.0) {
    fprintf(stderr,"      subtime ... f-ffty%7.2f cpu /%7.2f wall,"
            " (%.1000s)\n",
            double(cpu),double(wall),InstanceName());
  }
  fftyinversetime.GetTimes(cpu,wall);
  if(double(wall)>0.0) {
    fprintf(stderr,"      subtime ... i-ffty%7.2f cpu /%7.2f wall,"
            " (%.1000s)\n",
            double(cpu),double(wall),InstanceName());
  }

  convtime.GetTimes(cpu,wall);
  if(double(wall)>0.0) {
    fprintf(stderr,"      subtime ...   conv%7.2f cpu /%7.2f wall,"
            " (%.1000s)\n",
            double(cpu),double(wall),InstanceName());
  }

  dottime.GetTimes(cpu,wall);
  if(double(wall)>0.0) {
    fprintf(stderr,"      subtime ...    dot%7.2f cpu /%7.2f wall,"
            " (%.1000s)\n",
            double(cpu),double(wall),InstanceName());
  }

  for(int i=0;i<dvltimer_number;++i) {
    dvltimer[i].GetTimes(cpu,wall);
    if(double(wall)>0.0) {
      fprintf(stderr,"      subtime ... dvl[%d]%7.2f cpu /%7.2f wall,"
              " (%.1000s)\n",
              i,double(cpu),double(wall),InstanceName());
    }
    dvltimer[i].Reset();
  }

  inittime.Reset();
  fftforwardtime.Reset();
  fftinversetime.Reset();
  memcpytime.Reset();
  prectime.Reset();
  preptime.Reset();
  memsettime.Reset();
  tmptime.Reset();
  adsizetime.Reset();
  fftxforwardtime.Reset();
  fftxinversetime.Reset();
  fftyforwardtime.Reset();
  fftyinversetime.Reset();
  convtime.Reset();
  dottime.Reset();
#endif // REPORT_TIME
  mesh_id = 0;
  ReleaseMemory();
  return Oxs_GPU_Energy::Init();
}

void GPU_Demag::ReInitializeDevMemory(DEVSTRUCT& dev_struct) const {
    if(dev_struct.dev_MValue)	dev_MValue = dev_struct.dev_MValue;
	else{
			String msg=String("dev_struct.dev_MValue not initiated in : \"")
			  + String(ClassName()) + String("\".");
			throw Oxs_ExtError(this,msg.c_str());
	}
	
	if(dev_struct.dev_Ms) dev_Ms = dev_struct.dev_Ms;
	else{
		String msg=String("dev_struct.dev_Ms not initiated in : \"")
		  + String(ClassName()) + String("\".");
		throw Oxs_ExtError(this,msg.c_str());
	}
	
	if(dev_struct.dev_field) dev_Field = dev_struct.dev_field;
	else{
		String msg=String("dev_struct.dev_field not initiated in : \"")
		  + String(ClassName()) + String("\".");
		throw Oxs_ExtError(this,msg.c_str());
	}
	
	if(dev_struct.dev_energy) dev_Energy = dev_struct.dev_energy;
	else{
		String msg=String("dev_struct.dev_energy not initiated in : \"")
		  + String(ClassName()) + String("\".");
		throw Oxs_ExtError(this,msg.c_str());
	}
	
	if(dev_struct.dev_torque) dev_Torque = dev_struct.dev_torque;
	else{
		String msg=String("dev_struct.dev_torque not initiated in : \"")
		  + String(ClassName()) + String("\".");
		throw Oxs_ExtError(this,msg.c_str());
	}
  
  if(dev_struct.dev_local_sum) {
    dev_tmp = dev_struct.dev_local_sum;
  } else{
		String msg=String("dev_struct.dev_local_sum not initiated in : \"")
		  + String(ClassName()) + String("\".");
		throw Oxs_ExtError(this,msg.c_str());
	}
  
  if(dev_struct.dev_vol) {
    dev_volume = dev_struct.dev_vol;
  } else {
		String msg=String("dev_struct.dev_local_sum not initiated in : \"")
		  + String(ClassName()) + String("\".");
		throw Oxs_ExtError(this,msg.c_str());
	}
  
  if(dev_struct.dev_local_field) {
    dev_field_loc = dev_struct.dev_local_field;
  } else {
		String msg=String("dev_struct.dev_local_field not initiated in : \"")
		  + String(ClassName()) + String("\".");
		throw Oxs_ExtError(this,msg.c_str());
	}
  
  if(dev_struct.dev_local_energy) {
    dev_energy_loc = dev_struct.dev_local_energy;
  } else {
		String msg=String("dev_struct.dev_local_energy not initiated in : \"")
		  + String(ClassName()) + String("\".");
		throw Oxs_ExtError(this,msg.c_str());
	}
    
  if (cufftSetWorkArea(plan_fwd, dev_struct.dev_FFT_workArea) != CUFFT_SUCCESS) {
    string msg("error when cufftSetWorkArea plan_fwd on GPU\n");
    throw Oxs_ExtError(this, msg.c_str());  
  }
  
  if (cufftSetWorkArea(plan_bwd, dev_struct.dev_FFT_workArea) != CUFFT_SUCCESS) {
    string msg("error when cufftSetWorkArea plan_bwd on GPU\n");
    throw Oxs_ExtError(this, msg.c_str());  
  }
  
  cufftPlanWorkAreaSet = true;
}

void GPU_Demag::AllocDevMemory(DEVSTRUCT& dev_struct) const {

  alloc_device(dev_GreenFunc_k, asize * 6, _dev_num, "dev_GreenFunc_k");
  alloc_device(dev_Mtemp, csize * ODTV_VECSIZE, _dev_num, "dev_Mtemp");
}

void GPU_Demag::ReleaseMemory() const
{ // Conceptually const
  if(G!=0) 			 { delete[] G; G=0; }
  
  rdimx=rdimy=rdimz=0;
  cdimx=cdimy=cdimz=0;
  adimx=adimy=adimz=0;
  fdimx=fdimy=fdimz=0;
  
  ReleaseDevMemory();
}

template <typename T>
std::string my_to_string(T value)
{
	std::ostringstream os ;
	os << value ;
	return os.str() ;
}

void GPU_Demag::ReleaseDevMemory() const {

  release_device(dev_GreenFunc_k, _dev_num, "dev_GreenFunc_k");
  release_device(dev_Mtemp, _dev_num, "dev_Mtemp");
  
  if (cufftPlanCreated) {
    cufftResult_t cufftResult = cufftDestroy(plan_fwd);
    if (cufftResult != CUFFT_SUCCESS) {
      String msg = String("cufft error after cufftDestroy(plan_fwd) in : \"")
        + String(ClassName()) + String(" errorCode: ")
        + my_to_string(cufftResult) + String("\".");
      throw Oxs_ExtError(this,msg.c_str());
    };
    
    cufftResult = cufftDestroy(plan_bwd);
    if (cufftResult != CUFFT_SUCCESS) {
      String msg = String("cufft error after FcufftDestroy(plan_bwd) in : \"")
        + String(ClassName()) + String(" errorCode: ")
        + my_to_string(cufftResult) + String("\".");
      throw Oxs_ExtError(this,msg.c_str());
    };
    cufftPlanCreated = false;
    cufftPlanWorkAreaSet = false;
  }
}

void GPU_Demag::FillCoefficientArrays(
 const Oxs_Mesh* genmesh,
 DEVSTRUCT& dev_struct) const
{ // This routine is conceptually const.

  // GPU_Demag requires a rectangular mesh
  const Oxs_CommonRectangularMesh* mesh
    = dynamic_cast<const Oxs_CommonRectangularMesh*>(genmesh);
  if(mesh==NULL) {
    String msg=String("Object ")
      + String(genmesh->InstanceName())
      + String(" is not a rectangular mesh.");
    throw Oxs_ExtError(this,msg);
  }
  // Check periodicity
  const Oxs_RectangularMesh* rmesh 
    = dynamic_cast<const Oxs_RectangularMesh*>(mesh);
  const Oxs_PeriodicRectangularMesh* pmesh
    = dynamic_cast<const Oxs_PeriodicRectangularMesh*>(mesh);
  if(pmesh!=NULL) {
    // Rectangular, periodic mesh
    xperiodic = pmesh->IsPeriodicX();
    yperiodic = pmesh->IsPeriodicY();
    zperiodic = pmesh->IsPeriodicZ();

    // Check for supported periodicity
    if(xperiodic+yperiodic+zperiodic>2) {
      String msg=String("Periodic mesh ")
        + String(genmesh->InstanceName())
        + String("is 3D periodic, which is not supported by GPU_Demag."
                 "  Select no more than two of x, y, or z.");
      throw Oxs_ExtError(this,msg.c_str());
    }
    if(xperiodic+yperiodic+zperiodic>1) {
      String msg=String("Periodic mesh ")
        + String(genmesh->InstanceName())
      + String("is 2D periodic, which is not supported by GPU_Demag"
               " at this time.");
      throw Oxs_ExtError(this,msg.c_str());
    }
	if(xperiodic+yperiodic+zperiodic>0){
	  String msg=String("Periodic mesh ")
        + String(genmesh->InstanceName())
      + String("is 1D periodic, which is not supported by GPU_Demag"
               " at this time.");
      throw Oxs_ExtError(this,msg.c_str());
	}
  } else if (rmesh!=NULL) {
    // Rectangular, non-periodic mesh
    xperiodic=0; yperiodic=0; zperiodic=0;
  } else {
    String msg=String("Unknown mesh type: \"")
      + String(ClassName())
      + String("\".");
    throw Oxs_ExtError(this,msg.c_str());
  }
  // Clean-up from previous allocation, if any.
  ReleaseMemory();
#if REPORT_TIME
  cudaDeviceSynchronize();
    inittime.Start();
#endif // REPORT_TIME
  // Fill dimension variables
  rdimx = mesh->DimX();
  rdimy = mesh->DimY();
  rdimz = mesh->DimZ();
  if(rdimx==0 || rdimy==0 || rdimz==0) return; // Empty mesh!
  // Initialize fft object.  If a dimension equals 1, then zero
  // padding is not required.  Otherwise, zero pad to at least
  // twice the dimension.
  // NOTE: This is not coded yet, but if periodic is requested and
  // dimension is directly compatible with FFT (e.g., the dimension
  // is a power-of-2) then zero padding is not required.
  Oxs_FFT3DThreeVector::RecommendDimensions((rdimx==1 ? 1 : 2*rdimx),
                                            (rdimy==1 ? 1 : 2*rdimy),
                                            (rdimz==1 ? 1 : 2*rdimz),
                                            cdimx,cdimy,cdimz);
  OC_INDEX xfrm_size = ODTV_VECSIZE * 2 * cdimx * cdimy * cdimz;
  // "ODTV_VECSIZE" here is because we work with arrays if ThreeVectors,
  // and "2" because these are complex (as opposed to real)
  // quantities.
  if(xfrm_size<cdimx || xfrm_size<cdimy || xfrm_size<cdimz ||
     long(xfrm_size) != 
     long(2*ODTV_VECSIZE)*long(cdimx)*long(cdimy)*long(cdimz)) {
    // Partial overflow check
    char msgbuf[1024];
    Oc_Snprintf(msgbuf,sizeof(msgbuf),
                ": Product 2*ODTV_VECSIZE*cdimx*cdimy*cdimz = "
                "2*%d*%d*%d*%d too big to fit in a OC_INDEX variable",
                ODTV_VECSIZE,cdimx,cdimy,cdimz);
    String msg =
      String("OC_INDEX overflow in ")
      + String(InstanceName())
      + String(msgbuf);
    throw Oxs_ExtError(this,msg);
  }

  //********This is where demam.cc allocate temporary data
  //****Mtemp = new OXS_FFT_REAL_TYPE[ODTV_VECSIZE*rdimx*rdimy*rdimz];
  /// Temporary space to hold Ms[]*m[].  The plan is to make this space
  /// unnecessary by introducing FFT functions that can take Ms as input
  /// and do the multiplication on the fly.


  // The following 3 statements are cribbed from
  // Oxs_FFT3DThreeVector::SetDimensions().  The corresponding
  // code using that class is
  //
  //  Oxs_FFT3DThreeVector fft;
  //  fft.SetDimensions(rdimx,rdimy,rdimz,cdimx,cdimy,cdimz);
  //  fft.GetLogicalDimensions(ldimx,ldimy,ldimz);
  //
  //*******CAREFUL, THIS MAY BE NECESSARY****************************
  fftx.SetDimensions(rdimx, (cdimx==1 ? 1 : 2*(cdimx-1)), rdimy);
  ffty.SetDimensions(rdimy, cdimy,
                     ODTV_COMPLEXSIZE*ODTV_VECSIZE*cdimx,
                     ODTV_VECSIZE*cdimx);
  fftz.SetDimensions(rdimz, cdimz,
                     ODTV_COMPLEXSIZE*ODTV_VECSIZE*cdimx*cdimy,
                     ODTV_VECSIZE*cdimx*cdimy);
  OC_INDEX ldimx,ldimy,ldimz; // Logical dimensions
  // The following 3 statements are cribbed from
  // Oxs_FFT3DThreeVector::GetLogicalDimensions()
  ldimx = fftx.GetLogicalDimension();
  ldimy = ffty.GetLogicalDimension();
  ldimz = fftz.GetLogicalDimension();

  adimx = (ldimx/2)+1;
  adimy = (ldimy/2)+1;
  adimz = (ldimz/2)+1;
  
  fdimx = ldimx;
  fdimy = ldimy;
  fdimz = ldimz;

  rsize = rdimx * rdimy * rdimz;
  csize = cdimx * cdimy * cdimz;
  asize = adimx * adimy * adimz;
  fsize = fdimx * fdimy * fdimz;
#if VERBOSE_DEBUG && !defined(NDEBUG)
  fprintf(stderr,"RDIMS: (%d,%d,%d)\n",rdimx,rdimy,rdimz); /**/
  fprintf(stderr,"CDIMS: (%d,%d,%d)\n",cdimx,cdimy,cdimz); /**/
  fprintf(stderr,"LDIMS: (%d,%d,%d)\n",ldimx,ldimy,ldimz); /**/
  fprintf(stderr,"ADIMS: (%d,%d,%d)\n",adimx,adimy,adimz); /**/
#endif // NDEBUG

  //allocate memory on GPU RAM, which will be named device memory
  AllocDevMemory(dev_struct);

  // Dimension of array necessary to hold 3 sets of full interaction
  // coefficients in real space:
  OC_INDEX scratch_size = ODTV_VECSIZE * ldimx * ldimy * ldimz;
  if(scratch_size<ldimx || scratch_size<ldimy || scratch_size<ldimz) {
    // Partial overflow check
    String msg =
      String("OC_INDEX overflow in ")
      + String(InstanceName())
      + String(": Product 3*8*rdimx*rdimy*rdimz too big"
               " to fit in a OC_INDEX variable");
    throw Oxs_ExtError(this,msg);
  }
  // Allocate memory for FFT xfrm target H, and scratch space
  // for computing interaction coefficients
  OXS_FFT_REAL_TYPE* Hxfrm = new OXS_FFT_REAL_TYPE[xfrm_size];
  OXS_FFT_REAL_TYPE* scratch = new OXS_FFT_REAL_TYPE[scratch_size];

  if(Hxfrm==NULL || scratch==NULL) {
    // Safety check for those machines on which new[] doesn't throw
    // BadAlloc.
    String msg = String("Insufficient memory in Demag setup.");
    throw Oxs_ExtError(this,msg);
  }

  // According (16) in Newell's paper, the demag field is given by
  //                        H = -N*M
  // where N is the "demagnetizing tensor," with components Nxx, Nxy,
  // etc.  With the '-1' in 'scale' we store '-N' instead of 'N',
  // so we don't have to multiply the output from the FFT + iFFT
  // by -1 GetEnergy() below.

  // Fill interaction matrices with demag coefs from Newell's paper.
  // Note that A00, A11 and A22 are even in x,y and z.
  // A01 is odd in x and y, even in z.
  // A02 is odd in x and z, even in y.
  // A12 is odd in y and z, even in x.
  // We use these symmetries to reduce storage requirements.  If
  // f is real and even, then f^ is also real and even.  If f
  // is real and odd, then f^ is (pure) imaginary and odd.
  // As a result, the transform of each of the A## interaction
  // matrices will be real, with the same even/odd properties.
  //
  // Notation:  A00:=fs*Nxx, A01:=fs*Nxy, A02:=fs*Nxz,
  //                         A11:=fs*Nyy, A12:=fs*Nyz
  //                                      A22:=fs*Nzz
  //  where fs = -1/((ldimx/2)*ldimy*ldimz)

  OC_REAL8m dx = mesh->EdgeLengthX();
  OC_REAL8m dy = mesh->EdgeLengthY();
  OC_REAL8m dz = mesh->EdgeLengthZ();
  // For demag calculation, all that matters is the relative
  // size of dx, dy and dz.  If possible, rescale these to
  // integers, as this may help reduce floating point error
  // a wee bit.  If not possible, then rescale so largest
  // value is 1.0.
  {
    OC_REALWIDE p1,q1,p2,q2;
    if(Nb_FindRatApprox(dx,dy,1e-12,1000,p1,q1)
       && Nb_FindRatApprox(dz,dy,1e-12,1000,p2,q2)) {
      OC_REALWIDE gcd = Nb_GcdRW(q1,q2);
      dx = p1*q2/gcd;
      dy = q1*q2/gcd;
      dz = p2*q1/gcd;
    } else {
      OC_REALWIDE maxedge=dx;
      if(dy>maxedge) maxedge=dy;
      if(dz>maxedge) maxedge=dz;
      dx/=maxedge; dy/=maxedge; dz/=maxedge;
    }
  }
  OC_REALWIDE scale = 1.0/(4*WIDE_PI*dx*dy*dz);
  const OXS_DEMAG_REAL_ASYMP scaled_arad = asymptotic_radius
    * Oc_Pow(OXS_DEMAG_REAL_ASYMP(dx*dy*dz),
             OXS_DEMAG_REAL_ASYMP(1.)/OXS_DEMAG_REAL_ASYMP(3.));
  // Also throw in FFT scaling.  This allows the "NoScale" FFT routines
  // to be used.  NB: There is effectively a "-1" built into the
  // differencing sections below, because we compute d^6/dx^2 dy^2 dz^2
  // instead of -d^6/dx^2 dy^2 dz^2 as required.
  // Note: Using an Oxs_FFT3DThreeVector fft object, this would be just
  //    scale *= fft.GetScaling();
  scale *= fftx.GetScaling() * ffty.GetScaling() * fftz.GetScaling();

  OC_INDEX i,j,k;
  OC_INDEX sstridey = ODTV_VECSIZE*ldimx;
  OC_INDEX sstridez = sstridey*ldimy;
  OC_INDEX kstop=1; if(rdimz>1) kstop=rdimz+2;
  OC_INDEX jstop=1; if(rdimy>1) jstop=rdimy+2;
  OC_INDEX istop=1; if(rdimx>1) istop=rdimx+2;
  if(scaled_arad>0) {
    // We don't need to compute analytic formulae outside
    // asymptotic radius
    OC_INDEX ktest = static_cast<OC_INDEX>(Oc_Ceil(scaled_arad/dz))+2;
    if(ktest<kstop) kstop = ktest;
    OC_INDEX jtest = static_cast<OC_INDEX>(Oc_Ceil(scaled_arad/dy))+2;
    if(jtest<jstop) jstop = jtest;
    OC_INDEX itest = static_cast<OC_INDEX>(Oc_Ceil(scaled_arad/dx))+2;
    if(itest<istop) istop = itest;
  }

#if REPORT_TIME
  dvltimer[0].Start();
#endif // REPORT_TIME
  if(!xperiodic && !yperiodic && !zperiodic) {
    // Calculate Nxx, Nxy and Nxz in first octant, non-periodic case.
    // Step 1: Evaluate f & g at each cell site.  Offset by (-dx,-dy,-dz)
    //  so we can do 2nd derivative operations "in-place".
    for(k=0;k<kstop;k++) {
      OC_INDEX kindex = k*sstridez;
      OC_REALWIDE z = dz*(k-1);
      for(j=0;j<jstop;j++) {
        OC_INDEX jkindex = kindex + j*sstridey;
        OC_REALWIDE y = dy*(j-1);
        for(i=0;i<istop;i++) {
          OC_INDEX index = ODTV_VECSIZE*i+jkindex;
#ifndef NDEBUG
          if(index>=scratch_size) {
            String msg = String("Programming error:"
                                " array index out-of-bounds.");
            throw Oxs_ExtError(this,msg);
          }
#endif // NDEBUG
          OC_REALWIDE x = dx*(i-1);
          // Nyy(x,y,z) = Nxx(y,x,z);  Nzz(x,y,z) = Nxx(z,y,x);
          // Nxz(x,y,z) = Nxy(x,z,y);  Nyz(x,y,z) = Nxy(y,z,x);
          scratch[index]   = scale*Oxs_Newell_f(x,y,z);  // For Nxx
          scratch[index+1] = scale*Oxs_Newell_g(x,y,z);  // For Nxy
          scratch[index+2] = scale*Oxs_Newell_g(x,z,y);  // For Nxz
        }
      }
    }

    // Step 2a: Do d^2/dz^2
    if(kstop==1) {
      // Only 1 layer in z-direction of f/g stored in scratch array.
      for(j=0;j<jstop;j++) {
        OC_INDEX jkindex = j*sstridey;
        OC_REALWIDE y = dy*(j-1);
        for(i=0;i<istop;i++) {
          OC_INDEX index = ODTV_VECSIZE*i+jkindex;
          OC_REALWIDE x = dx*(i-1);
          // Function f is even in each variable, so for example
          //    f(x,y,-dz) - 2f(x,y,0) + f(x,y,dz)
          //        =  2( f(x,y,-dz) - f(x,y,0) )
          // Function g(x,y,z) is even in z and odd in x and y,
          // so for example
          //    g(x,-dz,y) - 2g(x,0,y) + g(x,dz,y)
          //        =  2g(x,0,y) = 0.
          // Nyy(x,y,z) = Nxx(y,x,z);  Nzz(x,y,z) = Nxx(z,y,x);
          // Nxz(x,y,z) = Nxy(x,z,y);  Nyz(x,y,z) = Nxy(y,z,x);
          scratch[index]   -= scale*Oxs_Newell_f(x,y,0);
          scratch[index]   *= 2;
          scratch[index+1] -= scale*Oxs_Newell_g(x,y,0);
          scratch[index+1] *= 2;
          scratch[index+2] = 0;
        }
      }
    } else {
      for(k=0;k<rdimz;k++) {
        OC_INDEX kindex = k*sstridez;
        for(j=0;j<jstop;j++) {
          OC_INDEX jkindex = kindex + j*sstridey;
          for(i=0;i<istop;i++) {
            OC_INDEX index = ODTV_VECSIZE*i+jkindex;
            scratch[index]   += -2*scratch[index+sstridez]
              + scratch[index+2*sstridez];
            scratch[index+1] += -2*scratch[index+sstridez+1]
              + scratch[index+2*sstridez+1];
            scratch[index+2] += -2*scratch[index+sstridez+2]
              + scratch[index+2*sstridez+2];
          }
        }
      }
    }
    // Step 2b: Do d^2/dy^2
    if(jstop==1) {
      // Only 1 layer in y-direction of f/g stored in scratch array.
      for(k=0;k<rdimz;k++) {
        OC_INDEX kindex = k*sstridez;
        OC_REALWIDE z = dz*k;
        for(i=0;i<istop;i++) {
          OC_INDEX index = ODTV_VECSIZE*i+kindex;
          OC_REALWIDE x = dx*(i-1);
          // Function f is even in each variable, so for example
          //    f(x,y,-dz) - 2f(x,y,0) + f(x,y,dz)
          //        =  2( f(x,y,-dz) - f(x,y,0) )
          // Function g(x,y,z) is even in z and odd in x and y,
          // so for example
          //    g(x,-dz,y) - 2g(x,0,y) + g(x,dz,y)
          //        =  2g(x,0,y) = 0.
          // Nyy(x,y,z) = Nxx(y,x,z);  Nzz(x,y,z) = Nxx(z,y,x);
          // Nxz(x,y,z) = Nxy(x,z,y);  Nyz(x,y,z) = Nxy(y,z,x);
          scratch[index]   -= scale * 
            ((Oxs_Newell_f(x,0,z-dz)+Oxs_Newell_f(x,0,z+dz))
             -2*Oxs_Newell_f(x,0,z));
          scratch[index]   *= 2;
          scratch[index+1]  = 0.0;
          scratch[index+2] -= scale * 
            ((Oxs_Newell_g(x,z-dz,0)+Oxs_Newell_g(x,z+dz,0))
             -2*Oxs_Newell_g(x,z,0));
          scratch[index+2] *= 2;
        }
      }
    } else {
      for(k=0;k<rdimz;k++) {
        OC_INDEX kindex = k*sstridez;
        for(j=0;j<rdimy;j++) {
          OC_INDEX jkindex = kindex + j*sstridey;
          for(i=0;i<istop;i++) {
            OC_INDEX index = ODTV_VECSIZE*i+jkindex;
            scratch[index]   += -2*scratch[index+sstridey]
              + scratch[index+2*sstridey];
            scratch[index+1] += -2*scratch[index+sstridey+1]
              + scratch[index+2*sstridey+1];
            scratch[index+2] += -2*scratch[index+sstridey+2]
              + scratch[index+2*sstridey+2];
          }
        }
      }
    }

    // Step 2c: Do d^2/dx^2
    if(istop==1) {
      // Only 1 layer in x-direction of f/g stored in scratch array.
      for(k=0;k<rdimz;k++) {
        OC_INDEX kindex = k*sstridez;
        OC_REALWIDE z = dz*k;
        for(j=0;j<rdimy;j++) {
          OC_INDEX index = kindex + j*sstridey;
          OC_REALWIDE y = dy*j;
          // Function f is even in each variable, so for example
          //    f(x,y,-dz) - 2f(x,y,0) + f(x,y,dz)
          //        =  2( f(x,y,-dz) - f(x,y,0) )
          // Function g(x,y,z) is even in z and odd in x and y,
          // so for example
          //    g(x,-dz,y) - 2g(x,0,y) + g(x,dz,y)
          //        =  2g(x,0,y) = 0.
          // Nyy(x,y,z) = Nxx(y,x,z);  Nzz(x,y,z) = Nxx(z,y,x);
          // Nxz(x,y,z) = Nxy(x,z,y);  Nyz(x,y,z) = Nxy(y,z,x);
          scratch[index]   -= scale * 
            ((4*Oxs_Newell_f(0,y,z)
              +Oxs_Newell_f(0,y+dy,z+dz)+Oxs_Newell_f(0,y-dy,z+dz)
              +Oxs_Newell_f(0,y+dy,z-dz)+Oxs_Newell_f(0,y-dy,z-dz))
             -2*(Oxs_Newell_f(0,y+dy,z)+Oxs_Newell_f(0,y-dy,z)
                 +Oxs_Newell_f(0,y,z+dz)+Oxs_Newell_f(0,y,z-dz)));
          scratch[index]   *= 2;                       // For Nxx
          scratch[index+2]  = scratch[index+1] = 0.0;  // For Nxy & Nxz
        }
      }
    } else {
      for(k=0;k<rdimz;k++) {
        OC_INDEX kindex = k*sstridez;
        for(j=0;j<rdimy;j++) {
          OC_INDEX jkindex = kindex + j*sstridey;
          for(i=0;i<rdimx;i++) {
            OC_INDEX index = ODTV_VECSIZE*i+jkindex;
            scratch[index]   += -2*scratch[index+  ODTV_VECSIZE]
              + scratch[index+2*ODTV_VECSIZE];
            scratch[index+1] += -2*scratch[index+  ODTV_VECSIZE+1]
              + scratch[index+2*ODTV_VECSIZE+1];
            scratch[index+2] += -2*scratch[index+  ODTV_VECSIZE+2]
              + scratch[index+2*ODTV_VECSIZE+2];
          }
        }
      }
    }

    // Special "SelfDemag" code may be more accurate at index 0,0,0.
    // Note: Using an Oxs_FFT3DThreeVector fft object, this would be
    //    scale *= fft.GetScaling();
    const OXS_FFT_REAL_TYPE selfscale
      = -1 * fftx.GetScaling() * ffty.GetScaling() * fftz.GetScaling();
    scratch[0] = Oxs_SelfDemagNx(dx,dy,dz);
    if(zero_self_demag) scratch[0] -= 1./3.;
    scratch[0] *= selfscale;

    scratch[1] = 0.0;  // Nxy[0] = 0.

    scratch[2] = 0.0;  // Nxz[0] = 0.

    // Step 2.5: Use asymptotic (dipolar + higher) approximation for far field
    /*   Dipole approximation:
     *
     *                        / 3x^2-R^2   3xy       3xz    \
     *             dx.dy.dz   |                             |
     *  H_demag = ----------- |   3xy   3y^2-R^2     3yz    |
     *             4.pi.R^5   |                             |
     *                        \   3xz      3yz     3z^2-R^2 /
     */
    // See Notes IV, 26-Feb-2007, p102.
    if(scaled_arad>=0.0) {
      // Note that all distances here are in "reduced" units,
      // scaled so that dx, dy, and dz are either small integers
      // or else close to 1.0.
      OXS_DEMAG_REAL_ASYMP scaled_arad_sq = scaled_arad*scaled_arad;
      OXS_FFT_REAL_TYPE fft_scaling = -1 *
        fftx.GetScaling() * ffty.GetScaling() * fftz.GetScaling();
      /// Note: Since H = -N*M, and by convention with the rest of this
      /// code, we store "-N" instead of "N" so we don't have to multiply
      /// the output from the FFT + iFFT by -1 in GetEnergy() below.

      OXS_DEMAG_REAL_ASYMP xtest
        = static_cast<OXS_DEMAG_REAL_ASYMP>(rdimx)*dx;
      xtest *= xtest;

      Oxs_DemagNxxAsymptotic ANxx(dx,dy,dz);
      Oxs_DemagNxyAsymptotic ANxy(dx,dy,dz);
      Oxs_DemagNxzAsymptotic ANxz(dx,dy,dz);

      for(k=0;k<rdimz;++k) {
        OC_INDEX kindex = k*sstridez;
        OXS_DEMAG_REAL_ASYMP z = dz*k;
        OXS_DEMAG_REAL_ASYMP zsq = z*z;
        for(j=0;j<rdimy;++j) {
          OC_INDEX jkindex = kindex + j*sstridey;
          OXS_DEMAG_REAL_ASYMP y = dy*j;
          OXS_DEMAG_REAL_ASYMP ysq = y*y;

          OC_INDEX istart = 0;
          OXS_DEMAG_REAL_ASYMP test = scaled_arad_sq-ysq-zsq;
          if(test>0) {
            if(test>xtest) {
              istart = rdimx+1;
            } else {
              istart = static_cast<OC_INDEX>(Oc_Ceil(Oc_Sqrt(test)/dx));
            }
          }
          for(i=istart;i<rdimx;++i) {
            OC_INDEX index = ODTV_VECSIZE*i+jkindex;
            OXS_DEMAG_REAL_ASYMP x = dx*i;
            scratch[index]   = fft_scaling*ANxx.NxxAsymptotic(x,y,z);
            scratch[index+1] = fft_scaling*ANxy.NxyAsymptotic(x,y,z);
            scratch[index+2] = fft_scaling*ANxz.NxzAsymptotic(x,y,z);
          }
        }
      }
#if 0
      fprintf(stderr,"ANxx(%d,%d,%d) = %#.16g (non-threaded)\n",
              int(rdimx-1),int(rdimy-1),int(rdimz-1),
              (double)ANxx.NxxAsymptotic(dx*(rdimx-1),
                                         dy*(rdimy-1),dz*(rdimz-1)));
      OC_INDEX icheck = ODTV_VECSIZE*(rdimx-1)
        + (rdimy-1)*sstridey + (rdimz-1)*sstridez;
      fprintf(stderr,"fft_scaling=%g, product=%#.16g\n",
              (double)fft_scaling,(double)scratch[icheck]);
#endif
    }
#if 0
    {
      OXS_FFT_REAL_TYPE fft_scaling = -1 *
        fftx.GetScaling() * ffty.GetScaling() * fftz.GetScaling();
      OXS_DEMAG_REAL_ASYMP gamma
        = Oc_Pow(OXS_DEMAG_REAL_ASYMP(dx*dy*dz),
                 OXS_DEMAG_REAL_ASYMP(1.)/OXS_DEMAG_REAL_ASYMP(3.));
      fprintf(stderr,"dx=%Lg, dy=%Lg, dz=%Lg\n",dx,dy,dz);
      OC_INDEX qi,qj,qk,qo;
      OXS_DEMAG_REAL_ASYMP qd;
#define FOO(QA,QB,QC) \
      qi = (OC_INDEX)Oc_Floor(0.5+(QA)*gamma/dx), \
        qj = (OC_INDEX)Oc_Floor(0.5+(QB)*gamma/dy),   \
        qk = (OC_INDEX)Oc_Floor(0.5+(QC)*gamma/dz),     \
        qo = ODTV_VECSIZE*qi + qj*sstridey + qk*sstridez, \
        qd = Oc_Sqrt((qi*dx)*(qi*dx)+(qj*dy)*(qj*dy)+(qk*dz)*(qk*dz))/gamma; \
      fprintf(stderr,"Nxx/Nxy(%3ld,%3ld,%3ld) (dist %24.16Le) = %24.16Le    %24.16Le\n", \
              qi,qj,qk,qd,scratch[qo]/fft_scaling,scratch[qo+1]/fft_scaling)
      FOO(7,3,2);
      FOO(15,7,4);
      FOO(22,10,6);
      FOO(30,14,8);
      FOO(45,21,12);
      FOO(60,28,16);
      FOO(90,42,24);
      FOO(120,56,32);
      FOO(240,112,64);
      // FOO(480,224,128);
    }
#endif
  }

  // Step 2.6: If periodic boundaries selected, compute periodic tensors
  // instead.
  // NOTE THAT CODE BELOW CURRENTLY ONLY SUPPORTS 1D PERIODIC!!!
  // NOTE 2: Keep this code in sync with that in
  //         GPU_Demag::IncrementPreconditioner
#if 1
SDA00_count = 0;
SDA01_count = 0;
#endif
  if(xperiodic) {
    OXS_FFT_REAL_TYPE fft_scaling = -1 *
      fftx.GetScaling() * ffty.GetScaling() * fftz.GetScaling();
    /// Note: Since H = -N*M, and by convention with the rest of this
    /// code, we store "-N" instead of "N" so we don't have to multiply
    /// the output from the FFT + iFFT by -1 in GetEnergy() below.

    Oxs_DemagPeriodicX pbctensor(dx,dy,dz,rdimx*dx,scaled_arad);

    for(k=0;k<rdimz;++k) {
      OC_INDEX kindex = k*sstridez;
      OXS_DEMAG_REAL_ASYMP z = dz*k;
      for(j=0;j<rdimy;++j) {
        OC_INDEX jkindex = kindex + j*sstridey;
        OXS_DEMAG_REAL_ASYMP y = dy*j;
        OXS_DEMAG_REAL_ASYMP Nxx, Nxy, Nxz;

        i=0;
        OXS_DEMAG_REAL_ASYMP x = dx*i;
        OC_INDEX index = ODTV_VECSIZE*i+jkindex;
        pbctensor.NxxNxyNxz(x,y,z,Nxx,Nxy,Nxz);
        scratch[index]   = fft_scaling*Nxx;
        scratch[index+1] = fft_scaling*Nxy;
        scratch[index+2] = fft_scaling*Nxz;

        for(i=1;2*i<rdimx;++i) {
          // Interior i; reflect results from left to right half
          x = dx*i;
          index = ODTV_VECSIZE*i+jkindex;
          OC_INDEX rindex = ODTV_VECSIZE*(rdimx-i)+jkindex;
          pbctensor.NxxNxyNxz(x,y,z,Nxx,Nxy,Nxz);
          scratch[index]   = fft_scaling*Nxx; // pbctensor computation
          scratch[index+1] = fft_scaling*Nxy; // *includes* base window
          scratch[index+2] = fft_scaling*Nxz; // term
          scratch[rindex]   =    scratch[index];   // Nxx is even
          scratch[rindex+1] = -1*scratch[index+1]; // Nxy is odd wrt x
          scratch[rindex+2] = -1*scratch[index+2]; // Nxz is odd wrt x
        }

        if(rdimx%2 == 0) { // Do midpoint seperately
          i = rdimx/2;
          x = dx*i;
          index = ODTV_VECSIZE*i+jkindex;
          pbctensor.NxxNxyNxz(x,y,z,Nxx,Nxy,Nxz);
          scratch[index]   = fft_scaling*Nxx;
          scratch[index+1] = fft_scaling*Nxy;
          scratch[index+2] = fft_scaling*Nxz;
        }

      }
    }
  }
  if(yperiodic) {
    OXS_FFT_REAL_TYPE fft_scaling = -1 *
      fftx.GetScaling() * ffty.GetScaling() * fftz.GetScaling();
    /// Note: Since H = -N*M, and by convention with the rest of this
    /// code, we store "-N" instead of "N" so we don't have to multiply
    /// the output from the FFT + iFFT by -1 in GetEnergy() below.

    Oxs_DemagPeriodicY pbctensor(dx,dy,dz,rdimy*dy,scaled_arad);

    for(k=0;k<rdimz;++k) {
      OC_INDEX kindex = k*sstridez;
      OXS_DEMAG_REAL_ASYMP z = dz*k;
      for(j=0;j<=rdimy/2;++j) {
        OC_INDEX jkindex = kindex + j*sstridey;
        OXS_DEMAG_REAL_ASYMP y = dy*j;
        if(0<j && 2*j<rdimy) {
          // Interior j; reflect results from lower to upper half
          OC_INDEX rjkindex = kindex + (rdimy-j)*sstridey;
          for(i=0;i<rdimx;++i) {
            OC_INDEX index = ODTV_VECSIZE*i+jkindex;
            OC_INDEX rindex = ODTV_VECSIZE*i+rjkindex;
            OXS_DEMAG_REAL_ASYMP x = dx*i;
            OXS_DEMAG_REAL_ASYMP Nxx, Nxy, Nxz;
            pbctensor.NxxNxyNxz(x,y,z,Nxx,Nxy,Nxz);
            scratch[index]   = fft_scaling*Nxx;
            scratch[index+1] = fft_scaling*Nxy;
            scratch[index+2] = fft_scaling*Nxz;
            scratch[rindex]   =    scratch[index];   // Nxx is even
            scratch[rindex+1] = -1*scratch[index+1]; // Nxy is odd wrt y
            scratch[rindex+2] =    scratch[index+2]; // Nxz is even wrt y
          }
        } else { // j==0 or midpoint
          for(i=0;i<rdimx;++i) {
            OC_INDEX index = ODTV_VECSIZE*i+jkindex;
            OXS_DEMAG_REAL_ASYMP x = dx*i;
            OXS_DEMAG_REAL_ASYMP Nxx, Nxy, Nxz;
            pbctensor.NxxNxyNxz(x,y,z,Nxx,Nxy,Nxz);
            scratch[index]   = fft_scaling*Nxx;
            scratch[index+1] = fft_scaling*Nxy;
            scratch[index+2] = fft_scaling*Nxz;
          }
        }
      }
    }
  }
  if(zperiodic) {
    OXS_FFT_REAL_TYPE fft_scaling = -1 *
      fftx.GetScaling() * ffty.GetScaling() * fftz.GetScaling();
    /// Note: Since H = -N*M, and by convention with the rest of this
    /// code, we store "-N" instead of "N" so we don't have to multiply
    /// the output from the FFT + iFFT by -1 in GetEnergy() below.

    Oxs_DemagPeriodicZ pbctensor(dx,dy,dz,rdimz*dz,scaled_arad);

    for(k=0;k<=rdimz/2;++k) {
      OC_INDEX kindex = k*sstridez;
      OXS_DEMAG_REAL_ASYMP z = dz*k;
      if(0<k && 2*k<rdimz) {
        // Interior k; reflect results from lower to upper half
        OC_INDEX rkindex = (rdimz-k)*sstridez;
        for(j=0;j<rdimy;++j) {
          OC_INDEX jkindex = kindex + j*sstridey;
          OC_INDEX rjkindex = rkindex + j*sstridey;
          OXS_DEMAG_REAL_ASYMP y = dy*j;
          for(i=0;i<rdimx;++i) {
            OXS_DEMAG_REAL_ASYMP x = dx*i;
            OC_INDEX index = ODTV_VECSIZE*i+jkindex;
            OC_INDEX rindex = ODTV_VECSIZE*i+rjkindex;
            OXS_DEMAG_REAL_ASYMP Nxx, Nxy, Nxz;
            pbctensor.NxxNxyNxz(x,y,z,Nxx,Nxy,Nxz);
            scratch[index]   = fft_scaling*Nxx;
            scratch[index+1] = fft_scaling*Nxy;
            scratch[index+2] = fft_scaling*Nxz;
            scratch[rindex]   =    scratch[index];   // Nxx is even
            scratch[rindex+1] =    scratch[index+1]; // Nxy is even wrt z
            scratch[rindex+2] = -1*scratch[index+2]; // Nxz is odd wrt z
          }
        }
      } else { // k==0 or midpoint
        for(j=0;j<rdimy;++j) {
          OC_INDEX jkindex = kindex + j*sstridey;
          OXS_DEMAG_REAL_ASYMP y = dy*j;
          for(i=0;i<rdimx;++i) {
            OXS_DEMAG_REAL_ASYMP x = dx*i;
            OC_INDEX index = ODTV_VECSIZE*i+jkindex;
            OXS_DEMAG_REAL_ASYMP Nxx, Nxy, Nxz;
            pbctensor.NxxNxyNxz(x,y,z,Nxx,Nxy,Nxz);
            scratch[index]   = fft_scaling*Nxx;
            scratch[index+1] = fft_scaling*Nxy;
            scratch[index+2] = fft_scaling*Nxz;
          }
        }
      }
    }
  }
#if REPORT_TIME
  dvltimer[0].Stop();
#endif // REPORT_TIME
#if 1
    printf("rdimx=%ld, rdimy=%ld, rdimz=%ld,"
           " SDA00_count = %ld, SDA01_count = %ld\n",
           (long int)rdimx,(long int)rdimy,(long int)rdimz,
           (long int)SDA00_count,(long int)SDA01_count);
#endif
#if 0
    for(k=0;k<rdimz;++k) {
      OC_INDEX kindex = k*sstridez;
      for(j=0;j<rdimy;++j) {
        OC_INDEX jkindex = kindex + j*sstridey;
        for(i=0;i<rdimx;++i) {
          OC_INDEX index = ODTV_VECSIZE*i+jkindex;
          printf("A[%2d][%2d][%2d].Nxx=%25.16e"
                 " Nxy=%25.16e Nxz=%25.16e\n",
                 i,j,k,
                 scratch[index],
                 scratch[index+1],
                 scratch[index+2]);
        }
      }
    }
#endif

  // Step 3: Use symmetries to reflect into other octants.
  //     Also, at each coordinate plane, set to 0.0 any term
  //     which is odd across that boundary.  It should already
  //     be close to 0, but will likely be slightly off due to
  //     rounding errors.
  // Symmetries: A00, A11, A22 are even in each coordinate
  //             A01 is odd in x and y, even in z.
  //             A02 is odd in x and z, even in y.
  //             A12 is odd in y and z, even in x.
#if REPORT_TIME
  dvltimer[1].Start();
#endif // REPORT_TIME
  for(k=0;k<rdimz;k++) {
    OC_INDEX kindex = k*sstridez;
    for(j=0;j<rdimy;j++) {
      OC_INDEX jkindex = kindex + j*sstridey;
      for(i=0;i<rdimx;i++) {
        OC_INDEX index = ODTV_VECSIZE*i+jkindex;

        if(i==0 || j==0) scratch[index+1] = 0.0;  // A01
        if(i==0 || k==0) scratch[index+2] = 0.0;  // A02

        OXS_FFT_REAL_TYPE tmpA00 = scratch[index];
        OXS_FFT_REAL_TYPE tmpA01 = scratch[index+1];
        OXS_FFT_REAL_TYPE tmpA02 = scratch[index+2];
        if(i>0) {
          OC_INDEX tindex = ODTV_VECSIZE*(ldimx-i)+j*sstridey+k*sstridez;
          scratch[tindex]   =     tmpA00;
          scratch[tindex+1] =  -1*tmpA01;
          scratch[tindex+2] =  -1*tmpA02;
        }
        if(j>0) {
          OC_INDEX tindex = ODTV_VECSIZE*i+(ldimy-j)*sstridey+k*sstridez;
          scratch[tindex]   =     tmpA00;
          scratch[tindex+1] =  -1*tmpA01;
          scratch[tindex+2] =     tmpA02;
        }
        if(k>0) {
          OC_INDEX tindex = ODTV_VECSIZE*i+j*sstridey+(ldimz-k)*sstridez;
          scratch[tindex]   =     tmpA00;
          scratch[tindex+1] =     tmpA01;
          scratch[tindex+2] =  -1*tmpA02;
        }
        if(i>0 && j>0) {
          OC_INDEX tindex
            = ODTV_VECSIZE*(ldimx-i) + (ldimy-j)*sstridey + k*sstridez;
          scratch[tindex]   =     tmpA00;
          scratch[tindex+1] =     tmpA01;
          scratch[tindex+2] =  -1*tmpA02;
        }
        if(i>0 && k>0) {
          OC_INDEX tindex
            = ODTV_VECSIZE*(ldimx-i) + j*sstridey + (ldimz-k)*sstridez;
          scratch[tindex]   =     tmpA00;
          scratch[tindex+1] =  -1*tmpA01;
          scratch[tindex+2] =     tmpA02;
        }
        if(j>0 && k>0) {
          OC_INDEX tindex
            = ODTV_VECSIZE*i + (ldimy-j)*sstridey + (ldimz-k)*sstridez;
          scratch[tindex]   =     tmpA00;
          scratch[tindex+1] =  -1*tmpA01;
          scratch[tindex+2] =  -1*tmpA02;
        }
        if(i>0 && j>0 && k>0) {
          OC_INDEX tindex
            = ODTV_VECSIZE*(ldimx-i) + (ldimy-j)*sstridey + (ldimz-k)*sstridez;
          scratch[tindex]   =     tmpA00;
          scratch[tindex+1] =     tmpA01;
          scratch[tindex+2] =     tmpA02;
        }
      }
    }
  }

  // Step 3.5: Fill in zero-padded overhang
  for(k=0;k<ldimz;k++) {
    OC_INDEX kindex = k*sstridez;
    if(k<rdimz || k>ldimz-rdimz) { // Outer k
      for(j=0;j<ldimy;j++) {
        OC_INDEX jkindex = kindex + j*sstridey;
        if(j<rdimy || j>ldimy-rdimy) { // Outer j
          for(i=rdimx;i<=ldimx-rdimx;i++) {
            OC_INDEX index = ODTV_VECSIZE*i+jkindex;
            scratch[index] = scratch[index+1] = scratch[index+2] = 0.0;
          }
        } else { // Inner j
          for(i=0;i<ldimx;i++) {
            OC_INDEX index = ODTV_VECSIZE*i+jkindex;
            scratch[index] = scratch[index+1] = scratch[index+2] = 0.0;
          }
        }
      }
    } else { // Middle k
      for(j=0;j<ldimy;j++) {
        OC_INDEX jkindex = kindex + j*sstridey;
        for(i=0;i<ldimx;i++) {
          OC_INDEX index = ODTV_VECSIZE*i+jkindex;
          scratch[index] = scratch[index+1] = scratch[index+2] = 0.0;
        }
      }
    }
  }

#if VERBOSE_DEBUG && !defined(NDEBUG)
  {
    OXS_FFT_REAL_TYPE fft_scaling = -1 *
      fftx.GetScaling() * ffty.GetScaling() * fftz.GetScaling();
    for(k=0;k<ldimz;++k) {
      for(j=0;j<ldimy;++j) {
        for(i=0;i<ldimx;++i) {
          OC_INDEX index = ODTV_VECSIZE*((k*ldimy+j)*ldimx+i);
          printf("A00[%02ld][%02ld][%02ld] = %#25.18Le\n",
                 i,j,k,(long double)(scratch[index]/fft_scaling));
          printf("A01[%02ld][%02ld][%02ld] = %#25.18Le\n",
                 i,j,k,(long double)(scratch[index+1]/fft_scaling));
          printf("A02[%02ld][%02ld][%02ld] = %#25.18Le\n",
                 i,j,k,(long double)(scratch[index+2]/fft_scaling));
        }
      }
    }
    fflush(stdout);
  }
#endif // NDEBUG
#if REPORT_TIME
  dvltimer[1].Stop();
#endif // REPORT_TIME

  // Step 4: Transform into frequency domain.  These lines are cribbed
  // from the corresponding code in Oxs_FFT3DThreeVector.
  // Note: Using an Oxs_FFT3DThreeVector fft object, this would be just
  //    fft.AdjustInputDimensions(ldimx,ldimy,ldimz);
  //    fft.ForwardRealToComplexFFT(scratch,Hxfrm);
  //    fft.AdjustInputDimensions(rdimx,rdimy,rdimz); // Safety
  {
#if REPORT_TIME
  dvltimer[2].Start();
#endif // REPORT_TIME
    fftx.AdjustInputDimensions(ldimx,ldimy);
    ffty.AdjustInputDimensions(ldimy,
                               ODTV_COMPLEXSIZE*ODTV_VECSIZE*cdimx,
                               ODTV_VECSIZE*cdimx);
    fftz.AdjustInputDimensions(ldimz,
                               ODTV_COMPLEXSIZE*ODTV_VECSIZE*cdimx*cdimy,
                               ODTV_VECSIZE*cdimx*cdimy);

    OC_INDEX rxydim = ODTV_VECSIZE*ldimx*ldimy;
    OC_INDEX cxydim = ODTV_COMPLEXSIZE*ODTV_VECSIZE*cdimx*cdimy;

    for(OC_INDEX m=0;m<ldimz;++m) {
      // x-direction transforms in plane "m"
      fftx.ForwardRealToComplexFFT(scratch+m*rxydim,Hxfrm+m*cxydim);
      
      // y-direction transforms in plane "m"
      ffty.ForwardFFT(Hxfrm+m*cxydim);
    }
    fftz.ForwardFFT(Hxfrm); // z-direction transforms

    fftx.AdjustInputDimensions(rdimx,rdimy);   // Safety
    ffty.AdjustInputDimensions(rdimy,
                               ODTV_COMPLEXSIZE*ODTV_VECSIZE*cdimx,
                               ODTV_VECSIZE*cdimx);
    fftz.AdjustInputDimensions(rdimz,
                               ODTV_COMPLEXSIZE*ODTV_VECSIZE*cdimx*cdimy,
                               ODTV_VECSIZE*cdimx*cdimy);

#if REPORT_TIME
  dvltimer[2].Stop();
#endif // REPORT_TIME
  }

  // Copy results from scratch into A00, A01, and A02.  We only need
  // store 1/8th of the results because of symmetries.
#if REPORT_TIME
  dvltimer[3].Start();
#endif // REPORT_TIME
  OC_INDEX astridey = adimx;
  OC_INDEX astridez = astridey*adimy;
  OC_INDEX a_size = astridez*adimz;
  
  assert(0 == G);
  G = new FD_TYPE[6*a_size];
  OC_INDEX cstridey = 2*ODTV_VECSIZE*cdimx; // "2" for complex data
  OC_INDEX cstridez = cstridey*cdimy;
  for(k=0;k<adimz;k++) for(j=0;j<adimy;j++) for(i=0;i<adimx;i++) {
    OC_INDEX aindex = i+j*astridey+k*astridez;
    OC_INDEX hindex = 2*ODTV_VECSIZE*i+j*cstridey+k*cstridez;
    G[0*a_size + aindex] = Hxfrm[hindex];   // A00
    G[1*a_size + aindex] = Hxfrm[hindex+2]; // A01
    G[2*a_size + aindex] = Hxfrm[hindex+4]; // A02
    // The G## values are all real-valued, so we only need to pull the
    // real parts out of Hxfrm, which are stored in the even offsets.
  }
  
#if REPORT_TIME
  dvltimer[3].Stop();
#endif // REPORT_TIME

#if 0 // TRANSFORM CHECK
  {
    fftx.AdjustInputDimensions(ldimx,ldimy);
    ffty.AdjustInputDimensions(ldimy,
                               ODTV_COMPLEXSIZE*ODTV_VECSIZE*cdimx,
                               ODTV_VECSIZE*cdimx);
    fftz.AdjustInputDimensions(ldimz,
                               ODTV_COMPLEXSIZE*ODTV_VECSIZE*cdimx*cdimy,
                               ODTV_VECSIZE*cdimx*cdimy);

    OC_INDEX rxydim = ODTV_VECSIZE*ldimx*ldimy;
    OC_INDEX cxydim = ODTV_COMPLEXSIZE*ODTV_VECSIZE*cdimx*cdimy;

    fftz.InverseFFT(Hxfrm); // z-direction transforms

    for(OC_INDEX m=0;m<ldimz;++m) {
      // y-direction transforms in plane "m"
      ffty.InverseFFT(Hxfrm+m*cxydim);

      // x-direction transforms in plane "m"
      fftx.InverseComplexToRealFFT(Hxfrm+m*cxydim,scratch+m*rxydim);
    }

    fftx.AdjustInputDimensions(rdimx,rdimy);   // Safety
    ffty.AdjustInputDimensions(rdimy,
                               ODTV_COMPLEXSIZE*ODTV_VECSIZE*cdimx,
                               ODTV_VECSIZE*cdimx);
    fftz.AdjustInputDimensions(rdimz,
                               ODTV_COMPLEXSIZE*ODTV_VECSIZE*cdimx*cdimy,
                               ODTV_VECSIZE*cdimx*cdimy);
  }
#endif

  // Repeat for Nyy, Nyz and Nzz. //////////////////////////////////////

#if REPORT_TIME
  dvltimer[4].Start();
#endif // REPORT_TIME
  if(!xperiodic && !yperiodic && !zperiodic) {
    // Step 1: Evaluate f & g at each cell site.  Offset by (-dx,-dy,-dz)
    //  so we can do 2nd derivative operations "in-place".
    for(k=0;k<kstop;k++) {
      OC_INDEX kindex = k*sstridez;
      OC_REALWIDE z = dz*(k-1);
      for(j=0;j<jstop;j++) {
        OC_INDEX jkindex = kindex + j*sstridey;
        OC_REALWIDE y = dy*(j-1);
        for(i=0;i<istop;i++) {
          OC_INDEX index = ODTV_VECSIZE*i+jkindex;
          OC_REALWIDE x = dx*(i-1);
          // Nyy(x,y,z) = Nxx(y,x,z);  Nzz(x,y,z) = Nxx(z,y,x);
          // Nxz(x,y,z) = Nxy(x,z,y);  Nyz(x,y,z) = Nxy(y,z,x);
          scratch[index]   = scale*Oxs_Newell_f(y,x,z);  // For Nyy
          scratch[index+1] = scale*Oxs_Newell_g(y,z,x);  // For Nyz
          scratch[index+2] = scale*Oxs_Newell_f(z,y,x);  // For Nzz
        }
      }
    }

    // Step 2a: Do d^2/dz^2
    if(kstop==1) {
      // Only 1 layer in z-direction of f/g stored in scratch array.
      for(j=0;j<jstop;j++) {
        OC_INDEX jkindex = j*sstridey;
        OC_REALWIDE y = dy*(j-1);
        for(i=0;i<istop;i++) {
          OC_INDEX index = ODTV_VECSIZE*i+jkindex;
          OC_REALWIDE x = dx*(i-1);
          // Function f is even in each variable, so for example
          //    f(x,y,-dz) - 2f(x,y,0) + f(x,y,dz)
          //        =  2( f(x,y,-dz) - f(x,y,0) )
          // Function g(x,y,z) is even in z and odd in x and y,
          // so for example
          //    g(x,-dz,y) - 2g(x,0,y) + g(x,dz,y)
          //        =  2g(x,0,y) = 0.
          // Nyy(x,y,z) = Nxx(y,x,z);  Nzz(x,y,z) = Nxx(z,y,x);
          // Nxz(x,y,z) = Nxy(x,z,y);  Nyz(x,y,z) = Nxy(y,z,x);
          scratch[index]   -= scale*Oxs_Newell_f(y,x,0);  // For Nyy
          scratch[index]   *= 2;
          scratch[index+1]  = 0.0;                    // For Nyz
          scratch[index+2] -= scale*Oxs_Newell_f(0,y,x);  // For Nzz
          scratch[index+2] *= 2;
        }
      }
    } else {
      for(k=0;k<rdimz;k++) {
        OC_INDEX kindex = k*sstridez;
        for(j=0;j<jstop;j++) {
          OC_INDEX jkindex = kindex + j*sstridey;
          for(i=0;i<istop;i++) {
            OC_INDEX index = ODTV_VECSIZE*i+jkindex;
            scratch[index]   += -2*scratch[index+sstridez]
              + scratch[index+2*sstridez];
            scratch[index+1] += -2*scratch[index+sstridez+1]
              + scratch[index+2*sstridez+1];
            scratch[index+2] += -2*scratch[index+sstridez+2]
              + scratch[index+2*sstridez+2];
          }
        }
      }
    }
    // Step 2b: Do d^2/dy^2
    if(jstop==1) {
      // Only 1 layer in y-direction of f/g stored in scratch array.
      for(k=0;k<rdimz;k++) {
        OC_INDEX kindex = k*sstridez;
        OC_REALWIDE z = dz*k;
        for(i=0;i<istop;i++) {
          OC_INDEX index = ODTV_VECSIZE*i+kindex;
          OC_REALWIDE x = dx*(i-1);
          // Function f is even in each variable, so for example
          //    f(x,y,-dz) - 2f(x,y,0) + f(x,y,dz)
          //        =  2( f(x,y,-dz) - f(x,y,0) )
          // Function g(x,y,z) is even in z and odd in x and y,
          // so for example
          //    g(x,-dz,y) - 2g(x,0,y) + g(x,dz,y)
          //        =  2g(x,0,y) = 0.
          // Nyy(x,y,z) = Nxx(y,x,z);  Nzz(x,y,z) = Nxx(z,y,x);
          // Nxz(x,y,z) = Nxy(x,z,y);  Nyz(x,y,z) = Nxy(y,z,x);
          scratch[index]   -= scale * 
            ((Oxs_Newell_f(0,x,z-dz)+Oxs_Newell_f(0,x,z+dz))
             -2*Oxs_Newell_f(0,x,z));
          scratch[index]   *= 2;   // For Nyy
          scratch[index+1] = 0.0;  // For Nyz
          scratch[index+2] -= scale * 
            ((Oxs_Newell_f(z-dz,0,x)+Oxs_Newell_f(z+dz,0,x))
             -2*Oxs_Newell_f(z,0,x));
          scratch[index+2] *= 2;   // For Nzz
        }
      }
    } else {
      for(k=0;k<rdimz;k++) {
        OC_INDEX kindex = k*sstridez;
        for(j=0;j<rdimy;j++) {
          OC_INDEX jkindex = kindex + j*sstridey;
          for(i=0;i<istop;i++) {
            OC_INDEX index = ODTV_VECSIZE*i+jkindex;
            scratch[index]   += -2*scratch[index+sstridey]
              + scratch[index+2*sstridey];
            scratch[index+1] += -2*scratch[index+sstridey+1]
              + scratch[index+2*sstridey+1];
            scratch[index+2] += -2*scratch[index+sstridey+2]
              + scratch[index+2*sstridey+2];
          }
        }
      }
    }
    // Step 2c: Do d^2/dx^2
    if(istop==1) {
      // Only 1 layer in x-direction of f/g stored in scratch array.
      for(k=0;k<rdimz;k++) {
        OC_INDEX kindex = k*sstridez;
        OC_REALWIDE z = dz*k;
        for(j=0;j<rdimy;j++) {
          OC_INDEX index = kindex + j*sstridey;
          OC_REALWIDE y = dy*j;
          // Function f is even in each variable, so for example
          //    f(x,y,-dz) - 2f(x,y,0) + f(x,y,dz)
          //        =  2( f(x,y,-dz) - f(x,y,0) )
          // Function g(x,y,z) is even in z and odd in x and y,
          // so for example
          //    g(x,-dz,y) - 2g(x,0,y) + g(x,dz,y)
          //        =  2g(x,0,y) = 0.
          // Nyy(x,y,z) = Nxx(y,x,z);  Nzz(x,y,z) = Nxx(z,y,x);
          // Nxz(x,y,z) = Nxy(x,z,y);  Nyz(x,y,z) = Nxy(y,z,x);
          scratch[index]   -= scale * 
            ((4*Oxs_Newell_f(y,0,z)
              +Oxs_Newell_f(y+dy,0,z+dz)+Oxs_Newell_f(y-dy,0,z+dz)
              +Oxs_Newell_f(y+dy,0,z-dz)+Oxs_Newell_f(y-dy,0,z-dz))
             -2*(Oxs_Newell_f(y+dy,0,z)+Oxs_Newell_f(y-dy,0,z)
                 +Oxs_Newell_f(y,0,z+dz)+Oxs_Newell_f(y,0,z-dz)));
          scratch[index]   *= 2;  // For Nyy
          scratch[index+1] -= scale * 
            ((4*Oxs_Newell_g(y,z,0)
              +Oxs_Newell_g(y+dy,z+dz,0)+Oxs_Newell_g(y-dy,z+dz,0)
              +Oxs_Newell_g(y+dy,z-dz,0)+Oxs_Newell_g(y-dy,z-dz,0))
             -2*(Oxs_Newell_g(y+dy,z,0)+Oxs_Newell_g(y-dy,z,0)
                 +Oxs_Newell_g(y,z+dz,0)+Oxs_Newell_g(y,z-dz,0)));
          scratch[index+1] *= 2;  // For Nyz
          scratch[index+2] -= scale * 
            ((4*Oxs_Newell_f(z,y,0)
              +Oxs_Newell_f(z+dz,y+dy,0)+Oxs_Newell_f(z+dz,y-dy,0)
              +Oxs_Newell_f(z-dz,y+dy,0)+Oxs_Newell_f(z-dz,y-dy,0))
             -2*(Oxs_Newell_f(z,y+dy,0)+Oxs_Newell_f(z,y-dy,0)
                 +Oxs_Newell_f(z+dz,y,0)+Oxs_Newell_f(z-dz,y,0)));
          scratch[index+2] *= 2;  // For Nzz
        }
      }
    } else {
      for(k=0;k<rdimz;k++) {
        OC_INDEX kindex = k*sstridez;
        for(j=0;j<rdimy;j++) {
          OC_INDEX jkindex = kindex + j*sstridey;
          for(i=0;i<rdimx;i++) {
            OC_INDEX index = ODTV_VECSIZE*i+jkindex;
            scratch[index]   += -2*scratch[index+  ODTV_VECSIZE]
              + scratch[index+2*ODTV_VECSIZE];
            scratch[index+1] += -2*scratch[index+  ODTV_VECSIZE+1]
              + scratch[index+2*ODTV_VECSIZE+1];
            scratch[index+2] += -2*scratch[index+  ODTV_VECSIZE+2]
              + scratch[index+2*ODTV_VECSIZE+2];
          }
        }
      }
    }

    // Special "SelfDemag" code may be more accurate at index 0,0,0.
    const OXS_FFT_REAL_TYPE selfscale
      = -1 * fftx.GetScaling() * ffty.GetScaling() * fftz.GetScaling();
    scratch[0] = Oxs_SelfDemagNy(dx,dy,dz);
    if(zero_self_demag) scratch[0] -= 1./3.;
    scratch[0] *= selfscale;

    scratch[1] = 0.0;  // Nyz[0] = 0.

    scratch[2] = Oxs_SelfDemagNz(dx,dy,dz);
    if(zero_self_demag) scratch[2] -= 1./3.;
    scratch[2] *= selfscale;

    // Step 2.5: Use asymptotic (dipolar + higher) approximation for far field
    /*   Dipole approximation:
     *
     *                        / 3x^2-R^2   3xy       3xz    \
     *             dx.dy.dz   |                             |
     *  H_demag = ----------- |   3xy   3y^2-R^2     3yz    |
     *             4.pi.R^5   |                             |
     *                        \   3xz      3yz     3z^2-R^2 /
     */
    // See Notes IV, 26-Feb-2007, p102.
    if(scaled_arad>=0.0) {
      // Note that all distances here are in "reduced" units,
      // scaled so that dx, dy, and dz are either small integers
      // or else close to 1.0.
      OXS_DEMAG_REAL_ASYMP scaled_arad_sq = scaled_arad*scaled_arad;
      OXS_FFT_REAL_TYPE fft_scaling = -1 *
        fftx.GetScaling() * ffty.GetScaling() * fftz.GetScaling();
      /// Note: Since H = -N*M, and by convention with the rest of this
      /// code, we store "-N" instead of "N" so we don't have to multiply
      /// the output from the FFT + iFFT by -1 in GetEnergy() below.

      OXS_DEMAG_REAL_ASYMP xtest
        = static_cast<OXS_DEMAG_REAL_ASYMP>(rdimx)*dx;
      xtest *= xtest;

      Oxs_DemagNyyAsymptotic ANyy(dx,dy,dz);
      Oxs_DemagNyzAsymptotic ANyz(dx,dy,dz);
      Oxs_DemagNzzAsymptotic ANzz(dx,dy,dz);

      for(k=0;k<rdimz;++k) {
        OC_INDEX kindex = k*sstridez;
        OXS_DEMAG_REAL_ASYMP z = dz*k;
        OXS_DEMAG_REAL_ASYMP zsq = z*z;
        for(j=0;j<rdimy;++j) {
          OC_INDEX jkindex = kindex + j*sstridey;
          OXS_DEMAG_REAL_ASYMP y = dy*j;
          OXS_DEMAG_REAL_ASYMP ysq = y*y;

          OC_INDEX istart = 0;
          OXS_DEMAG_REAL_ASYMP test = scaled_arad_sq-ysq-zsq;
          if(test>0) {
            if(test>xtest) {
              istart = rdimx+1;
            } else {
              istart = static_cast<OC_INDEX>(Oc_Ceil(Oc_Sqrt(test)/dx));
            }
          }
          for(i=istart;i<rdimx;++i) {
            OC_INDEX index = ODTV_VECSIZE*i+jkindex;
            OXS_DEMAG_REAL_ASYMP x = dx*i;
            scratch[index]   = fft_scaling*ANyy.NyyAsymptotic(x,y,z);
            scratch[index+1] = fft_scaling*ANyz.NyzAsymptotic(x,y,z);
            scratch[index+2] = fft_scaling*ANzz.NzzAsymptotic(x,y,z);
          }
        }
      }
    }
  }

  // Step 2.6: If periodic boundaries selected, compute periodic tensors
  // instead.
  // NOTE THAT CODE BELOW ONLY SUPPORTS 1D PERIODIC!!!
  // NOTE 2: Keep this code in sync with that in
  //         GPU_Demag::IncrementPreconditioner
  if(xperiodic) {
    OXS_FFT_REAL_TYPE fft_scaling = -1 *
      fftx.GetScaling() * ffty.GetScaling() * fftz.GetScaling();
    /// Note: Since H = -N*M, and by convention with the rest of this
    /// code, we store "-N" instead of "N" so we don't have to multiply
    /// the output from the FFT + iFFT by -1 in GetEnergy() below.

    Oxs_DemagPeriodicX pbctensor(dx,dy,dz,rdimx*dx,scaled_arad);

    for(k=0;k<rdimz;++k) {
      OC_INDEX kindex = k*sstridez;
      OXS_DEMAG_REAL_ASYMP z = dz*k;
      for(j=0;j<rdimy;++j) {
        OC_INDEX jkindex = kindex + j*sstridey;
        OXS_DEMAG_REAL_ASYMP y = dy*j;
        OXS_DEMAG_REAL_ASYMP Nyy, Nyz, Nzz;

        i=0;
        OXS_DEMAG_REAL_ASYMP x = dx*i;
        OC_INDEX index = ODTV_VECSIZE*i+jkindex;
        pbctensor.NyyNyzNzz(x,y,z,Nyy,Nyz,Nzz);
        scratch[index]   = fft_scaling*Nyy;
        scratch[index+1] = fft_scaling*Nyz;
        scratch[index+2] = fft_scaling*Nzz;

        for(i=1;2*i<rdimx;++i) {
          // Interior i; reflect results from left to right half
          x = dx*i;
          index = ODTV_VECSIZE*i+jkindex;
          OC_INDEX rindex = ODTV_VECSIZE*(rdimx-i)+jkindex;
          pbctensor.NyyNyzNzz(x,y,z,Nyy,Nyz,Nzz);
          scratch[index]   = fft_scaling*Nyy;  // pbctensor computation
          scratch[index+1] = fft_scaling*Nyz;  // *includes* base window
          scratch[index+2] = fft_scaling*Nzz;  // term
          scratch[rindex]   = scratch[index];   // Nyy is even
          scratch[rindex+1] = scratch[index+1]; // Nyz is even wrt x
          scratch[rindex+2] = scratch[index+2]; // Nzz is even
        }

        if(rdimx%2 == 0) { // Do midpoint seperately
          i = rdimx/2;
          x = dx*i;
          index = ODTV_VECSIZE*i+jkindex;
          pbctensor.NyyNyzNzz(x,y,z,Nyy,Nyz,Nzz);
          scratch[index]   = fft_scaling*Nyy;
          scratch[index+1] = fft_scaling*Nyz;
          scratch[index+2] = fft_scaling*Nzz;
        }

      }
    }
  }
  if(yperiodic) {
    OXS_FFT_REAL_TYPE fft_scaling = -1 *
      fftx.GetScaling() * ffty.GetScaling() * fftz.GetScaling();
    /// Note: Since H = -N*M, and by convention with the rest of this
    /// code, we store "-N" instead of "N" so we don't have to multiply
    /// the output from the FFT + iFFT by -1 in GetEnergy() below.

    Oxs_DemagPeriodicY pbctensor(dx,dy,dz,rdimy*dy,scaled_arad);

    for(k=0;k<rdimz;++k) {
      OC_INDEX kindex = k*sstridez;
      OXS_DEMAG_REAL_ASYMP z = dz*k;
      for(j=0;j<=rdimy/2;++j) {
        OC_INDEX jkindex = kindex + j*sstridey;
        OXS_DEMAG_REAL_ASYMP y = dy*j;
        if(0<j && 2*j<rdimy) {
          // Interior j; reflect results from lower to upper half
          OC_INDEX rjkindex = kindex + (rdimy-j)*sstridey;
          for(i=0;i<rdimx;++i) {
            OC_INDEX index = ODTV_VECSIZE*i+jkindex;
            OC_INDEX rindex = ODTV_VECSIZE*i+rjkindex;
            OXS_DEMAG_REAL_ASYMP x = dx*i;
            OXS_DEMAG_REAL_ASYMP Nyy, Nyz, Nzz;
            pbctensor.NyyNyzNzz(x,y,z,Nyy,Nyz,Nzz);
            scratch[index]   = fft_scaling*Nyy;
            scratch[index+1] = fft_scaling*Nyz;
            scratch[index+2] = fft_scaling*Nzz;
            scratch[rindex]   =    scratch[index];   // Nyy is even
            scratch[rindex+1] = -1*scratch[index+1]; // Nyz is odd wrt y
            scratch[rindex+2] =    scratch[index+2]; // Nzz is even
          }
        } else { // j==0 or midpoint
          for(i=0;i<rdimx;++i) {
            OC_INDEX index = ODTV_VECSIZE*i+jkindex;
            OXS_DEMAG_REAL_ASYMP x = dx*i;
            OXS_DEMAG_REAL_ASYMP Nyy, Nyz, Nzz;
            pbctensor.NyyNyzNzz(x,y,z,Nyy,Nyz,Nzz);
            scratch[index]   = fft_scaling*Nyy;
            scratch[index+1] = fft_scaling*Nyz;
            scratch[index+2] = fft_scaling*Nzz;
          }
        }
      }
    }
  }
  if(zperiodic) {
    OXS_FFT_REAL_TYPE fft_scaling = -1 *
      fftx.GetScaling() * ffty.GetScaling() * fftz.GetScaling();
    /// Note: Since H = -N*M, and by convention with the rest of this
    /// code, we store "-N" instead of "N" so we don't have to multiply
    /// the output from the FFT + iFFT by -1 in GetEnergy() below.

    Oxs_DemagPeriodicZ pbctensor(dx,dy,dz,rdimz*dz,scaled_arad);

    for(k=0;k<=rdimz/2;++k) {
      OC_INDEX kindex = k*sstridez;
      OXS_DEMAG_REAL_ASYMP z = dz*k;
      if(0<k && 2*k<rdimz) {
        // Interior k; reflect results from lower to upper half
        OC_INDEX rkindex = (rdimz-k)*sstridez;
        for(j=0;j<rdimy;++j) {
          OC_INDEX jkindex = kindex + j*sstridey;
          OC_INDEX rjkindex = rkindex + j*sstridey;
          OXS_DEMAG_REAL_ASYMP y = dy*j;
          for(i=0;i<rdimx;++i) {
            OXS_DEMAG_REAL_ASYMP x = dx*i;
            OC_INDEX index = ODTV_VECSIZE*i+jkindex;
            OC_INDEX rindex = ODTV_VECSIZE*i+rjkindex;
            OXS_DEMAG_REAL_ASYMP Nyy, Nyz, Nzz;
            pbctensor.NyyNyzNzz(x,y,z,Nyy,Nyz,Nzz);
            scratch[index]   = fft_scaling*Nyy;
            scratch[index+1] = fft_scaling*Nyz;
            scratch[index+2] = fft_scaling*Nzz;
            scratch[rindex]   =    scratch[index];   // Nyy is even
            scratch[rindex+1] = -1*scratch[index+1]; // Nyz is odd wrt z
            scratch[rindex+2] =    scratch[index+2]; // Nzz is even
          }
        }
      } else { // k==0 or midpoint
        for(j=0;j<rdimy;++j) {
          OC_INDEX jkindex = kindex + j*sstridey;
          OXS_DEMAG_REAL_ASYMP y = dy*j;
          for(i=0;i<rdimx;++i) {
            OC_INDEX index = ODTV_VECSIZE*i+jkindex;
            OXS_DEMAG_REAL_ASYMP x = dx*i;
            OXS_DEMAG_REAL_ASYMP Nyy, Nyz, Nzz;
            pbctensor.NyyNyzNzz(x,y,z,Nyy,Nyz,Nzz);
            scratch[index]   = fft_scaling*Nyy;
            scratch[index+1] = fft_scaling*Nyz;
            scratch[index+2] = fft_scaling*Nzz;
          }
        }
      }
    }
  }
#if REPORT_TIME
  dvltimer[4].Stop();
#endif // REPORT_TIME

  // Step 3: Use symmetries to reflect into other octants.
  //     Also, at each coordinate plane, set to 0.0 any term
  //     which is odd across that boundary.  It should already
  //     be close to 0, but will likely be slightly off due to
  //     rounding errors.
  // Symmetries: A00, A11, A22 are even in each coordinate
  //             A01 is odd in x and y, even in z.
  //             A02 is odd in x and z, even in y.
  //             A12 is odd in y and z, even in x.
#if REPORT_TIME
  dvltimer[5].Start();
#endif // REPORT_TIME
  for(k=0;k<rdimz;k++) {
    OC_INDEX kindex = k*sstridez;
    for(j=0;j<rdimy;j++) {
      OC_INDEX jkindex = kindex + j*sstridey;
      for(i=0;i<rdimx;i++) {
        OC_INDEX index = ODTV_VECSIZE*i+jkindex;

        if(j==0 || k==0) scratch[index+1] = 0.0;  // A12

        OXS_FFT_REAL_TYPE tmpA11 = scratch[index];
        OXS_FFT_REAL_TYPE tmpA12 = scratch[index+1];
        OXS_FFT_REAL_TYPE tmpA22 = scratch[index+2];
        if(i>0) {
          OC_INDEX tindex = ODTV_VECSIZE*(ldimx-i)+j*sstridey+k*sstridez;
          scratch[tindex]   =     tmpA11;
          scratch[tindex+1] =     tmpA12;
          scratch[tindex+2] =     tmpA22;
        }
        if(j>0) {
          OC_INDEX tindex = ODTV_VECSIZE*i+(ldimy-j)*sstridey+k*sstridez;
          scratch[tindex]   =     tmpA11;
          scratch[tindex+1] =  -1*tmpA12;
          scratch[tindex+2] =     tmpA22;
        }
        if(k>0) {
          OC_INDEX tindex = ODTV_VECSIZE*i+j*sstridey+(ldimz-k)*sstridez;
          scratch[tindex]   =     tmpA11;
          scratch[tindex+1] =  -1*tmpA12;
          scratch[tindex+2] =     tmpA22;
        }
        if(i>0 && j>0) {
          OC_INDEX tindex
            = ODTV_VECSIZE*(ldimx-i) + (ldimy-j)*sstridey + k*sstridez;
          scratch[tindex]   =     tmpA11;
          scratch[tindex+1] =  -1*tmpA12;
          scratch[tindex+2] =     tmpA22;
        }
        if(i>0 && k>0) {
          OC_INDEX tindex
            = ODTV_VECSIZE*(ldimx-i) + j*sstridey + (ldimz-k)*sstridez;
          scratch[tindex]   =     tmpA11;
          scratch[tindex+1] =  -1*tmpA12;
          scratch[tindex+2] =     tmpA22;
        }
        if(j>0 && k>0) {
          OC_INDEX tindex
            = ODTV_VECSIZE*i + (ldimy-j)*sstridey + (ldimz-k)*sstridez;
          scratch[tindex]   =     tmpA11;
          scratch[tindex+1] =     tmpA12;
          scratch[tindex+2] =     tmpA22;
        }
        if(i>0 && j>0 && k>0) {
          OC_INDEX tindex
            = ODTV_VECSIZE*(ldimx-i) + (ldimy-j)*sstridey + (ldimz-k)*sstridez;
          scratch[tindex]   =     tmpA11;
          scratch[tindex+1] =     tmpA12;
          scratch[tindex+2] =     tmpA22;
        }
      }
    }
  }

  // Step 3.5: Fill in zero-padded overhang
  for(k=0;k<ldimz;k++) {
    OC_INDEX kindex = k*sstridez;
    if(k<rdimz || k>ldimz-rdimz) { // Outer k
      for(j=0;j<ldimy;j++) {
        OC_INDEX jkindex = kindex + j*sstridey;
        if(j<rdimy || j>ldimy-rdimy) { // Outer j
          for(i=rdimx;i<=ldimx-rdimx;i++) {
            OC_INDEX index = ODTV_VECSIZE*i+jkindex;
            scratch[index] = scratch[index+1] = scratch[index+2] = 0.0;
          }
        } else { // Inner j
          for(i=0;i<ldimx;i++) {
            OC_INDEX index = ODTV_VECSIZE*i+jkindex;
            scratch[index] = scratch[index+1] = scratch[index+2] = 0.0;
          }
        }
      }
    } else { // Middle k
      for(j=0;j<ldimy;j++) {
        OC_INDEX jkindex = kindex + j*sstridey;
        for(i=0;i<ldimx;i++) {
          OC_INDEX index = ODTV_VECSIZE*i+jkindex;
          scratch[index] = scratch[index+1] = scratch[index+2] = 0.0;
        }
      }
    }
  }
#if REPORT_TIME
  dvltimer[5].Stop();
#endif // REPORT_TIME

#if VERBOSE_DEBUG && !defined(NDEBUG)
  {
    OXS_FFT_REAL_TYPE fft_scaling = -1 *
      fftx.GetScaling() * ffty.GetScaling() * fftz.GetScaling();
    for(k=0;k<ldimz;++k) {
      for(j=0;j<ldimy;++j) {
        for(i=0;i<ldimx;++i) {
          OC_INDEX index = ODTV_VECSIZE*((k*ldimy+j)*ldimx+i);
          printf("A11[%02ld][%02ld][%02ld] = %#25.18Le\n",
                 i,j,k,(long double)(scratch[index]/fft_scaling));
          printf("A12[%02ld][%02ld][%02ld] = %#25.18Le\n",
                 i,j,k,(long double)(scratch[index+1]/fft_scaling));
          printf("A22[%02ld][%02ld][%02ld] = %#25.18Le\n",
                 i,j,k,(long double)(scratch[index+2]/fft_scaling));
        }
      }
    }
    fflush(stdout);
  }
#endif // NDEBUG

  // Step 4: Transform into frequency domain.  These lines are cribbed
  // from the corresponding code in Oxs_FFT3DThreeVector.
  // Note: Using an Oxs_FFT3DThreeVector fft object, this would be just
  //    fft.AdjustInputDimensions(ldimx,ldimy,ldimz);
  //    fft.ForwardRealToComplexFFT(scratch,Hxfrm);
  //    fft.AdjustInputDimensions(rdimx,rdimy,rdimz); // Safety
  {
#if REPORT_TIME
  dvltimer[6].Start();
#endif // REPORT_TIME
    fftx.AdjustInputDimensions(ldimx,ldimy);
    ffty.AdjustInputDimensions(ldimy,
                               ODTV_COMPLEXSIZE*ODTV_VECSIZE*cdimx,
                               ODTV_VECSIZE*cdimx);
    fftz.AdjustInputDimensions(ldimz,
                               ODTV_COMPLEXSIZE*ODTV_VECSIZE*cdimx*cdimy,
                               ODTV_VECSIZE*cdimx*cdimy);

    OC_INDEX rxydim = ODTV_VECSIZE*ldimx*ldimy;
    OC_INDEX cxydim = ODTV_COMPLEXSIZE*ODTV_VECSIZE*cdimx*cdimy;

    for(OC_INDEX m=0;m<ldimz;++m) {
      // x-direction transforms in plane "m"
      fftx.ForwardRealToComplexFFT(scratch+m*rxydim,Hxfrm+m*cxydim);
      
      // y-direction transforms in plane "m"
      ffty.ForwardFFT(Hxfrm+m*cxydim);
    }
    fftz.ForwardFFT(Hxfrm); // z-direction transforms

    fftx.AdjustInputDimensions(rdimx,rdimy);   // Safety
    ffty.AdjustInputDimensions(rdimy,
                               ODTV_COMPLEXSIZE*ODTV_VECSIZE*cdimx,
                               ODTV_VECSIZE*cdimx);
    fftz.AdjustInputDimensions(rdimz,
                               ODTV_COMPLEXSIZE*ODTV_VECSIZE*cdimx*cdimy,
                               ODTV_VECSIZE*cdimx*cdimy);

#if REPORT_TIME
  dvltimer[6].Stop();
#endif // REPORT_TIME
  }

  // At this point we no longer need the "scratch" array, so release it.
  delete[] scratch;

  // Copy results from scratch into A11, A12, and A22.  We only need
  // store 1/8th of the results because of symmetries.
#if REPORT_TIME
  dvltimer[7].Start();
#endif // REPORT_TIME
  for(k=0;k<adimz;k++) for(j=0;j<adimy;j++) for(i=0;i<adimx;i++) {
    OC_INDEX aindex = i+j*astridey+k*astridez;
    OC_INDEX hindex = 2*ODTV_VECSIZE*i+j*cstridey+k*cstridez;
    G[3*a_size + aindex] = Hxfrm[hindex];   // A00
    G[4*a_size + aindex] = Hxfrm[hindex+2]; // A01
    G[5*a_size + aindex] = Hxfrm[hindex+4]; // A02
    // The G## values are all real-valued, so we only need to pull the
    // real parts out of Hxfrm, which are stored in the even offsets.
  }
  delete[] Hxfrm;
#if REPORT_TIME
  dvltimer[7].Stop();
#endif // REPORT_TIME

  // ********Build cufft plans, kernel size, kernelConfig*************
  int fsize_inv[3];
  fsize_inv[0] = fdimz, fsize_inv[1] = fdimy, fsize_inv[2] = fdimx;
  
  if (cufftCreate(&plan_fwd) != CUFFT_SUCCESS) {
    string msg("error when building create plan_fwd on GPU,  try to reduce problem size...\n");
    throw Oxs_ExtError(this, msg.c_str());  
  }
  
  if (cufftCreate(&plan_bwd) != CUFFT_SUCCESS) {
    string msg("error when building create plan_bwd on GPU,  try to reduce problem size...\n");
    throw Oxs_ExtError(this, msg.c_str());  
  }
  
  if (cufftSetCompatibilityMode(plan_fwd, CUFFT_COMPATIBILITY_NATIVE) != CUFFT_SUCCESS) {
    string msg("error when cufftSetCompatibilityMode plan_fwd on GPU,  try to reduce problem size...\n");
    throw Oxs_ExtError(this, msg.c_str());  
  }
  
  if (cufftSetCompatibilityMode(plan_bwd, CUFFT_COMPATIBILITY_NATIVE) != CUFFT_SUCCESS) {
    string msg("error when cufftSetCompatibilityMode plan_bwd on GPU,  try to reduce problem size...\n");
    throw Oxs_ExtError(this, msg.c_str());  
  }
  
  if (cufftSetAutoAllocation(plan_fwd, 0) != CUFFT_SUCCESS) {
    string msg("error when cufftSetAutoAllocation plan_fwd on GPU\n");
    throw Oxs_ExtError(this, msg.c_str());  
  }
  
  if (cufftSetAutoAllocation(plan_bwd, 0) != CUFFT_SUCCESS) {
    string msg("error when cufftSetAutoAllocation plan_bwd on GPU\n");
    throw Oxs_ExtError(this, msg.c_str());  
  }
  
  size_t workSize_fwd;
  size_t workSize_bwd;
  if(cufftMakePlanMany(plan_fwd, 3, fsize_inv, NULL, 0, 0, NULL, 0, 0, FWDFFT_METHOD, 3, &workSize_fwd) != CUFFT_SUCCESS){
    string msg("error when cufftMakePlanMany plan_fwd on GPU,  try to reduce problem size...\n");
    throw Oxs_ExtError(this, msg.c_str());  
  }

  if(cufftMakePlanMany(plan_bwd, 3, fsize_inv, NULL, 0, 0, NULL, 0, 0, BWDFFT_METHOD, 3, &workSize_bwd) != CUFFT_SUCCESS){
    string msg("error when cufftMakePlanMany plan_bwd on GPU,  try to reduce problem size...\n");
    throw Oxs_ExtError(this, msg.c_str());  
  }
  
  getFlatKernelSize(rsize, BLK_SIZE, Knl_Grid_rsize, Knl_Blk_rsize);
  getFlatKernelSize(csize, BLK_SIZE, Knl_Grid_csize, Knl_Blk_csize);
  
  cufftPlanCreated = true;

#ifdef ASYNCPY  
  cudaMallocHost( &tmp_spin, 3*rsize*sizeof(FD_TYPE) );
  cudaMallocHost( &tmp_field, 3*rsize*sizeof(FD_TYPE) );
  cudaMallocHost( &tmp_energy, rsize*sizeof(FD_TYPE) );
#else
  tmp_spin  = new FD_TYPE[3*rsize];
  tmp_field = new FD_TYPE[3*rsize];
  tmp_energy = new FD_TYPE[rsize];
#endif
  tmp_Ms = new FD_TYPE[rsize];

    embed_convolution = 0;
    embed_block_size = 0;  // A cry for help...
#if REPORT_TIME
  cudaDeviceSynchronize();
    inittime.Stop();
#endif // REPORT_TIME
#if 1
    printf("FINAL SDA00_count = %ld, SDA01_count = %ld\n",
           (long int)SDA00_count,(long int)SDA01_count);
#endif
}


void GPU_Demag::GPU_GetEnergy 
(const Oxs_SimState& state, Oxs_EnergyData& oed, 
 DEVSTRUCT& dev_struct, unsigned int flag_outputH,
 unsigned int flag_outputE, unsigned int flag_outputSumE,
 const OC_BOOL &flag_accum) const
{
  
  if(mesh_id != state.mesh->Id()) {
#ifdef GPU_TIME
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
#endif
    mesh_id = 0; // Safety

    FillCoefficientArrays(state.mesh, dev_struct);


#ifdef GPU_TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  gputime = fopen ("gputime.txt","a");
  fprintf(gputime, "Execution Time for FillCoefficientArrays: %f ms\n", elapsedTime);
  fclose (gputime);
  
  cudaEventRecord(start, 0);
#endif
  memUpload_device(dev_GreenFunc_k, G, 6 * asize, _dev_num);

  mesh_id = state.mesh->Id();
#ifdef GPU_TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  gputime = fopen ("gputime.txt","a");
  fprintf(gputime, "Execution Time for memcpy G & tmp_Ms: %f ms\n", elapsedTime);
  fclose (gputime);
  
  cudaEventRecord(start, 0);
#endif
  }
  
    
  if(dev_MValue != dev_struct.dev_MValue || !cufftPlanWorkAreaSet) {
    ReInitializeDevMemory(dev_struct);
  }

#ifdef GPU_TIME_ITER
  FILE* cputime;
  LARGE_INTEGER frequency;        // ticks per second
  LARGE_INTEGER t1, t2;           // ticks
  double MyelapsedTime;
  // get ticks per second
  QueryPerformanceFrequency(&frequency);
  // start timer
  QueryPerformanceCounter(&t1);
#endif  

#ifdef GPU_TIME  
  cudaEventRecord(start, 0);
#endif
  
#ifdef GPU_TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  gputime = fopen ("gputime.txt","a");
  fprintf(gputime, "Execution Time for tmp_spin cpy: %f ms\n", elapsedTime);
  fclose (gputime);
  
  cudaEventRecord(start, 0);
#endif 

  // Use supplied buffer space, and reflect that use in oed.
  
#if REPORT_TIME
    adsizetime.Start();
#endif // REPORT_TIME

#if REPORT_TIME
    adsizetime.Stop();
#endif // REPORT_TIME

 #ifdef GPU_TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  gputime = fopen ("gputime.txt","a");
  fprintf(gputime, "Execution Time for unknown adjust size func: %f ms\n", elapsedTime);
  fclose (gputime);
  
  cudaEventRecord(start, 0);
#endif 

#ifdef GPU_TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  gputime = fopen ("gputime.txt","a");
  fprintf(gputime, "Execution Time for memcpy dev_MValue cpy: %f ms\n", elapsedTime);
  fclose (gputime);
  
  cudaEventRecord(start, 0);
#endif 

#if REPORT_TIME
	cudaDeviceSynchronize();
    memsettime.Start();
#endif // REPORT_TIME
  memPurge_device((FD_TYPE*)dev_Mtemp, ODTV_VECSIZE * ODTV_COMPLEXSIZE * csize, _dev_num);
#if REPORT_TIME
	cudaDeviceSynchronize();
    memsettime.Stop();
#endif // REPORT_TIME

#ifdef GPU_TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  gputime = fopen ("gputime.txt","a");
  fprintf(gputime, "Execution Time for my_memset: %f ms\n", elapsedTime);
  fclose (gputime);
  
  cudaEventRecord(start, 0);
#endif 
  // Fill Mtemp with Ms[]*spin[].  The plan is to eventually
  // roll this step into the forward FFT routine.
#if REPORT_TIME
	cudaDeviceSynchronize();
    tmptime.Start();
#endif // REPORT_TIME
  OC_INDEX rdimxy = rdimx * rdimy;
  OC_INDEX fdimxy = fdimx * fdimy;
#if REPORT_TIME
	cudaDeviceSynchronize();
    tmptime.Stop();
#endif // REPORT_TIME
#if REPORT_TIME
	cudaDeviceSynchronize();
    preptime.Start();
#endif // REPORT_TIME
  pad(Knl_Grid_rsize, Knl_Blk_rsize, dev_MValue, (FD_TYPE*)dev_Mtemp,
    dev_Ms, rdimx, rdimxy, rsize, fdimx, fdimxy, fsize);
#if REPORT_TIME
	cudaDeviceSynchronize();
    preptime.Stop();
#endif // REPORT_TIME

#ifdef GPU_TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  gputime = fopen ("gputime.txt","a");
  fprintf(gputime, "Execution Time for Init_Mtemp: %f ms\n", elapsedTime);
  fclose (gputime);
  
  cudaEventRecord(start, 0);
#endif 
  if(!embed_convolution) {//IT IS SUPPOSED THAT embed_convoluation = 0
    // Do not embed convolution inside z-axis FFTs.  Instead,
    // first compute full forward FFT, then do the convolution
    // (really matrix-vector A^*M^ multiply), and then do the
    // full inverse FFT.
    
	
//*****************R2C FFT*****************************
	
    // Calculate FFT of Mtemp
#if REPORT_TIME
	cudaDeviceSynchronize();
    fftforwardtime.Start();
#endif // REPORT_TIME
    // Transform into frequency domain;
    {
      cufftResult_t cufftResult = FWDFFT_EXE ( plan_fwd, (FD_TYPE*)dev_Mtemp , dev_Mtemp);
      if (cufftResult != CUFFT_SUCCESS) {
        String msg = String("cufft error after FWDFFT_EXE in : \"")
          + String(ClassName()) + String(" errorCode: ")
          + my_to_string(cufftResult) + String("\".");
        throw Oxs_ExtError(this,msg.c_str());
      };
    }
#if REPORT_TIME
	cudaDeviceSynchronize();
    fftforwardtime.Stop();
#endif // REPORT_TIME

#ifdef GPU_TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  gputime = fopen ("gputime.txt","a");
  fprintf(gputime, "Execution Time for FWDFFT_EXE: %f ms\n", elapsedTime);
  fclose (gputime);
  
  cudaEventRecord(start, 0);
#endif 
//*****************CONVOLUTION*****************


    // Calculate field components in frequency domain.  Make use of
    // realness and even/odd properties of interaction matrices Axx.
    // Note that in transform space only the x>=0 half-space is
    // stored.
    // Symmetries: A00, A11, A22 are even in each coordinate
    //             A01 is odd in x and y, even in z.
    //             A02 is odd in x and z, even in y.
    //             A12 is odd in y and z, even in x.
    assert(adimx>=cdimx);
    assert(cdimy-adimy<adimy);
    assert(cdimz-adimz<adimz);
#if REPORT_TIME
	cudaDeviceSynchronize();
    convtime.Start();
#endif // REPORT_TIME
	multiplication(Knl_Grid_csize, Knl_Blk_csize, dev_Mtemp, dev_GreenFunc_k, adimx, adimy, adimz,
		cdimx, cdimy, cdimz);//FFTR2C_size[0], FFTR2C_size[1], FFTR2C_size[2]);
#if REPORT_TIME
	cudaDeviceSynchronize();
    convtime.Stop();
#endif // REPORT_TIME

#ifdef GPU_TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  gputime = fopen ("gputime.txt","a");
  fprintf(gputime, "Execution Time for convolution: %f ms\n", elapsedTime);
  fclose (gputime);
  
  cudaEventRecord(start, 0);
#endif 
//*****************COMPLEX TO REAL TRANSFORM***********
//*****************Hsta Adding*************************
#if REPORT_TIME
	cudaDeviceSynchronize();
    fftinversetime.Start();
#endif // REPORT_TIME
    // Transform back into space domain.
    {
      cufftResult_t cufftResult = BWDFFT_EXE( plan_bwd, dev_Mtemp, 
        (FD_TYPE*)dev_Mtemp);
      if (cufftResult != CUFFT_SUCCESS) {
        String msg = String("cufft error after BWDFFT_EXE in : \"")
          + String(ClassName()) + String(" errorCode: ")
          + my_to_string(cufftResult) + String("\".");
        throw Oxs_ExtError(this,msg.c_str());
      };
    }
#if REPORT_TIME
	cudaDeviceSynchronize();
    fftinversetime.Stop();
#endif // REPORT_TIME

#ifdef GPU_TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  gputime = fopen ("gputime.txt","a");
  fprintf(gputime, "Execution Time for BWDFFT_EXE: %f ms\n", elapsedTime);
  fclose (gputime);
  
  cudaEventRecord(start, 0);
#endif 
  } 
  else { 
	assert( embed_convolution == 0);
  } // if(!embed_convolution)

#if REPORT_TIME
	cudaDeviceSynchronize();
    dottime.Start();
#endif // REPORT_TIME
  // Calculate pointwise energy density: -0.5*MU0*<M,H>
  //*******const OXS_FFT_REAL_TYPE emult =  -0.5 * MU0;energy[i] = emult * dot * Ms[i];
#if defined(GPU_CPU_TRANS)
  unpad(Knl_Grid_rsize, Knl_Blk_rsize, dev_Field, (FD_TYPE *)dev_Mtemp, dev_MValue, 
		dev_Ms, dev_Energy, dev_Torque, rdimx, rdimxy, rsize, fdimx, fdimxy, fsize,
		flag_outputH != 0, flag_outputE != 0 || flag_outputSumE != 0, 
    flag_accum, dev_energy_loc, dev_field_loc);
#endif

#if REPORT_TIME
  cudaDeviceSynchronize();
  dottime.Stop();
#endif // REPORT_TIME 

#ifdef GPU_CPU_TRANS
	 oed.energy = oed.energy_buffer;
	 oed.field = oed.field_buffer;
	 Oxs_MeshValue<OC_REAL8m>& energy = *oed.energy_buffer;
	 Oxs_MeshValue<ThreeVector>& field = *oed.field_buffer;
	 energy.AdjustSize(state.mesh);
	 field.AdjustSize(state.mesh);
	 if (flag_outputH) {
		FD_TYPE *tmp_field = new FD_TYPE[rsize*ODTV_VECSIZE];
        memDownload_device(tmp_field, dev_field_loc, rsize * ODTV_VECSIZE, _dev_num);
		for(int m = rsize - 1; m>=0 ; --m) {
		  field[m] = ThreeVector(tmp_field[m], tmp_field[m+rsize], tmp_field[m+2*rsize]);
		}
		if(tmp_field) delete[] tmp_field;
	}
	
	if(flag_outputE) {
		FD_TYPE *tmp_energy = new FD_TYPE[rsize];
        memDownload_device(tmp_energy, dev_energy_loc, rsize, _dev_num);
		for(int i = 0; i < rsize; ++i) {
			energy[i] = tmp_energy[i];
		}
		if(tmp_energy) delete[] tmp_energy;
	}
  
  if (flag_outputSumE) {
    FD_TYPE* &dev_energyVolumeProduct = dev_tmp;
    dotProduct(rsize, BLK_SIZE, dev_energy_loc, 
      dev_volume, dev_energyVolumeProduct);
    FD_TYPE energy_sum = sum_device(rsize, 
      dev_energyVolumeProduct, dev_tmp, DEV_NUM, 
      maxGridSize, maxTotalThreads);
    oed.energy_sum = energy_sum;
  }
#endif 

#ifdef GPU_TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  gputime = fopen ("gputime.txt","a");
  fprintf(gputime, "Execution Time for Field_Energy: %f ms\n", elapsedTime);
  fclose (gputime);
  
  cudaEventRecord(start, 0);
#endif 

#ifdef GPU_TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  gputime = fopen ("gputime.txt","a");
  fprintf(gputime, "Execution Time for memcpy dev_Field & dev_Energy: %f ms\n", elapsedTime);
  fclose (gputime);
  
  cudaEventRecord(start, 0);
#endif

#if REPORT_TIME
    prectime.Start();
#endif // REPORT_TIME

 #if REPORT_TIME
    prectime.Stop();
#endif // REPORT_TIME 
#ifdef GPU_TIME
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  gputime = fopen ("gputime.txt","a");
  fprintf(gputime, "Execution Time for final result cpy: %f ms\n", elapsedTime);
  fclose (gputime);
  cudaEventRecord(start, 0);
//  exit(0);
#endif

#ifdef GPU_TIME_ITER
  // stop timer
  QueryPerformanceCounter(&t2);

  // compute and print the elapsed time in millisec
  MyelapsedTime = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
  cputime = fopen ("gputime.txt","a");
  fprintf(cputime, "Execution Time for GPU Demag: %f ms\n\n", MyelapsedTime);
  fclose (cputime);
#endif
}