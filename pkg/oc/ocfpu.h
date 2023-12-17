/* FILE: ocfpu.h                    -*-Mode: c++-*-
 *
 *	Floating point unit control
 *
 * NOTICE: Plase see the file ../../LICENSE
 *
 * Last modified on: $Date: 2012-09-19 22:54:04 $
 * Last modified by: $Author: donahue $
 *
 */

#ifndef _OCFPU
#define _OCFPU

#ifndef OC_USE_X87
# define OC_USE_X87 0
#endif
#ifndef OC_USE_SSE
# define OC_USE_SSE 0
#endif

#if OC_USE_SSE
#include <xmmintrin.h>  // SSE (version "1")
#endif

#include "autobuf.h"

/* End includes */     /* Optional directive to pimake */

class Oc_FpuControlData {
 public:

  // NOTE: In multi-threaded builds, the following public members are
  //       made thread safe via a single mutex in the ocfpu.cc file.
  void ReadData();  // Fills class data from thread FPU control registers
  void WriteData(); // Sets FPU control registers from class data
  void GetDataString(Oc_AutoBuf& buf);

 private:

#if OC_USE_X87
  OC_UINT2 x87_data;
  void Setx87ControlWord (OC_UINT2 mode);
  OC_UINT2 Getx87ControlWord();
#endif

#if OC_USE_SSE
  OC_UINT2 sse_data;
  void SetSSEControlWord (OC_UINT2 mode);
  OC_UINT2 GetSSEControlWord();
#endif
};

#endif // _OCFPU
