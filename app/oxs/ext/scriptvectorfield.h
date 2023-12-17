/* FILE: scriptvectorfield.h      -*-Mode: c++-*-
 *
 * Tcl script vector field object, derived from Oxs_VectorField
 * class.
 *
 */

#ifndef _OXS_SCRIPTVECTORFIELD
#define _OXS_SCRIPTVECTORFIELD

#include <vector>

#include "oc.h"
#include "nb.h"

#include "scalarfield.h"
#include "vectorfield.h"

OC_USE_STD_NAMESPACE;

/* End includes */

class Oxs_ScriptVectorField:public Oxs_VectorField {
private:
  OC_BOOL norm_set;
  OC_REAL8m norm;
  OC_REAL8m multiplier;
  Oxs_ThreeVector basept;
  Oxs_ThreeVector scale;
  Nb_ArrayWrapper< Oxs_OwnedPointer<Oxs_ScalarField> > scalarfields;
  Nb_ArrayWrapper< Oxs_OwnedPointer<Oxs_VectorField> > vectorfields;
  vector<Nb_TclCommandLineOption> command_options;
  Nb_TclCommand cmd;
public:
  virtual const char* ClassName() const; // ClassName() is
  /// automatically generated by the OXS_EXT_REGISTER macro.

  Oxs_ScriptVectorField
  (const char* name,     // Child instance id
   Oxs_Director* newdtr, // App director
   const char* argstr);  // MIF input block parameters

  virtual ~Oxs_ScriptVectorField();

  virtual void Value(const ThreeVector& pt,ThreeVector& value) const;

  virtual void FillMeshValue(const Oxs_Mesh* mesh,
			     Oxs_MeshValue<ThreeVector>& array) const;

};


#endif // _OXS_SCRIPTVECTORFIELD
