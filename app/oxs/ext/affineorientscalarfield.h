/* FILE: affineorientscalarfield.h      -*-Mode: c++-*-
 *
 * Scalar field object, derived from Oxs_ScalarField class,
 * that applies an affine orientation (pre-)transformation
 * to another scalar field object.
 *
 */

#ifndef _OXS_AFFINEORIENTSCALARFIELD
#define _OXS_AFFINEORIENTSCALARFIELD

#include <vector>

#include "oc.h"

#include "threevector.h"
#include "util.h"
#include "scalarfield.h"

/* End includes */

class Oxs_AffineOrientScalarField:public Oxs_ScalarField {
private:
  Oxs_ThreeVector offset;
  Oxs_ThreeVector row1,row2,row3;
  Oxs_OwnedPointer<Oxs_ScalarField> field;
  static void InvertMatrix(Oxs_Ext* obj,
			   Oxs_ThreeVector& A1,
			   Oxs_ThreeVector& A2,
			   Oxs_ThreeVector& A3,
			   OC_REAL8m checkslack);
                           /// Ai's are matrix rows
public:
  virtual const char* ClassName() const; // ClassName() is
  /// automatically generated by the OXS_EXT_REGISTER macro.
  Oxs_AffineOrientScalarField
  (const char* name,     // Child instance id
   Oxs_Director* newdtr, // App director
   const char* argstr);  // MIF input block parameters

  virtual ~Oxs_AffineOrientScalarField();

  virtual OC_REAL8m Value(const ThreeVector& pt) const;

  virtual void FillMeshValue(const Oxs_Mesh* mesh,
			     Oxs_MeshValue<OC_REAL8m>& array) const;
  virtual void IncrMeshValue(const Oxs_Mesh* mesh,
			     Oxs_MeshValue<OC_REAL8m>& array) const;
  virtual void MultMeshValue(const Oxs_Mesh* mesh,
			     Oxs_MeshValue<OC_REAL8m>& array) const;

};


#endif // _OXS_AFFINEORIENTSCALARFIELD