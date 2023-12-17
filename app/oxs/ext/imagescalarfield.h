/* FILE: imagescalarfield.h      -*-Mode: c++-*-
 *
 * Image scalar field object, derived from Oxs_ScalarField class.
 *
 */

#ifndef _OXS_IMAGESCALARFIELD
#define _OXS_IMAGESCALARFIELD

#include "oc.h"

#include "scalarfield.h"

/* End includes */

class Oxs_ImageScalarField:public Oxs_ScalarField {
public:
  virtual const char* ClassName() const; // ClassName() is
  /// automatically generated by the OXS_EXT_REGISTER macro.

  Oxs_ImageScalarField
  (const char* name,     // Child instance id
   Oxs_Director* newdtr, // App director
   const char* argstr);  // MIF input block parameters

  virtual ~Oxs_ImageScalarField();

  virtual OC_REAL8m Value(const ThreeVector& pt) const;

  virtual void FillMeshValue(const Oxs_Mesh* mesh,
			     Oxs_MeshValue<OC_REAL8m>& array) const
  { DefaultFillMeshValue(mesh,array); }

  virtual void IncrMeshValue(const Oxs_Mesh* mesh,
			     Oxs_MeshValue<OC_REAL8m>& array) const
  { DefaultIncrMeshValue(mesh,array); }

  virtual void MultMeshValue(const Oxs_Mesh* mesh,
			     Oxs_MeshValue<OC_REAL8m>& array) const
  { DefaultMultMeshValue(mesh,array); }

private:
  Oxs_Box bbox;

  enum ViewPlane { xy, zx, yz } view;
  // View plane selection:
  //  Default is "xy", for which
  //   x increases from left to right of image, and
  //   y increases from bottom to top of image.
  //  Selection "zx" specifies
  //   z increases from left to right of image, and
  //   x increases from bottom to top of image.
  //  Selection "yz" specifies
  //   y increases from left to right of image, and
  //   z increases from bottom to top of image.

  enum Exterior_Handling {
    EXTERIOR_INVALID, EXTERIOR_ERROR, EXTERIOR_BOUNDARY, EXTERIOR_DEFAULT
  } exterior;
  OC_REAL8m default_value;

  OC_INT4m image_width,image_height;

  vector<OC_REAL8m> value_array;

  OC_REAL8m column_offset,column_scale;
  OC_REAL8m row_offset,row_scale;
};

#endif // _OXS_IMAGESCALARFIELD
