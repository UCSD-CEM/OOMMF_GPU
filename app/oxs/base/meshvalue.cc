/* FILE: meshvalue.cc                 -*-Mode: c++-*-
 *
 * Oxs_MeshValue templated class, intended for use with the
 * Oxs_Mesh family.
 *
 * Most of the definitions for this templated class are in
 * the header file meshvalue.h, q.v.  The exception being
 * specializations below, for which we need to arrange only
 * one copy to be instantiated.
 *
 * Incidentally, even without the specializations, this file
 * is needed to make some pre-linkers happy.
 *
 * See the file key.h for a discussion on the reasons for
 * putting all template definitions in the header file.
 */

#include "meshvalue.h"

// Routines for Oxs_MeshValue<OC_REAL8m> output
void Oxs_MeshValueOutputField
(Tcl_Channel channel,    // Output channel
 OC_BOOL do_headers,     // If false, then output only raw data
 const char* title,      // Long filename or title
 const char* desc,       // Description to embed in output file
 vector<String>& valuelabels, // Value label, such as "Exchange energy density"
 vector<String>& valueunits,  // Value units, such as "J/m^3".
 Vf_Ovf20_MeshType meshtype,   // Either rectangular or irregular
 Vf_OvfDataStyle datastyle,   // vf_oascii, vf_obin4, or vf_obin8
 const char* textfmt,  // vf_oascii output only, printf-style format
 const Vf_Ovf20_MeshNodes* mesh,    // Mesh
 const Oxs_MeshValue<OC_REAL8m>* val,  // Scalar array
 Vf_OvfFileVersion ovf_version)
{
  Vf_Ovf20FileHeader fileheader;

  // Fill header
  mesh->DumpGeometry(fileheader,meshtype);
  fileheader.title.Set(String(title));
  fileheader.valuedim.Set(1);  // Scalar field
  fileheader.valuelabels.Set(valuelabels);
  fileheader.valueunits.Set(valueunits);
  fileheader.desc.Set(String(desc));
  fileheader.ovfversion = ovf_version;
  if(!fileheader.IsValid()) {
    OXS_THROW(Oxs_ProgramLogicError,"Oxs_MeshValueOutputField(T=OC_REAL8m)"
              " failed to create a valid OVF fileheader.");
  }

  // Write header
  if(do_headers) fileheader.WriteHeader(channel);

  // Write data
  Vf_Ovf20VecArrayConst data_info(1,val->Size(),val->arr);
  fileheader.WriteData(channel,datastyle,textfmt,mesh,
                       data_info,!do_headers);
}

// For Oxs_MeshValue<ThreeVector> output
void Oxs_MeshValueOutputField
(Tcl_Channel channel,    // Output channel
 OC_BOOL do_headers,        // If false, then output only raw data
 const char* title,      // Long filename or title
 const char* desc,       // Description to embed in output file
 vector<String>& valuelabels, // Value label, such as "Exchange energy density"
 vector<String>& valueunits,  // Value units, such as "J/m^3".
 Vf_Ovf20_MeshType meshtype,   // Either rectangular or irregular
 Vf_OvfDataStyle datastyle,   // vf_oascii, vf_obin4, or vf_obin8
 const char* textfmt,  // vf_oascii output only, printf-style format
 const Vf_Ovf20_MeshNodes* mesh,    // Mesh
 const Oxs_MeshValue<ThreeVector>* val,  // Scalar array
 Vf_OvfFileVersion ovf_version)
{
  Vf_Ovf20FileHeader fileheader;

  // Fill header
  mesh->DumpGeometry(fileheader,meshtype);
  fileheader.title.Set(String(title));
  fileheader.valuedim.Set(3);  // Vector field
  fileheader.valuelabels.Set(valuelabels);
  fileheader.valueunits.Set(valueunits);
  fileheader.desc.Set(String(desc));
  fileheader.ovfversion = ovf_version;
  if(!fileheader.IsValid()) {
    OXS_THROW(Oxs_ProgramLogicError,"Oxs_MeshValueOutputField(T=ThreeVector)"
              " failed to create a valid OVF fileheader.");
  }

  // Write header
  if(do_headers) fileheader.WriteHeader(channel);

  const OC_REAL8m* arrptr = NULL;
  Nb_ArrayWrapper<OC_REAL8m> rbuf;
  if(sizeof(ThreeVector) == 3*sizeof(OC_REAL8m)) {
    // Use MeshValue array directly
    arrptr = reinterpret_cast<const OC_REAL8m*>(val->arr);
  } else {
    // Need intermediate buffer space
    const OC_INDEX valsize = val->Size();
    rbuf.SetSize(3*valsize);
    for(OC_INDEX i=0;i<valsize;++i) {
      rbuf[3*i]   = val->arr[i].x;
      rbuf[3*i+1] = val->arr[i].y;
      rbuf[3*i+2] = val->arr[i].z;
    }
    arrptr = rbuf.GetPtr();
  }


  // Write data
  Vf_Ovf20VecArrayConst data_info(3,val->Size(),arrptr);
  fileheader.WriteData(channel,datastyle,textfmt,mesh,
                       data_info,!do_headers);
}
