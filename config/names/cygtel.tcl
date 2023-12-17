# cygtel.tcl
#
# Defines the Oc_Config name 'cygtel' to indicate the Windows 95/NT operating
# system running on an Intel architecture, using Cygwin Tcl/Tk.
#
# NB: The Oc_IsCygwinPlatform proc is defined in oommf/pkg/oc/procs.tcl.
Oc_Config New _ [string tolower [file rootname [file tail [info script]]]] {
   return [Oc_IsCygwinPlatform]
}
