# FILE: options.tcl
#
# A resource file for the OOMMF project.
#
# Configurable options of Oc_Classes may be set in this file via calls 
# to [Oc_Option Add].  
#
# Each block below defines the default value of one configurable option
# in OOMMF (the value which would prevail if this file didn't exist).
# Some blocks include alternatives, commented out.  You may edit this
# file to configure OOMMF to use different option values.  However,
# future OOMMF distributions may overwrite this file.  OOMMF also
# queries the file ./local/options.tcl for configuraton information, if
# it exists, and that file will not be overwritten by future
# distributions of OOMMF.  To make permanent configuration changes to
# OOMMF, copy this file to ./local/options.tcl and edit it.

########################################################################
# The port on which the OOMMF host service directory application
# listens for connections.  The default port number is 15136, but
# this may be overridden with the environment variable OOMMF_HOSTPORT.
global env
if {[info exists env(OOMMF_HOSTPORT)]} {
   Oc_Option Add * Net_Host port $env(OOMMF_HOSTPORT)
} else {
   Oc_Option Add * Net_Host port 15136
}

########################################################################
# Socket access options.  Net_Server setting controls user id checks
# on server sockets, Net_Link setting controls user id checks on
# client sockets.  Set to 0 to disable checks, 2 to force checks, and
# 1 to use checks if possible.
# 
Oc_Option Add * Net_Server checkUserIdentities 0
Oc_Option Add * Net_Link checkUserIdentities 0

########################################################################
# The (partial) Tcl command used to launch a browser for HTML files.
#
# When an OOMMF application needs to display the HTML contents 
# located by an URL, it invokes this command with an additional
# argument: the absolute URL locating the HTML to display.
# By default, this command invokes mmHelp, OOMMF's own HTML
# browsing application.  To use a different HTML browser,
# supply the appropriate partial Tcl command for launching
# the browser of your choice.  (If you choose Microsoft's
# Internet Explorer, see also the following configurable
# option.)
# 
# OOMMF's native HTML browser, mmHelp:
Oc_Option Add * {} htmlBrowserCmd {Oc_Application Exec mmHelp}
#
# Netscape's Navigator/Communicator -- may need full path
#Oc_Option Add * {} htmlBrowserCmd \
#    {exec {C:\Program Files\Netscape\Communicator\Program\netscape.exe}}
#
# Microsoft's Internet Explorer -- may need full path.  Also
# if your release of Internet Explorer is less than 4.0,
# be sure to select MSIE compatibile file URLs below
# with "Oc_Option Add * Oc_Url msieCompatible 1".
#Oc_Option Add * {} htmlBrowserCmd \
#    {exec {C:\Program Files\Internet Explorer/iexplore.exe}}
########################################################################
# Formatting of file: URLs
#
# The file: URLs generated by the Oc_Class Oc_Url follow RFC 1738 and
# RFC 1808 by default, so that they look like 
# file://localhost/path/to/file.html .  That format is not acceptable
# to releases of Microsoft's Internet Explorer before 4.0.  They
# insists on the format: file:/path/to/file.html .  Set the value
# below to 1 to enable compatibility with pre-4.0 releases of Microsoft
# Internet Explorer. 
#
Oc_Option Add * Oc_Url msieCompatible 1
########################################################################
# Base font size for mmHelp
Oc_Option Add mmHelp Font baseSize 16
########################################################################
# Color database for Nb_GetColor
#
# List of Tcl files to source.  Each file should fill the
# nbcolordatabase array with rgb triplets, 8-bits per color.
# File names should be absolute paths.
Oc_Option Add * Color filename [list \
  [file join [file dirname [Oc_DirectPathname [info script]]] colors.config]]
########################################################################
# Input filter programs for mmDisp and mmGraph
#
# The value for each is a list with an even number of elements.  The
# first element of each pair is a list of file name extensions that
# match files containing a certain data format.  The second element of
# each pair is a program that translates from that format into one
# which the OOMMF application can read.
#   The main option here is "decompress", used by the Nb_InputFilter
# class to launch programs like gzip and unzip to decompress files.  The
# ovf and odt options provide support for converting from other data
# formats to OVF (for mmDisp) and ODT (for mmGraph).  These are not
# enabled by default, but are supplied as hooks for end-user or third
# party use.  Files are sent first through the decompression filters, if
# applicable.
#   NOTE: These filters may be  used by a variety of applications (e.g.,
# mmDisp, avf2ppm, avf2ps), so the proper setting for the "app" field
# of the Oc_Option Add command is '*', not mmDisp, mmGraph, or similar.
Oc_Option Add * Nb_InputFilter decompress {{.gz .z .zip} {gzip -dc}}
Oc_Option Add * Nb_InputFilter ovf {}
Oc_Option Add * Nb_InputFilter odt {}
#
# Example extended decompress filter setting:
#   Oc_Option Add * Nb_InputFilter decompress \
#      { {.gz .z} {gzip -dc}   .bz2 bunzip2   .zip funzip }
#
#
########################################################################
# Default curve line widths in mmGraph.  This should be an non-negative
# integer.  Performace on Windows can be improved by setting this to 1.
global tcl_platform
if {[string match windows $tcl_platform(platform)]} {
   Oc_Option Add mmGraph Ow_GraphWin default_curve_width 1
} else {
   Oc_Option Add mmGraph Ow_GraphWin default_curve_width 2
}
#
########################################################################
# Default symbol frequency and size in mmGraph.  These should be
# non-negative integers.  To default to no symbols, set
# default_symbol_freq to 0.  Symbol size is in pixels.
Oc_Option Add mmGraph Ow_GraphWin default_symbol_freq  0
Oc_Option Add mmGraph Ow_GraphWin default_symbol_size 10
#
########################################################################
# Default curve coloring selection method in mmGraph; should be either
# "curve" or "segment".  If "curve", then each curve is a different
# color.  If "segment", then each segment of each curve is a different
# color.
Oc_Option Add mmGraph Ow_GraphWin default_color_selection curve
#
########################################################################
# Default canvas background coloring for mmGraph; in order to be
# consistent with mmGraph's configuration dialog, value should be
# either "white" or "#042" (which is a dark green), but any valid
# color is accepted.
Oc_Option Add mmGraph Ow_GraphWin default_canvas_color white
#
########################################################################
# Icon settings.  owWindowIconType should be one of color, b/w, or none.
# owWindowIconSize should be either 16 or 32.  The default setting of 16
# selects the smaller 16x16 pixel icons; change this to 32 if you want
# the larger 32x32 pixel icons.
# 
Oc_Option Add * {} owWindowIconType color
Oc_Option Add * {} owWindowIconSize 16
if {[string match windows $tcl_platform(platform)]} {
   # There is a Tk bug on Windows that swaps the red and blue color
   # planes in color icons.  Setting this option to 1 tells the OOMMF
   # icon code to work around this bug.  This bug is fixed in Tk 8.4.16.
   foreach {major minor pl} [split [info patchlevel] .] {break}
   if {$major>8 || ($major==8 && $minor>4) || \
          ($major==8 && $minor==4 && $pl>15)} {
      Oc_Option Add * {} owWindowIconSwapRB 0
   } else {
      Oc_Option Add * {} owWindowIconSwapRB 1
   }
} else {
   Oc_Option Add * {} owWindowIconSwapRB 0
}
#
########################################################################
# Control "watch" cursor display in Ow library routines
# Performace on Windows can be improved by setting this to 1.
# Oc_Option Add * {} owDisableWatchCursor 0
#
########################################################################
# Slave MIF interpreter control in Oxs
#
# Value should be one of safe, custom or unsafe
Oc_Option Add * MIFinterp safety custom
#
########################################################################
# Number of threads to run (per process), for thread enabled builds.
# Usually, this is set in the applicable oommf/config/platform/ file,
# but that value may be overridden here.  Additionally, the value
# here may be overridden by the OOMMF_THREADS environment variable,
# or may be set on the command line of thread-aware apps (e.g., oxsii
# and boxsi).
#   In most instances, this line should be left commented out, so that
# the generally more reliable oommf/config/platform setting is retained.
# Oc_Option Add * Threads oommf_thread_count 2
#

########################################################################
# Platform-generic default flags for compiling
# To enable compiler warnings, add '-warn 1'
# To enable debugger support, add '-debug 1'
# To disable some slow runtime error checks, add '-def NDEBUG'
Oc_Option Add * Platform cflags {-def NDEBUG}
#Oc_Option Add * Platform cflags {-warn 1 -debug 1}
#

#cuiwl: nvcc flags might be added here
Oc_Option Add * Platform cuflags {-gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52}

########################################################################
# Platform-generic default flags for linking
Oc_Option Add * Platform lflags {}
########################################################################
# Whether Oc_Classes should enforce the 'const' keyword
Oc_Option Add * Oc_Class enforceConst 0
########################################################################
# Whether Oc_Classes should enforce the 'private' keyword
Oc_Option Add * Oc_Class enforcePrivate 0
########################################################################
########################################################################
# Evalute the file ./local/options.tcl if it exists.  

set fn [file join [file dirname [Oc_DirectPathname [info script]]] local \
        options.tcl]
if {[file readable $fn]} {
    if {[catch {source $fn} msg]} {
        global errorInfo errorCode
	set msg [join [split $msg \n] \n\t]
	error "Error sourcing local options file:\n    $fn:\n\t$msg" \
		$errorInfo $errorCode
    }
}

