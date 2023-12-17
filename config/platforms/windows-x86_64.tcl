# FILE: windows-x86_64.tcl
#
# Configuration feature definitions for the configuration 'windows-x86_64'
#
# Editing instructions begin at "START EDIT HERE" below.

set config [Oc_Config RunPlatform]

set scriptfn [Oc_DirectPathname [info script]]
if {![string match [string tolower [file rootname [file tail $scriptfn]]] \
        [$config GetValue platform_name]]} {
    error "Configuration file '$scriptfn'
sourced by '[$config GetValue platform_name]'"
}

set localfn [file join [file dirname $scriptfn] local \
                [file tail $scriptfn]]
if {[file readable $localfn]} {
    if {[catch {source $localfn} msg]} {
        global errorInfo errorCode
	set msg [join [split $msg \n] \n\t]
	error "Error sourcing local platform file:\n    $localfn:\n\t$msg" \
		$errorInfo $errorCode
    }
}

if {[catch {$config GetValue program_compiler_c++_override}] \
       && ![catch {$config GetValue program_compiler_c++} _]} {
   # If program_compiler_c++ is set, but program_compiler_c++_override
   # is not, then assume user set the former instead of the latter,
   # and so copy the former to the latter to preserve the setting
   # across the setting of program_compiler_c++ in the "REQUIRED
   # CONFIGURATION" section below.
   $config SetValue program_compiler_c++_override $_
}

## Support for the automated buildtest scripts
if {[info exists env(OOMMF_BUILDTEST)] && $env(OOMMF_BUILDTEST)} {
   source [file join [file dirname [info script]] buildtest.tcl]
}


########################################################################
# START EDIT HERE
# In order to properly build, install, and run on your computing
# platform, the OOMMF software must know certain features of your
# computing environment.  In this file are lines which set the value of
# certain features of your computing environment.  Each line looks like:
#
# $config SetValue <feature> {<value>}
#
# where each <feature> is the name of some feature of interest,
# and <value> is the value which is assigned to that feature in a
# description of your computing environment.  Your task is to edit
# the values as necessary to properly describe your computing
# environment.
#
# The character '#' at the beginning of a line is a comment character.
# It causes the contents of that line to be ignored.  To select
# among lines providing alternative values for a feature, uncomment the
# line containing the proper value.
#
# The features in this file are divided into three sections.  The first
# section (REQUIRED CONFIGURATION) includes features which require you
# to provide a value.  The second section (OPTIONAL CONFIGURATION)
# includes features which have usable default values, but which you
# may wish to customize.  The third section (ADVANCED CONFIGURATION)
# contains features which you probably do not need or want to change
# without a good reason.
########################################################################
# REQUIRED CONFIGURATION

# NOTE: The rest of the REQUIRED CONFIGURATION is required only
# for building OOMMF software from source code.  If you downloaded
# a distribution with pre-compiled executables, no more configuration
# is required.
#
# Set the feature 'program_compiler_c++' to the program to run on this
# platform to compile source code files written in the language C++ into
# object files.  Select from the choices below.  If the compiler is not
# in your path, be sure to use the whole pathname.  Also include any
# options required to instruct your compiler to only compile, not link.
#
# If your compiler is not listed below, additional features will
# have to be added in the ADVANCED CONFIGURATION section below to
# describe to the OOMMF software how to operate your compiler.  Send
# e-mail to the OOMMF developers for assistance.
#
# Microsoft Visual C++
# <URL:http://msdn.microsoft.com/visualc/>
$config SetValue program_compiler_c++ {cl /c /openmp}
#$config SetValue program_compiler_c++ {cl /c }
#$config SetValue program_compiler_c++ {cl /c /Zi /FD}  ;for debugging
#

#cuiwl: set nvcc 
$config SetValue program_compiler_cuda {nvcc -c -m 64 -w -Xcompiler "/openmp /nologo /Zi /DNDEBUG /Ox /fp:fast /D_CRT_SECURE_NO_DEPRECATE"}
#$config SetValue program_compiler_cuda {nvcc -c -g}   ;for debugging

# MINGW32 + gcc
#$config SetValue program_compiler_c++ {g++ -c}
#

########################################################################
# SUPPORT PROCEDURES
#
# Load routines to guess the CPU, determine compiler version, and
# provide appropriate cpu-specific and compiler version-specific
# optimization flags.
source [file join [file dirname [Oc_DirectPathname [info script]]]  \
         cpuguess-windows-x86_64.tcl]

# On Windows, child threads get the system default x87 control word,
# which in particular means the floating point precision is set to use a
# 53 bit mantissa ( = 8-byte float)), rather than the 64 bit mantissa (
# = 10-byte float) used by default by the gcc and bcc (and possibly
# other) compilers for the main thread.  In principle (if not in
# practice), the SSE control word could be similarly affected.  The
# following option activates code to copy fpu control registers from the
# parent thread to its children.
$config SetValue \
   program_compiler_c++_property_init_thread_fpu_control_word 1

########################################################################
# LOCAL CONFIGURATION
#
# The following options may be defined in the
# platforms/local/windows-x86_64.tcl file:
#
## Set the feature 'path_directory_temporary' to the name of an existing
## directory on your computer in which OOMMF software should write
## temporary files.  All OOMMF users must have write access to this
## directory.
# $config SetValue path_directory_temporary {C:\temp}
#
## Specify whether or not to build in thread support.
## Thread support is included automatically if the tclsh interpreter used
## during the build process is threaded.  If you have a thread enabled
## tclsh, but don't want oommf_threads, override here.
# $config SetValue oommf_threads 0  ;# 1 to force threaded build,
#                                   ## 0 to force non-threaded build.
#
## Specify the number of default threads.  This is only meaningful
## for builds with thread support.
# $config SetValue thread_count 4  ;# Replace '4' with desired thread count.
#
## Use SSE intrinsics?  If so, specify level here.  Set to 0 to not use
## SSE intrinsics.  Leave unset to get the default (which may depend
## on the selected compiler).
# $config SetValue sse_level 2  ;# Replace '2' with desired level
#
## Override default C++ compiler.  Note the "_override" suffix
## on the value name.
# $config SetValue program_compiler_c++_override {g++ -c}
#
## Processor architecture for compiling.  The default is "generic"
## which should produce an executable that runs on any cpu model for
## the given platform.  Optionally, one may specify "host", in which
## case the build scripts will try to automatically detect the
## processor type on the current system, and select compiler options
## specific to that processor model.  The resulting binary will
## generally not run on other architectures.
# $config SetValue program_compiler_c++_cpu_arch host
#
## Variable type used for array indices, OC_INDEX.  This is a signed
## type which by default is sized to match the pointer width.  You can
## force the type by setting the following option.  The value should
## be a three item list, where the first item is the name of the
## desired (signed) type, the second item is the name of the
## corresponding unsigned type, and the third is the width of these
## types, in bytes.  It is assumed that both the signed and unsigned
## types are the same width, as otherwise significant code breakage is
## expected.  Example:
# $config SetValue program_compiler_c++_oc_index_type {__int64 {unsigned __int64} 8}
#
## For OC_INDEX type checks.  If set to 1, then various segments in
## the code are activated which will detect some array index type
## mismatches at link time.  These tests are not comprehensive, and
## will probably break most third party code, but may be useful during
## development testing.
# $config SetValue program_compiler_c++_oc_index_checks 1
#
## Flags to add to compiler "opts" string:
# $config SetValue program_compiler_c++_add_flags \
#                          {-funroll-loops}
#
## Flags to remove from compiler "opts" string:
# $config SetValue program_compiler_c++_remove_flags \
#                          {-fomit-frame-pointer -fprefetch-loop-arrays}
#
###################
# Default handling of local defaults:
#
if {[catch {$config GetValue oommf_threads}]} {
   # Value not set in platforms/local/windows-x86_64.tcl,
   # so use Tcl setting.
   global tcl_platform
   if {[info exists tcl_platform(threaded)] \
          && $tcl_platform(threaded)} {
      $config SetValue oommf_threads 1  ;# Yes threads
   } else {
      $config SetValue oommf_threads 0  ;# No threads
   }
}
$config SetValue thread_count_auto_max 4 ;# Arbitrarily limit
## maximum number of "auto" threads to 4.
if {[catch {$config GetValue thread_count}]} {
   # Value not set in platforms/local/windows-x86_64.tcl, so try
   # to get value from environment:
   if {[info exists env(NUMBER_OF_PROCESSORS)]} {
      set processor_count $env(NUMBER_OF_PROCESSORS)
      set auto_max [$config GetValue thread_count_auto_max]
      if {$processor_count>$auto_max} {
         # Limit automatically set thread count to auto_max
         set processor_count $auto_max
      }
      $config SetValue thread_count $processor_count
   }
}

if {[catch {$config GetValue program_compiler_c++_override} compiler] == 0} {
    $config SetValue program_compiler_c++ $compiler
}

########################################################################
# ADVANCED CONFIGURATION

# Compiler option processing...
if {[catch {$config GetValue program_compiler_c++} ccbasename]} {
   set ccbasename {}  ;# C++ compiler not selected
} else {
   set ccbasename [file tail [lindex $ccbasename 0]]
}

#cuiwl: for nvcc
if {[catch {$config GetValue program_compiler_cuda} cubasename]} {
	set cubasename {}
} else {
	set cubasename [file tail [lindex $cubasename 0]]
}

# Microsoft Visual C++ compiler
if {[string match cl $ccbasename]} {
   set compilestr [$config GetValue program_compiler_c++]
   if {![info exists cl_version]} {
      set cl_version [GuessClVersion [lindex $compilestr 0]]
   }
   $config SetValue program_compiler_c++_banner_cmd \
      [list GetClBannerVersion  [lindex $compilestr 0]]
   lappend compilestr /nologo /GR ;# /GR turns on RTTI
   set cl_major_version [lindex $cl_version 0]

   if {[lindex $cl_major_version 0]>7} {
      # The exception handling specification switch "/GX"
      # is deprecated in version 8.  /EHa enables C++
      # exceptions with SEH exceptions, /EHs enables C++
      # exceptions without SEH exceptions, and /EHc sets
      # extern "C" to default to nothrow.
      lappend compilestr /EHac
   } else {
      lappend compilestr /GX
   }
   $config SetValue program_compiler_c++ $compilestr
   unset compilestr

   # Optimization options for Microsoft Visual C++
   #
   # VC++ 6 (1998) and earlier are not supported.
   #
   # Options for VC++ 7.0 (2002), 7.1 (2003):
   #            Disable optimizations: /Od
   #             Maximum optimization: /Ox
   #      Enable runtime debug checks: /GZ
   #   Optimize for Pentium processor: /G5
   #         Optimize for Pentium Pro: /G6
   #
   # Options for VC++ 8.0 (2005), 9.0 (2008):
   #                  Disable optimizations: /Od
   #                   Maximum optimization: /Ox
   #                    Enable stack checks: /GZ
   #                   Require SSE2 support: /arch:SSE2
   # Fast (less predictable) floating point: /fp:fast
   #     Use portable but insecure lib fcns: /D_CRT_SECURE_NO_DEPRECATE
   #
   # Default optimization
   #   set opts {}
   # Max optimization
   set opts [GetClGeneralOptFlags $cl_version x86_64]
   # Aggressive optimization flags, some of which are specific to
   # particular cl versions, but are all processor agnostic.  CPU
   # specific opts are introduced in farther below.  See
   # cpuguess-windows-x86_64.tcl and x86-support.tcl for details.

   # CPU model architecture specific options.  To override, set value
   # program_compiler_c++_cpu_arch in
   # oommf/config/platform/local/windows-x86_64.tcl.
   if {[catch {$config GetValue program_compiler_c++_cpu_arch} cpu_arch]} {
      set cpu_arch generic
   }

   set cpuopts {}
   if {![string match generic [string tolower $cpu_arch]]} {
      # Arch specific build.  If cpu_arch is "host", then try to
      # guess.  Otherwise, assume user knows what he is doing and has
      # inserted an appropriate cpu_arch string, i.e., one that
      # matches the format and known types as returned from GuessCpu.
      if {[string match host $cpu_arch]} {
         set cpu_arch [GuessCpu]
      }
      # Use/don't use SSE intrinsics.  In the cpu_arch!=generic case,
      # the default behavior is to set this from the third element of
      # the GuessCpu return.  If cpu_arch==generic, then the default
      # is no.  You can always override the default behavior setting
      # the $config sse_level value in the local platform file (see
      # LOCAL CONFIGURATION above).
      if {[catch {$config GetValue sse_level}]} {
         # sse_level not set in LOCAL CONFIGURATION block
         $config SetValue sse_level [lindex $cpu_arch 2]
      }
      set cpuopts [GetClCpuOptFlags $cl_version $cpu_arch x86_64]
      # Note: In the cpu_arch != generic case, GetClCpuOptFlags will
      # include the appropriate SSE flags.  If possible, use this
      # rather than trying to manually set the SSE level because
      # GetClCpuOptFlags knows something about what options are
      # available in which versions of cl
   }
   unset cpu_arch
   # You can override the above results by directly setting or
   # unsetting the cpuopts variable, e.g.,
   #
   #    set cpuopts [list /arch:SSE2]
   # or
   #    unset cpuopts
   #
   if {[info exists cpuopts] && [llength $cpuopts]>0} {
      set opts [concat $opts $cpuopts]
   }

   # Use/don't use SSE source-code intrinsics (as opposed to compiler
   # generated SSE instructions, which are controlled by the /arch:
   # command line option.  The default is '2', because x86_64
   # guarantees at least SSE2.  You can override the value by setting
   # the $config sse_level value in the local platform file (see LOCAL
   # CONFIGURATION above).
   if {[catch {$config GetValue sse_level}]} {
      $config SetValue sse_level 2
   }

   # Silence security warnings
   if {$cl_major_version>7} {
      lappend opts /D_CRT_SECURE_NO_DEPRECATE
   }

   # Make user requested tweaks to compile line
   if {![catch {$config GetValue program_compiler_c++_add_flags} extraflags]} {
      foreach elt $extraflags {
         if {[lsearch -exact $opts $elt]<0} {
            lappend opts $elt
         }
      }
   }
   if {![catch {$config GetValue program_compiler_c++_remove_flags} noflags]} {
      foreach elt $noflags {
         regsub -all -- $elt $opts {} opts
      }
      regsub -all -- {\s+-} $opts { -} opts  ;# Compress spaces
      regsub -- {\s*$} $opts {} opts
   }

   #cuiwl: set comd options for nvcc
   $config SetValue program_compiler_cuda_option_opt ""
   $config SetValue program_compiler_cuda_option_out {format "\"-o=%s\""}
   $config SetValue program_compiler_cuda_option_inc {format "\"-I %s\""}
   
   # NOTE: If you want good performance, be sure to edit ../options.tcl
   #  or ../local/options.tcl to include the line
   #    Oc_Option Add * Platform cflags {-def NDEBUG}
   #  so that the NDEBUG symbol is defined during compile.
   $config SetValue program_compiler_c++_option_opt "format \"$opts\""

   $config SetValue program_compiler_c++_option_out {format "\"/Fo%s\""}
   $config SetValue program_compiler_c++_option_src {format "\"/Tp%s\""}
   $config SetValue program_compiler_c++_option_inc {format "\"/I%s\""}
   $config SetValue program_compiler_c++_option_warn {
      format "/W4 /wd4505 /wd4702"
   }
   #   Warning C4505 is about removal of unreferenced local functions.
   # This seems to be a common occurrence when using templates with the
   # so-called "Borland" model.
   #   Warning C4702 is about unreachable code.  A lot of warnings of
   # this type are generated in the STL; I'm not even sure they are all
   # true.
   #
   # $config SetValue program_compiler_c++_option_debug {format "/MLd"}
   $config SetValue program_compiler_c++_option_debug {format "/Zi"}
   $config SetValue program_compiler_c++_option_def {format "\"/D%s\""}

   # Use OOMMF supplied erf() error function
   $config SetValue program_compiler_c++_property_no_erf 1

   # Use _hypot() in place of hypot()
   $config SetValue program_compiler_c++_property_use_underscore_hypot 1

   # Use _getpid() in place of getpid()
   $config SetValue program_compiler_c++_property_use_underscore_getpid 1

   # Widest natively support floating point type
   $config SetValue program_compiler_c++_typedef_realwide "double"

   # Under the Microsoft Visual C++ compiler, 80-bit floating point
   # is not supported; both double and long double are 64-bits. and
   # there are no extra precision intermediate values.
   $config SetValue program_compiler_c++_property_fp_double_extra_precision 0

   # Visual C++ 6.0 does not support direct conversion from
   # unsigned __int64 to double.  If automatic detection doesn't
   # work, set cl_version directly to 5, 6, or 7, as appropriate.
   $config SetValue program_compiler_c++_uint64_to_double 0
   if {$cl_major_version>=7} {
      $config SetValue program_compiler_c++_uint64_to_double 1
   }

   # Visual C++ 8.0 does not provide the _mm_cvtsd_f64 intrinsic.
   # Visual C++ 9.0 and later do.
   if {$cl_major_version < 9} {
      $config SetValue program_compiler_c++_missing_cvtsd_f64 1
   } else {
      $config SetValue program_compiler_c++_missing_cvtsd_f64 0
   }

   # The program to run on this platform to create a single library file out
   # of many object files.
   # Microsoft Visual C++'s library maker
   $config SetValue program_libmaker {link /lib}
   # If your link doesn't accept the /lib option, try this instead:
   # $config SetValue program_libmaker {lib}
   $config SetValue program_libmaker_option_obj {format \"%s\"}
   $config SetValue program_libmaker_option_out {format "\"/OUT:%s\""}

   # The program to run on this platform to link together object files and
   # library files to create an executable binary.
   # Microsoft Visual C++'s linker
   $config SetValue program_linker {link /nodefaultlib:vcomp100}
   #cuiwl, nvcc linker for device code
   $config SetValue device_linker {nvcc -arch=sm_30 -dlink}
   # $config SetValue program_linker {link /DEBUG} ;# For debugging
   $config SetValue program_linker_option_obj {format \"%s\"}
   #$config SetValue program_linker_option_out {format "-o %s"}
   $config SetValue program_linker_option_out {format "\"/OUT:%s\""}
   $config SetValue program_linker_option_lib {format \"%s\"}
   $config SetValue program_linker_option_sub {format "\"/SUBSYSTEM:%s\""}
   #cuiwl: set the cuda libs
   set cuda_home ""
   if {[info exist env(CUDA_HOME)]} {
		#Sidi modified 02/23/15, for compatibility in Windows
		set tmp_name $env(CUDA_PATH)
		set tmp_name_list [file split $tmp_name] 
		set cuda_home [eval file join $tmp_name_list]
		set cuda_lib_path "$cuda_home/lib/x64"
   }
   #Sidi modified 02/23/15
   set cudartLib "$cuda_lib_path/cudart.lib"
   set cudaLib "$cuda_lib_path/cuda.lib"
   set cudadevrtLib "$cuda_lib_path/cudadevrt.lib"
   set cufftLib "$cuda_lib_path/cufft.lib"
   # set cudartLib "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/lib/x64/cudart.lib"
   # set cudaLib "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/lib/x64/cuda.lib"
   # set cudadevrtLib "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/lib/x64/cudadevrt.lib"
   # set cufftLib "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v5.5/lib/x64/cufft.lib"
   $config SetValue CUDA_LIB [list $cudartLib $cudaLib $cufftLib]
   #cuiwl: set the mkl libs
   set mklcoreLib "C:/Program Files (x86)/Intel/Composer XE 2013/mkl/lib/intel64/mkl_core.lib"
   set mklintelLib "C:/Program Files (x86)/Intel/Composer XE 2013/mkl/lib/intel64/mkl_intel_lp64.lib"
   set mklthreadedLib "C:/Program Files (x86)/Intel/Composer XE 2013/mkl/lib/intel64/mkl_intel_thread.lib"
   $config SetValue MKL_LIB ""
   #[list $mklcoreLib $mklintelLib $mklthreadedLib]
   ##cuiwl: set the omp libs
   #set ompLib "C:/Program Files (x86)/Intel/Composer XE 2011 SP1/compiler/lib/intel64/libiomp5md.lib"
   $config SetValue OMP_LIB ""
   #[list $ompLib]
   
   $config SetValue TCL_LIB_SPEC [$config GetValue TCL_VC_LIB_SPEC]
   $config SetValue TK_LIB_SPEC [$config GetValue TK_VC_LIB_SPEC]
   # Note: advapi32 is needed for GetUserName function in Nb package.
   $config SetValue TK_LIBS {user32.lib advapi32.lib}
   $config SetValue TCL_LIBS {user32.lib advapi32.lib}
   $config SetValue program_linker_uses_-L-l {0}
   $config SetValue program_linker_uses_-I-L-l {0}

   unset cl_version
   unset cl_major_version
} elseif {[string match g++* $ccbasename]} {
   # ... for MINGW32 + GNU g++ C++ compiler
   if {![info exists gcc_version]} {
      set gcc_version [GuessGccVersion \
                          [lindex [$config GetValue program_compiler_c++] 0]]
   }
   $config SetValue program_compiler_c++_banner_cmd \
      [list GetGccBannerVersion  \
          [lindex [$config GetValue program_compiler_c++] 0]]

   # Optimization options
   # set opts [list -O0 -fno-inline -ffloat-store]  ;# No optimization
   # set opts [list -O%s]                      ;# Minimal optimization
   set opts [GetGccGeneralOptFlags $gcc_version]
   # Aggressive optimization flags, some of which are specific to
   # particular gcc versions, but are all processor agnostic.  CPU
   # specific opts are introduced in farther below.  See
   # x86-support.tcl for details.

   # CPU model architecture specific options.  To override, set Option
   # program_compiler_c++_cpu_arch in oommf/config/options.tcl (or,
   # preferably, in oommf/config/local/options.tcl).  See note about SSE
   # below.
   if {[catch {$config GetValue program_compiler_c++_cpu_arch} cpu_arch]} {
      set cpu_arch generic
   }
   set cpuopts {}
   if {![string match generic [string tolower $cpu_arch]]} {
      # Arch specific build.  If cpu_arch is "host", then try to
      # guess.  Otherwise, assume user knows what he is doing and has
      # inserted an appropriate cpu_arch string, i.e., one that
      # matches the format and known types as returned from GuessCpu.
      if {[string match host $cpu_arch]} {
         set cpu_arch [GuessCpu]
      }
      # Use/don't use SSE intrinsics.  In the cpu_arch!=generic case,
      # the default behavior is to set this from the third element of
      # the GuessCpu return.  If cpu_arch==generic, then the default is
      # 2 (minimum level guaranteed on x86_64).  You can always override
      # the default behavior setting the $config sse_level value in the
      # local platform file (see LOCAL CONFIGURATION above).
      if {[catch {$config GetValue sse_level}]} {
         # sse_level not set in LOCAL CONFIGURATION block
         $config SetValue sse_level [lindex $cpu_arch 2]
      }
      set cpuopts [GetGccCpuOptFlags $gcc_version $cpu_arch]
   }
   unset cpu_arch
   # You can override the above results by directly setting or
   # unsetting the cpuopts variable, e.g.,
   #
   #    set cpuopts [list -march=athlon]
   # or
   #    unset cpuopts
   #
   if {[info exists cpuopts] && [llength $cpuopts]>0} {
      set opts [concat $opts $cpuopts]
   }

   # SSE support
   if {[catch {$config GetValue sse_level} sse_level]} {
      set sse_level 2  ;# Default setting for x86_64
      $config SetValue sse_level $sse_level
   }
   if {!$sse_level} {
      # Strip out all SSE options
      regsub -all -- {^-mfpmath=sse\s+|\s+-mfpmath=sse(?=\s|$)} $opts {} opts
      regsub -all -- {^-msse\d*\s+|\s+-msse\d*(?=\s|$)} $opts {} opts
      lappend opts -mfpmath=387
   }

   # Default warnings disable
   set nowarn [list -Wno-non-template-friend]
   if {[info exists nowarn] && [llength $nowarn]>0} {
      set opts [concat $opts $nowarn]
   }
   catch {unset nowarn}

   # Make user requested tweaks to compile line
   if {![catch {$config GetValue program_compiler_c++_add_flags} extraflags]} {
      foreach elt $extraflags {
         if {[lsearch -exact $opts $elt]<0} {
            lappend opts $elt
         }
      }
   }
   if {![catch {$config GetValue program_compiler_c++_remove_flags} noflags]} {
      foreach elt $noflags {
         regsub -all -- $elt $opts {} opts
      }
      regsub -all -- {\s+-} $opts { -} opts  ;# Compress spaces
      regsub -- {\s*$} $opts {} opts
   }

   $config SetValue program_compiler_c++_option_opt "format \"$opts\""
   # NOTE: If you want good performance, be sure to edit ../options.tcl
   #  or ../local/options.tcl to include the line
   #    Oc_Option Add * Platform cflags {-def NDEBUG}
   #  so that the NDEBUG symbol is defined during compile.
   $config SetValue program_compiler_c++_option_out {format "-o \"%s\""}
   $config SetValue program_compiler_c++_option_src {format \"%s\"}
   $config SetValue program_compiler_c++_option_inc {format "\"-I%s\""}
   $config SetValue program_compiler_c++_option_debug {format "-g"}
   $config SetValue program_compiler_c++_option_def {format "\"-D%s\""}

   # Compiler warnings:
   # Omitted: -Wredundant-decls -Wshadow -Wcast-align
   # I would also like to use -Wcast-qual, but casting away const is
   # needed on some occasions to provide "conceptual const" functions in
   # place of "bitwise const"; cf. p76-78 of Meyer's book, "Effective C++."
   #
   # NOTE: -Wno-uninitialized is required after -Wall by gcc 2.8+ because
   # of an apparent bug.  -Winline is out because of failures in the STL.
   # Depending on the gcc version, the following options may also be
   # available:     -Wbad-function-cast     -Wstrict-prototypes
   #                -Wmissing-declarations  -Wnested-externs
   # Update: gcc 3.4.5 issues an uninitialized warning in the STL,
   #  so -Wno-uninitialized is necessary.
   $config SetValue program_compiler_c++_option_warn {format "-Wall \
        -W -Wpointer-arith -Wwrite-strings \
        -Woverloaded-virtual -Wsynth -Werror -Wno-uninitialized \
        -Wno-unused-function"}

   # Wide floating point type.
   # NOTE: On the x86_64+gcc platform, "long double" provides better
   # precision than "double", but at a cost of increased memory usage
   # and a decrease in speed.  (The long double takes 16 bytes of
   # storage as opposed to 8 for double, but provides the x86 80-bit
   # native floating point format having approx. 19 decimal digits
   # precision as opposed to 16 for double.)
   # Default is "double".
   # $config SetValue program_compiler_c++_typedef_realwide "long double"

   # Experimental: The OC_REAL8m type is intended to be at least
   # 8 bytes wide.  Generally OC_REAL8m is typedef'ed to double,
   # but you can try setting this to "long double" for extra
   # precision (and extra slowness).  If this is set to "long double",
   # then so should realwide in the preceding stanza.
   # $config SetValue program_compiler_c++_typedef_real8m "long double"

   # Directories to exclude from explicit include search path, i.e.,
   # the -I list.  Some versions of gcc complain if "system" directories
   # appear in the -I list.
   $config SetValue \
      program_compiler_c++_system_include_path [list /usr/include]
	  
   #cuiwl: set the cuda lib path (not sure)
   $config SetValue \ 
      program_compiler_cuda_system_include_path "$cuda_home\include"

   $config SetValue \
	  program_compiler_mkl_system_include_path  "C:\Program Files (x86)\Intel\Composer XE 2013\mkl\include"
   
   # Other compiler properties
   $config SetValue \
      program_compiler_c++_property_optimization_breaks_varargs 0

   # The program to run on this platform to create a single library file out
   # of many object files.
   # ... GNU ar ...
   $config SetValue program_libmaker {ar cr}
   $config SetValue program_libmaker_option_obj {format \"%s\"}
   $config SetValue program_libmaker_option_out {format \"%s\"}

   # The program to run on this platform to link together object files and
   # library files to create an executable binary.

   # NOTE: g++-built executables link in libgcc and libstdc++ libraries
   #       from the g++ distribution.  The default it to link these as
   #       shared libraries, in which case the directory containing
   #       these libraries (typically C:\MinGW\bin) needs to be included
   #       in the PATH environment variable.  Alternatively, one can add
   #       the -static-libgcc -static-libstdc++ options to the link
   #       line, in which case the PATH doesn't need to be set, but the
   #       executables will be larger.
   #       Pick one:
   # $config SetValue program_linker [list $ccbasename]
   $config SetValue program_linker [list $ccbasename -static-libgcc -static-libstdc++]

   $config SetValue program_linker_option_obj {format \"%s\"}
   $config SetValue program_linker_option_out {format "-o \"%s\""}
   $config SetValue program_linker_option_lib {format \"%s\"}
   proc fufufubar { subsystem } {
      if {[string match CONSOLE $subsystem]} {
         return "-Wl,--subsystem,console"
      }
      return "-Wl,--subsystem,windows"
   }
   $config SetValue program_linker_option_sub {fufufubar}
   $config SetValue program_linker_uses_-L-l {1}
   # First try "Visual C" names for Tcl/Tk libraries.  If that
   # doesn't work, then try unix naming convention, which should
   # pick up MinGW builds of Tcl/Tk.
   set tcllib [$config GetValue TCL_VC_LIB_SPEC]
   set tklib  [$config GetValue TK_VC_LIB_SPEC]
   if {![file exists $tcllib]} {
      set path [file dirname $tcllib]
      set fname [file tail $tcllib]
      regsub {\..*$} $fname {} fname
      set tname [file join $path lib${fname}.a]
      if {[file exists $tname]} {
         set tcllib $tname
      }
   }
   if {![file exists $tklib]} {
      set path [file dirname $tklib]
      set fname [file tail $tklib]
      regsub {\..*$} $fname {} fname
      set tname [file join $path lib${fname}.a]
      if {[file exists $tname]} {
         set tklib $tname
      }
   }
   $config SetValue TCL_LIB_SPEC $tcllib
   $config SetValue TK_LIB_SPEC $tklib
   $config SetValue TK_LIBS {}
   $config SetValue TCL_LIBS {}

   unset gcc_version
} else {
   puts stderr "Warning: Requested compiler \"$ccbasename\" is not supported."
}
catch {unset ccbasename}

# The absolute, native filename of the null device
$config SetValue path_device_null {nul:}

# A partial Tcl command (or script) which when completed by lappending
# a file name stem and evaluated returns the corresponding file name for
# an executable on this platform
$config SetValue script_filename_executable {format %s.exe}

# A partial Tcl command (or script) which when completed by lappending
# a file name stem and evaluated returns the corresponding file name for
# an object file on this platform
$config SetValue script_filename_object {format %s.obj}

# A partial Tcl command (or script) which when completed by lappending
# a file name stem and evaluated returns the corresponding file name for
# a static library on this platform
$config SetValue script_filename_static_library {format %s.lib}

# A list of partial Tcl commands (or scripts) which when completed by
# lappending a file name stem and evaluated returns the corresponding
# file name for an intermediate file produced by the linker on this platform
$config SetValue script_filename_intermediate [list \
   {format %s.ilk} {format %s.pdb} {format %s.map}]

########################################################################
unset config
