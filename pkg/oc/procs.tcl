# FILE: procs.tcl
#
# A collection of Tcl procedures (not Oc_Classes) which are part of the
# Oc extension
#
# Need to split this into files likely to load together
#
# Last modified on: $Date: 2012-09-27 20:58:35 $
# Last modified by: $Author: dgp $

# Returns the absolute, direct pathname of its argument
# Note: This code has been "borrowed" into the mifconvert
#   application.  Bugs and improvements should be echoed
#   there.
proc Oc_DirectPathname { pathname } {
    global Oc_DirectPathnameCache
    set canCdTo [file dirname $pathname]
    set rest [file tail $pathname]
    switch -exact -- $rest {
        .	-
        .. {
            set canCdTo [file join $canCdTo $rest]
            set rest ""
        }
    }
    if {[string match absolute [file pathtype $canCdTo]]} {
        set index $canCdTo
    } else {
        set index [file join [pwd] $canCdTo]
    }
    if {[info exists Oc_DirectPathnameCache($index)]} {
        return [file join $Oc_DirectPathnameCache($index) $rest]
    }
    if {[catch {set savedir [pwd]} msg]} {
        return -code error "Can't determine pathname for\n\t$pathname:\n\t$msg"
    }
    # Try to [cd] to where we can [pwd]
    while {[catch {cd $canCdTo}]} {
        switch -exact -- [file tail $canCdTo] {
            "" {
                # $canCdTo is the root directory, and we can't cd to it.
                # This means we know the direct pathname, even though we
                # can't cd to it or any of its ancestors.
                set Oc_DirectPathnameCache($index) $canCdTo	;# = '/'
                return [file join $Oc_DirectPathnameCache($index) $rest]
            }
            . {
                # Do nothing.  Leave $rest unchanged
            }
            .. {
                # NOMAC: Assuming '..' means 'parent directory'
                # Don't want to shift '..' onto $rest.
                # Make recursive call instead.
                set Oc_DirectPathnameCache($index) [file dirname \
                        [Oc_DirectPathname [file dirname $canCdTo]]]
                return [file join $Oc_DirectPathnameCache($index) $rest]
            }
            default {
                ;# Shift one path component from $canCdTo to $rest
                set rest [file join [file tail $canCdTo] $rest]
            }
        }
        set canCdTo [file dirname $canCdTo]
        set index [file dirname $index]
    }
    # We've successfully changed the working directory to $canCdTo
    # Try to use [pwd] to get the direct pathname of the working directory
    catch {set Oc_DirectPathnameCache($index) [pwd]}
    # Shouldn't be a problem with a [cd] back to the original working directory
    cd $savedir
    if {![info exists Oc_DirectPathnameCache($index)]} {
        # Strange case where we could [cd] into $canCdTo, but [pwd] failed.
        # Try a recursive call to resolve matters.
        set Oc_DirectPathnameCache($index) [Oc_DirectPathname $canCdTo]
    }
    return [file join $Oc_DirectPathnameCache($index) $rest]
}

# Routine to completely resolve file links
# Raises an error if pathname is not a link, or
# if the link cannot be resolved
proc Oc_ResolveLink { pathname } {
    if {[catch {file type $pathname} ftype]} {
	if {![file exists $pathname]} {
	    return -code error "File $pathname does not exist"
	}
	return -code error $ftype
    }
    if {![string match "link" $ftype]} {
	return -code error "$pathname is not a link"
    }
    if {![file exists $pathname]} {
	# This should catch loops and hanging links
	return -code error "Link $pathname cannot be resolved."
    }
    set workname $pathname
    set arr($workname) 1
    for {set depth 0} {$depth<20} {incr depth} {
	set tmpname [file readlink $workname]
	set ptype [file pathtype $tmpname]
	switch -exact -- $ptype {
	    absolute { set workname $tmpname }
	    relative {
		set workname [file join [file dirname $workname] $tmpname]
	    }
	    default {
		return -code error "Unsupported pathtype: $ptype"
	    }
	}
	if {![string match "link" [file type $workname]]} {
	    return $workname
	}
	if {[info exists arr($workname)]} {
	    return -code error \
		    "Link $pathname cannot be resolved: Loop detected"
	}
	set arr($workname) 1
    }
    return -code error "Link $pathname cannot be resolved: > $depth levels"
}

proc Oc_MakeHeaderWrappers {outdir} {
    # Some C++ compiler systems (HP's aCC) have only old-style
    # header files, e.g., <iostream.h> instead of <iostream>.
    # The procedure checks
    #
    #    program_compiler_c++_property_oldstyle_headers
    #
    # for a list of such files, and creates for each a simple
    # wrapper file using the new naming convention that includes
    # a file with the old naming convention.  It drops these
    # wrapper files into the "outdir" directory specified.
    #   The return value is 0 on success, >0 otherwise.
    set errcount 0
    set config [Oc_Config RunPlatform]
    if {![catch {
	$config GetValue program_compiler_c++_property_oldstyle_headers
    } headers] && [llength $headers]>0} {
	foreach file $headers {
	    # Strip trailing .h, if any
	    regsub -nocase -- {\.h$} $file {} file

	    # Open output file
	    set outfile [file join $outdir $file]
	    puts "Creating header wrapper [file join [pwd] $outfile] ..."
	    if {[catch {open $outfile w} fileid]} {
		puts stderr "Unable to open machine wrapper header\
			file $outfile for writing"
		incr errcount
		continue
	    }

	    # Dump include file workaround
	    puts $fileid \
"/* FILE: $file           -*-Mode: c++-*-
 *
 * Machine specific header wrapper, generated by \[Oc_MakeHeaderWrappers\]
 *
*/"
            puts $fileid "#include <$file.h>"

            # Close output file
	    close $fileid
	}
    }
    return $errcount
}

# The next proc is used in oommf/config/names/cygtel.tcl to determine
# the OOMMF platform name, and further below in this file to set the
# OC_SYSTEM_TYPE and OC_SYSTEM_SUBTYPE macros for ocport.h.
proc Oc_IsCygwinPlatform {} {
   global tcl_platform env
   if {![string match intel $tcl_platform(machine)] &&
       ![string match i?86  $tcl_platform(machine)]} {
      return 0
   }
   if {[string match cygwin* [string tolower $tcl_platform(os)]]} {
      return 1
   }
   if {[info exists env(OSTYPE)] && 
       [string match cygwin* [string tolower $env(OSTYPE)]]} {
      return 1
   }
   if {[string match cyg* [file tail [info nameofexecutable]]]} {
      return 1
   }
   if {![catch {exec uname} osname] &&
       [regexp -nocase -- cygwin $osname]} {
      return 1
   }

   return 0
}

proc Oc_MakePortHeader {varinfo outfile} {
    # Eventually make calls to objects representing local configuration.
    Oc_MakeHeaderWrappers [file dirname $outfile]

    puts "Updating [file join [pwd] $outfile] ..."
    global tcl_platform
    # See if we can tell what platform we are on
    set config [Oc_Config RunPlatform]
    set systemtype unknown  ;# For local use
    set systemsubtype unknown
    if {![info exists tcl_platform(platform)]} {
	set systemtype unix  ;# Can't tell, so assume unix
    } else {
	if {[string compare $tcl_platform(platform) "unix"] == 0} {
	    set systemtype unix
	    if {[string compare $tcl_platform(os) "Darwin"] == 0} {
		set systemsubtype darwin
	    }
	} elseif {[string compare $tcl_platform(platform) "windows"] == 0} {
	    if {[Oc_IsCygwinPlatform]} {
		# Building under the cygwin toolkit
		set systemtype unix
                set systemsubtype cygwin
	    } else {
		set systemtype windows
		if {[string compare $tcl_platform(os) "Windows NT"] == 0} {
		    set systemsubtype winnt
		}
	    }
	} else {
           error "Unsupported platform: $tcl_platform(platform)"
	}
    }


    # Run varinfo and parse output
    set varinfo_flags {}
    if {![catch {
	$config GetValue program_compiler_c++_property_strict_atan2
    }]} {
	# Property already set (probably from platform file).  Keep
	# this value and disable atan2 test in varinfo.
	lappend varinfo_flags "--skip-atan2"
    }
    if {[catch {eval exec $varinfo $varinfo_flags 2>@ stderr} varlist]} {
	# error running varinfo, probably killed by atan2 test.
	# Try again, disabling that test
	$config SetValue program_compiler_c++_property_strict_atan2 1
	lappend varinfo_flags --skip-atan2
	if {[catch {eval exec $varinfo $varinfo_flags 2>@ stderr} varlist]} {
	    set msg "Error running $varinfo $varinfo_flags:\n$varlist"
	    error $msg $msg
	}
    }
    append varlist "\n"  ;# Simplify whole-line regexp searches

    set varinttypelist {char short int long {long long} __int64}
    set varfloattypelist {float double {long double}}
    foreach vartype [concat $varinttypelist $varfloattypelist] {
	set varsize($vartype)  -1   ;# Safety
	set varorder($vartype) -1
	regexp \
        "Type *$vartype *is *(\[0-9\]*) bytes wide *Byte order: *(\[0-9\]*)" \
           $varlist tempmatch varsize($vartype) varorder($vartype)
    }
    set varsize(pointer) -1
    regexp {void \* is *([0-9]*)} $varlist tempmatch varsize(pointer)

    foreach varwidth {FLT DBL} {
        regexp "\n${varwidth}_EPSILON: (\[^\n\]+)" \
                $varlist tempmatch vareps($varwidth)
        regexp "\nSQRT_${varwidth}_EPSILON: (\[^\n\]+)" \
                $varlist tempmatch vareps(SQRT_$varwidth)
    }
    regexp "\nLDBL_EPSILON: (\[^\n\]+)" \
            $varlist tempmatch vareps(LDBL)
    regexp "\nCUBE_ROOT_DBL_EPSILON: (\[^\n\]+)" \
	    $varlist tempmatch vareps(CUBE_ROOT_DBL)
    regexp "\nCalculated Double Epsilon: (\[^\n\]+)" \
	    $varlist tempmatch vareps(COMPUTED_DBL)
    regexp "\nCalculated HUGEFLOAT Epsilon: (\[^\n\]+)" \
	    $varlist tempmatch vareps(COMPUTED_HUGE)


    if {[catch {
	$config GetValue program_compiler_c++_property_strict_atan2
    }]} {
	# Config value program_compiler_c++_property_strict_atan2
	# has not been set, so make use of varinfo test.
	# Initialize atan2_value to NaN (Not-a-Number).  If varinfo
	# has reported on atan2(0,0), and if the value is in the range
	# [-Pi,Pi], then set program_compiler_c++_property_strict_atan
	# false, which allows OOMMF code to make direct calls to the
	# system math library atan2 function.  Otherwise, see that
	# Oc_Atan2 gets wrapped around atan2 calls to protect against
	# the (0,0) input case.
	set atan2_value "NaN"
        regexp -- "\nReturn value from atan2\\(0,0\\): *(.*\[^\n\])" \
		$varlist tempmatch atan2_value
	set atan2_value [string trim $atan2_value]
	if {![catch {expr $atan2_value>-3.15 && $atan2_value<3.15} result] \
	    && $result} {
	    # Looks like (0,0) is in the domain of atan2
	    $config SetValue program_compiler_c++_property_strict_atan2 0
	} else {
	    # atan2(0,0) probably returns NaN.  In any case, enable
	    # special handling of (0,0) for atan2
	    $config SetValue program_compiler_c++_property_strict_atan2 1
	}
    }


    if {[catch {
       $config GetValue program_compiler_c++_property_bad_wide2int
    }]} {
       # Config value program_compiler_c++_property_bad_wide2int
       # has not been set, so make use of varinfo test.
       if {[regexp -- "\nGood floorl.\n" $varlist]} {
          $config SetValue program_compiler_c++_property_bad_wide2int 0
       } else {
          $config SetValue program_compiler_c++_property_bad_wide2int 1
       }
    }

    if {[catch {
       $config GetValue program_compiler_c++_property_pagesize
    }]} {
       # Config value program_compiler_c++_property_pagesize
       # has not been set, so make use of varinfo test.
       if {![regexp "\nMemory pagesize: *(\[0-9\]+) *bytes" \
               $varlist tempmatch memory_pagesize] || $memory_pagesize<=0} {
          set memory_pagesize 4096   ;# Pagesize is unknown; use best guess
       }
       $config SetValue program_compiler_c++_property_pagesize $memory_pagesize
    }

    if {[catch {
       $config GetValue program_compiler_c++_property_cache_linesize
    }]} {
       # Config value program_compiler_c++_property_cache_linesize
       # has not been set, so check varinfo output.
       if {![regexp "\nCache linesize: *(\[0-9\]+) *bytes" \
               $varlist tempmatch cache_linesize] || $cache_linesize<=0} {
          set cache_linesize 64  ;# Cache linesize is unknown; use best guess
       }
       $config SetValue program_compiler_c++_property_cache_linesize \
          $cache_linesize
    }

    # Dump header info
    set porth [subst \
{/* FILE: ocport.h             -*-Mode: c++-*-
 *
 * Machine specific #define's and typedef's, generated by \[Oc_MakePortHeader\]
 *
 */

#ifndef _OC_PORT
#define _OC_PORT

#define OOMMF_API_INDEX [$config OommfApiIndex]

#include <stdlib.h>
#include <limits.h>}]

    # Some compilers have broken cmath header files.
    if {![catch {
	$config GetValue \
		program_compiler_c++_property_bad_cmath_header
    } _] && $_} {
	append porth {

/* The cmath header file is specified as broken in the platform
 * configuration file (in the directory oommf/config/platforms), so
 * include math.h instead of cmath.  Depending on the compiler
 * this may disable automatic usage of "long double" math
 * library functions.
 */
#include <float.h>
#include <math.h>
}
    } else {
       append porth "\n#include <cfloat>\n#include <cmath>"
    }

    # Does compiler not support C++ exceptions?
    if {[catch {
	$config GetValue program_compiler_c++_property_no_exceptions
    } _] || !$_} {
	append porth {

// See Stroustrup, Section 16.1.3.
#include <exception>		// The base class std::exception and the
				// standard exception std::bad_exception
#include <new>			// The standard exception std::bad_alloc
#include <typeinfo>		// The standard exceptions std::bad_cast
				// and std::bad_typeid
#define OC_THROW(x) throw x
}
    } else {
	error "C++ compiler must support exceptions!"
    }

    # Windows specific includes
    if {[string compare $systemtype windows] == 0} {
        append porth {

/* getpid() prototype for Windows*/
#include <process.h>

/* Windows header file.  NB: This defines a lot of stuff we      */
/* don't need or really want, like macros min(x,y) and max(x,y). */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef WIN32_LEAN_AND_MEAN
}
    }

    # Unix specific includes
    if {[string compare $systemtype unix] == 0} {
        append porth {
/* For unix */
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>  /* Child process cleanup */
}
    }

    append porth {
/* End includes */
}

    append porth "
#define CONFIG_TCL_MAJOR_VERSION [$config GetValue TCL_MAJOR_VERSION]
#define CONFIG_TCL_MINOR_VERSION [$config GetValue TCL_MINOR_VERSION]
#define CONFIG_TK_MAJOR_VERSION [$config GetValue TK_MAJOR_VERSION]
#define CONFIG_TK_MINOR_VERSION [$config GetValue TK_MINOR_VERSION]"
    if {[catch {set tclpl [$config GetValue TCL_PATCH_LEVEL]}]} {
	regsub {^[0-9]+\.[0-9]+} [info patchlevel] {} tclpl
    }
    append porth "
#define CONFIG_TCL_PATCH_LEVEL \"[$config GetValue TCL_VERSION]$tclpl\""
    if {[catch {set tkpl [$config GetValue TK_PATCH_LEVEL]}]} {
	# Assume Tcl and Tk patch levels are in sync
	# Otherwise would need Tk loaded to access $tk_patchLevel
	regsub {^[0-9]+\.[0-9]+} [info patchlevel] {} tkpl
    }
    append porth "
#define CONFIG_TK_PATCH_LEVEL \"[$config GetValue TK_VERSION]$tkpl\""

    proc PL2LS {pl} {
	switch -- [string index $pl 0] {
		a {return [list 0 [string range $pl 1 end]]}
		b {return [list 1 [string range $pl 1 end]]}
		p -
		. {return [list 2 [string range $pl 1 end]]}
		""  {return [list 2 0]}
		default {return -code error "Bad patchLevel value: $pl"}
	}
    }
    foreach {tclrl tclrs} [PL2LS $tclpl] {break}
    foreach {tkrl tkrs} [PL2LS $tkpl] {break}
    rename PL2LS {}

    append porth "
#define CONFIG_TCL_RELEASE_LEVEL $tclrl
#define CONFIG_TCL_RELEASE_SERIAL $tclrs
#define CONFIG_TK_RELEASE_LEVEL $tkrl
#define CONFIG_TK_RELEASE_SERIAL $tkrs\n"

    catch {append porth "#define CONFIG_TCL_LIBRARY\
	    [$config GetValue TCL_LIBRARY]\n"}

    # Does compiler support the 'using namespace std' directive?
    if {[catch {
	$config GetValue program_compiler_c++_property_no_std_namespace
    } _] || !$_} {
	append porth \
    "#define OC_USE_STD_NAMESPACE using namespace std\n" \
    "#define OC_USE_EXCEPTION typedef std::exception EXCEPTION\n" \
    "#define OC_USE_BAD_ALLOC typedef std::bad_alloc BAD_ALLOC\n" \
    "#define OC_USE_STRING typedef std::string String\n"
    } else {
	append porth \
    "#define OC_USE_STD_NAMESPACE\n" \
    "#define OC_USE_EXCEPTION typedef exception EXCEPTION\n" \
    "#define OC_USE_BAD_ALLOC typedef bad_alloc BAD_ALLOC\n" \
    "#define OC_USE_STRING typedef string String\n"
    }

    # Does compiler have strict atan2 function?
    if {![catch {
	$config GetValue program_compiler_c++_property_strict_atan2
    } _] && $_} {
	append porth "
/* Substitute domain checked atan2 */
#define atan2(y,x) Oc_Atan2((y),(x))\n"
    }

    # Does compiler have bad long double -> to integer routines?
    if {![catch {
	$config GetValue program_compiler_c++_property_bad_wide2int
    } _] && $_} {
	append porth "
/* Force use of double versions of real to int functions */
/* to work around bug in long double library.            */
#define floor(x)  floor(double(x))
#define floorl(x) floor(double(x))
#define ceil(x)   ceil(double(x))
#define ceill(x)  ceil(double(x))\n"
    }

    # Does compiler have erf function?
    if {![catch {
	$config GetValue program_compiler_c++_property_no_erf
    } _] && $_} {
	append porth "
/* Use OOMMF provided error function */
#define erf(x) Oc_Erf((x))\n"
    }

    # Some systems don't have hypot(x,y) in system libs.
    if {![catch {
	$config GetValue \
		program_compiler_c++_property_no_hypot
    } _] && $_} {
	append porth "
/* Use OOMMF provided hypot function */
#define hypot(x,y) Oc_Hypot((x),(y))\n"
   }

    # Use "_func" in place of "func" for a handful of standard library
    # functions.  This is primarily (only?) for Microsoft Visual C++
    # 2005 (aka version 8) which marks the older, portable names as
    # deprecated.
    if {![catch {
	$config GetValue program_compiler_c++_property_use_underscore_hypot
    } _] && $_} {
	append porth "
/* Use _hypot in place of hypot */
#define hypot(x,y) _hypot((x),(y))\n"
    }
    if {![catch {
	$config GetValue program_compiler_c++_property_use_underscore_getpid
    } _] && $_} {
	append porth "
/* Wrapper for system getpid call; use _getpid in place of getpid */
inline int Oc_GetPid() { return _getpid(); }\n"
    } else {
	append porth "
/* Wrapper for system getpid call */
inline int Oc_GetPid() { return getpid(); }\n"
}

    # Does compiler have non-ansi sprintf that returns pointer instead
    # of string length?
    if {![catch {
	$config GetValue program_compiler_c++_property_nonansi_sprintf
    } _] && $_} {
	# Provide wrapper
	append porth {
/* Wrapper to make sprintf ansi-compliant */
#define OC_SPRINTF_WRAP(x) strlen(x)
}
    } else {
	# Dummy wrapper
	append porth {
/* Dummy wrapper for ansi-compliant sprintf */
#define OC_SPRINTF_WRAP(x) (x)
}
    }

    # Does compiler support vsnprintf?
    if {![catch {
	$config GetValue program_compiler_c++_property_no_vsnprintf
    } _] && $_} {
	# No
	append porth {
/* Platform does not have vsnprintf */
#define OC_HAS_VSNPRINTF 0
}
    } else {
	# Yes
	append porth {
/* Platform supports vsnprintf */
#define OC_HAS_VSNPRINTF 1
}
    }

    # Does compiler support strerror_r ?
    if {![catch {
	$config GetValue program_compiler_c++_property_no_strerror_r
    } _] && $_} {
	# No
	append porth {
/* Platform does not have strerror_r */
#define strerror_r(a,b,c) strncpy((b), strerror(a), (c))
}
    }



    # Some STL implementations have broken map<*,*> containers that
    # cannot properly destroy an empty map.  Define a macro for those
    # compilers so that dummy entries can be placed in a map<> before
    # attempting map deletion to work around the problem.
    if {![catch {
	$config GetValue \
		program_compiler_c++_property_stl_map_broken_empty_delete
    } _] && $_} {
	append porth {
/* The STL implementation of map<> is broken.  Empty map<>s cannot
 * be deleted.  Define a macro to surround insertion of a pair into
 * a map<> before its deletion to work around the problem.
 */
#define OC_STL_MAP_BROKEN_EMPTY_DELETE
}
    }

    # Some STL implementations have broken map<*,*> containers that
    # don't like const keys.  Define a macro for workaround code.
    if {![catch {
	$config GetValue \
		program_compiler_c++_property_stl_map_broken_const_key
    } _] && $_} {
	append porth {
/* The STL implementation of map<> is broken.  Const keys are not
 * supported (at least from some key types).  Define a macro for
 * workaround code.
 */
#define OC_STL_MAP_BROKEN_CONST_KEY
}
    }

    # The Open Watcom compiler v1.3 optimizer and/or associated
    # STLport-4.6.2 has some bug involving vector<> and class
    # member function returns with reference parameters.  This
    # bug is exercised in the
    #    void Oxs_Ext::GetGroupedUIntListInitValue(const String&,
    #                                              vector<OC_UINT4m>&)
    # function in oommf/app/oxs/base/ext.cc.  Define a macro to
    # support workaround hacks.
    if {![catch {
	$config GetValue \
		program_compiler_c++_property_watcom_broken_vector_return
    } _] && $_} {
	append porth {
/* Open Watcom optimizer or STLport-4.6.2 bug on member function
 * returns with reference call parameters.  Provide macro for
 * workarouds.
 */
#define OC_OPEN_WATCOM_BROKEN_VECTOR_RETURNS
}
    }

    # Some C++ compilers don't have a complete exception class.
    # In particular, for those missing "uncaught_exception"
    # define a dummy version that always returns false.  This
    # may hamper some error reporting.
    if {![catch {
	$config GetValue \
		program_compiler_c++_property_missing_uncaught_exception
    } _] && $_} {
	append porth {
/* The compiler is missing the uncaught_exception call.  As a workaround,
 * define here a dummy routine.  This may hamper some error reporting.
 */
#define uncaught_exception() (1)
}
    }

    # Some older C++ compilers may be missing placement new[].
    # If they have placement new, then use that as a workaround.
    if {![catch {
        $config GetValue \
		program_compiler_c++_no_placment_new_array
    } _] && $_} {
	append porth {
/* The compiler is missing placement new[] (for arrays).  Use
 * the single item placement new as a workaround.
 */
#define OC_NO_PLACEMENT_NEW_ARRAY
}
    }

    # Fill in missing function prototypes
    set missing_protos \
	[$config Features program_compiler_c++_prototype_supply_*] 
    if {[llength $missing_protos]>0} {
       append porth "\n/* Function prototypes requested by config/platforms */\n"
       foreach func $missing_protos {
	   set proto [$config GetValue $func]
	   if {![regexp ";\[ \n\]*$" $proto]} {
	       append proto ";"   ;# Append trailing semi-colon
	   }
	   append porth "$proto\n"
       }
    }


    # Write universal (I hope!) typedef's
    append porth {
/* Variable type declarations.  The '****#m' */
/* types are at *least* '#' bytes wide.      */
typedef  int                OC_BOOL;
typedef  unsigned char      OC_BYTE;
typedef  char               OC_CHAR;
typedef  unsigned char      OC_UCHAR;
}
    if {![catch {$config GetValue \
          program_compiler_c++_property_has_signed_char} _] && $_} {
       append porth "typedef  signed char        OC_SCHAR;\n"
    } else {
       append porth "typedef  char               OC_SCHAR;\n"
    }

    # Write float typedef's
    append porth "\n"
    if { $varsize(float) != $varsize(double) } {
	append porth "typedef  float              OC_REAL$varsize(float);\n"
        if {$varsize(float) == 8} { set real8type "float" }
    }
    append porth "typedef  double             OC_REAL$varsize(double);\n"
    if {$varsize(double) == 8} { set real8type "double" }

    if {![catch {$config GetValue program_compiler_c++_typedef_real4m} \
         real4mtype]} {
           append porth [format "typedef  %-18s OC_REAL4m;\n" $real4mtype]
    } else {
       if { $varsize(float) >= 4 } {
	   append porth "typedef  float              OC_REAL4m;\n"
       }
    }
    if {![catch {$config GetValue program_compiler_c++_typedef_real8m} \
         real8mtype]} {
           append porth [format "typedef  %-18s OC_REAL8m;\n" $real8mtype]
    } else {
       if { $varsize(float) >= 8 } {
           append porth "typedef  float              OC_REAL8m;\n"
           set real8mtype "float"
       } elseif { $varsize(double) >= 8 } {
           append porth "typedef  double             OC_REAL8m;\n"
           set real8mtype "double"
       }
    }
    if {[catch {$config GetValue program_compiler_c++_typedef_realwide} \
	    widetype]} {
	set widetype "OC_REAL8m"
    }
    append porth [format \
	    "typedef  %-18s OC_REALWIDE;  /* Widest native float */\n" \
	    $widetype]

    # The following code breaks if real8mtype is another typedef,
    # but to handle that case it seems we would need to run the
    # compiler on a constructed test code.
    set real8m_is_double 1
    if {[string match "long double" $real8mtype] || \
        [string match "float" $real8mtype] } {
       set real8m_is_double 0
    }
    set real8m_is_real8 1
    if {![string match $real8type $real8mtype]} {
       set real8m_is_real8 0
    }
    set realwide_is_real8 0
    if {[string match "OC_REAL8" $widetype] ||
        ([string match "double" $widetype] && $varsize(double)==8) ||
        ([string match "OC_REAL8m" $widetype] && $real8m_is_real8)} {
       set realwide_is_real8 1
   }

    append porth [format \
            "#define OC_REAL8m_IS_DOUBLE %d\n" $real8m_is_double]
    append porth [format \
            "#define OC_REAL8m_IS_REAL8 %d\n" $real8m_is_real8]
    append porth [format \
            "#define OC_REALWIDE_IS_REAL8 %d\n" $realwide_is_real8]

    append porth "
#define OC_REAL4_EPSILON       $vareps(FLT)
#define OC_SQRT_REAL4_EPSILON  $vareps(SQRT_FLT)
#define OC_REAL8_EPSILON       $vareps(DBL)
#define OC_SQRT_REAL8_EPSILON  $vareps(SQRT_DBL)
#define OC_CUBE_ROOT_REAL8_EPSILON $vareps(CUBE_ROOT_DBL)\n\n"


    if {![catch {$config GetValue \
          program_compiler_c++_property_fp_double_extra_precision} _]} {
       if {$_} {
          append porth "#define OC_FP_DOUBLE_EXTRA_PRECISION 1\n"
       } else {
          append porth "#define OC_FP_DOUBLE_EXTRA_PRECISION 0\n"
       }
    } else {
       # Guess based on data from varinfo
       if {$vareps(COMPUTED_DBL)<0.55*$vareps(DBL) ||
           ([info exists vareps(LDBL)] &&
            10*$vareps(LDBL)<$vareps(DBL) &&
            10000*$vareps(LDBL)>$vareps(DBL))} {
          append porth "#define OC_FP_DOUBLE_EXTRA_PRECISION 1\n"
       } else {
          append porth "#define OC_FP_DOUBLE_EXTRA_PRECISION 0\n"
       }
    }
    if {![catch {$config GetValue \
          program_compiler_c++_property_fp_long_double_extra_precision} _] \
        && $_} {
       append porth "#define OC_FP_LONG_DOUBLE_EXTRA_PRECISION 1\n\n"
    } else {
       # Default guess is that long double does not have extra precision
       append porth "#define OC_FP_LONG_DOUBLE_EXTRA_PRECISION 0\n\n"
    }

    # Write integer typedef's
    set int_type_widths {}
    if { $varsize(short) < $varsize(int) } {
       append porth "typedef  short              OC_INT$varsize(short);\n"
       append porth "typedef  unsigned short     OC_UINT$varsize(short);\n"
       lappend int_type_widths $varsize(short)
    }
    append porth "typedef  int                OC_INT$varsize(int);\n"
    append porth "typedef  unsigned int       OC_UINT$varsize(int);\n"
    lappend int_type_widths $varsize(int)
    if { $varsize(long) > $varsize(int) } {
	append porth "typedef  long                OC_INT$varsize(long);\n"
	append porth "typedef  unsigned long       OC_UINT$varsize(long);\n"
        lappend int_type_widths $varsize(long)
    }
    if {[string compare $systemtype windows] == 0 \
           && $varsize(long) < 8} {
       append porth "typedef  __int64            OC_INT8;\n"
       append porth "typedef  unsigned __int64   OC_UINT8;\n"
       lappend int_type_widths 8
    } elseif { $varsize(long) < $varsize(long\ long)} {
	append porth \
           "typedef  long long          OC_INT$varsize(long\ long);\n"
	append porth \
           "typedef  unsigned long long OC_UINT$varsize(long\ long);\n"
        lappend int_type_widths $varsize(long\ long)
    }
    foreach msize { 2 4 8 16 } {
	if { $varsize(int) >= $msize } {
	    # Use type "int" if possible, as this is likely
	    # to be the preferred machine word size
	    append porth "typedef  int                OC_INT${msize}m;\n"
	    append porth "typedef  unsigned int       OC_UINT${msize}m;\n"
            set varsize(INT${msize}m) $varsize(int)
	} elseif { $varsize(long) >=$msize } {
	    # Otherwise, fall back on long type
	    append porth "typedef  long               OC_INT${msize}m;\n"
	    append porth "typedef  unsigned long      OC_UINT${msize}m;\n"
            set varsize(INT${msize}m) $varsize(long)
        } elseif {[string compare $systemtype windows] == 0 \
                  && $msize == 8} {
	    append porth "typedef  __int64            OC_INT8m;\n"
	    append porth "typedef  unsigned __int64   OC_UINT8m;\n"
            set varsize(INT8m) 8
	} elseif { $varsize(long\ long) >=$msize } {
	    append porth "typedef  long long          OC_INT${msize}m;\n"
	    append porth "typedef  unsigned long long OC_UINT${msize}m;\n"
            set varsize(INT${msize}m) $varsize(long\ long)
        }
    }
    foreach vs { 2 4 8 16 } {
       if {[lsearch -exact $int_type_widths $vs] >= 0} {
          append porth "#define OC_HAS_INT$vs 1\n"
       } else {
          append porth "#define OC_HAS_INT$vs 0\n"
       }
    }
    unset int_type_widths

    append porth "\n/* Width of integer types */\n"
    if {[info exists varsize(int)]} {
	append porth "#define OC_INT_WIDTH $varsize(int)\n"
    }
    if {[info exists varsize(long)]} {
	append porth "#define OC_LONG_WIDTH $varsize(long)\n"
    }
    if {[info exists varsize(INT4m)]} {
	append porth "#define OC_INT4m_WIDTH $varsize(INT4m)\n"
    }
    if {[info exists varsize(INT8m)]} {
	append porth "#define OC_INT8m_WIDTH $varsize(INT8m)\n"
    }

    # Pointers
    append porth "\n/* Width of pointer type */\n"
    append porth "#define OC_POINTER_WIDTH $varsize(pointer)\n"

    # Indexes into arrays
    append porth "\n/* OC_INDEX is the suggested type for array indices.  */\n"
    append porth   "/*   It is a signed int type that is at least 4 bytes */\n"
    append porth   "/*   wide and not narrower than the pointer type.     */\n"
    append porth   "/* OC_UINDEX is the unsigned version of OC_INDEX.  It */\n"
    append porth   "/*   is intended for special-purpose use only; use    */\n"
    append porth   "/*   OC_INDEX where possible.                         */\n"  
    if {![catch {
       $config GetValue program_compiler_c++_oc_index_type
    } oc_index_type_data]} {
       # oc_index types specified in platform file.  This should be a
       # three item list: the OC_INDEX type, the OC_UINDEX type, and the
       # width in bytes of these types.  (Presumably OC_INDEX and
       # OC_UINDEX are the same width.  If not, the small amount of code
       # that actually uses OC_UINDEX will probably die horribly.)
       if {[llength $oc_index_type_data]!=3} {
          error "Invalid oc_index_type_data: $oc_index_type_data"
       }
       set oc_index_type  [lindex $oc_index_type_data 0]
       set oc_uindex_type [lindex $oc_index_type_data 1]
       set oc_index_width [lindex $oc_index_type_data 2]
       if {![regexp {^[0-9]+$} $oc_index_width]} {
          error "Invalid oc_index_width value: $oc_index_width"
       }
    } else {
       # oc_index types not specified in platform files;
       # use automated setting.
       if {$varsize(pointer) < 4} {
          # Unlikely case, and would probably break lots of other
          # stuff, but cover it here anyway.
          set oc_index_type OC_INT4
          set oc_uindex_type OC_UINT4
          set oc_index_width 4
       } elseif {$varsize(pointer) <= $varsize(int)} {
          set oc_index_type int
          set oc_uindex_type {unsigned int}
          set oc_index_width $varsize(int)
       } elseif {[string compare $systemtype windows] == 0 \
                    && $varsize(pointer) > $varsize(long) \
                    && $varsize(pointer) == 8} {
          # Use __int64 type
          set oc_index_type __int64
          set oc_uindex_type {unsigned __int64}
          set oc_index_width 8
       } else { 
          # If the above don't work, fallback is long
          set oc_index_type long
          set oc_uindex_type {unsigned long}
          set oc_index_width $varsize(long)
       }
    }
    append porth "typedef $oc_index_type OC_INDEX;\n"
    append porth "typedef $oc_uindex_type OC_UINDEX;\n"
    append porth "#define OC_INDEX_WIDTH $oc_index_width\n"

    if {![catch {
       $config GetValue program_compiler_c++_oc_index_checks
    } oc_index_checks]} {
       # Note: If enabled, OC_INDEX_CHECKS will probably break most
       # third-party extensions.  This macro is intended primarily
       # for internal development work.
       append porth "#define OC_INDEX_CHECKS $oc_index_checks\n"
    } else {
       append porth "#define OC_INDEX_CHECKS 0\n"
    }


    # Byte order.  For now just use 4-byte wide ordering
    foreach vartype { int long short float double } {
	if { $varsize($vartype) == 4 } {
	    append porth "\n#define OC_BYTEORDER $varorder($vartype)\n"
	    break
	}
    }

   # Use the legacy x86 fpu (i.e., the x87?)
   global tcl_platform
   if {[string match "i*86" $tcl_platform(machine)]    \
       || [string match "x86*" $tcl_platform(machine)] \
       || [string match "*x86" $tcl_platform(machine)] \
       || [string compare "amd64" $tcl_platform(machine)] == 0 \
       || [string compare "intel" $tcl_platform(machine)] == 0 } {
      append porth {
#define OC_USE_X87 1
}  }


   # SSE?
   if {[catch {$config GetValue sse_level} sse_level]} {
      set sse_level 0   ;# Default
   }

   if {$sse_level>0} {
      append porth [subst {
/* Use SSE intrinsics, level $sse_level and lower */
/* Macro OC_USE_SSE is similar to OC_SSE_LEVEL, but is only true
 * if SSE level is at least 2, and the OC_REAL8m type is 8 bytes wide
 * (and therefore agrees with the SSE double precision type.)
 */
#define OC_SSE_LEVEL $sse_level
#if OC_SSE_LEVEL>1 && OC_REAL8m_IS_REAL8 && OC_HAS_INT8
# define OC_USE_SSE OC_SSE_LEVEL
  union OC_SSE_MASK {
     OC_UINT8 imask;
     OC_REAL8 dval;
  };
#else
# define OC_USE_SSE 0
#endif
}] } else {
      append porth {
/* Don't use SSE intrinsics */
#define OC_SSE_LEVEL 0
#define OC_USE_SSE 0
}}

   if {$sse_level>=2} {
      if {[catch {$config GetValue program_compiler_c++_missing_cvtsd_f64} \
              _]} {
         set _ 0  ;# Assume "not missing" as default
      }
      if {$_} {
         append porth {
// SSE intrinsic _mm_cvtsd_f64 not provided by compiler
#define OC_COMPILER_HAS_MM_CVTSD_F54 0
}     } else {
         append porth {
// SSE intrinsic _mm_cvtsd_f64 provided by compiler
#define OC_COMPILER_HAS_MM_CVTSD_F54 1
}}

      if {[catch {$config GetValue program_compiler_c++_broken_storel_pd} \
              _]} {
         set _ 0  ;# Assume "not broken" as default
      }
      if {$_} {
         # _mm_storel_pd broken (or missing).  Provide workaround.
         append porth {
// Wrapper replacing SSE2 intrinsic _mm_storel_pd calls with the equivalent
// SSE2 intrinsic _mm_store_sd.  (_mm_storel_pd implementation is broken
// on some compilers.)
#define _mm_storel_pd(x,y)  _mm_store_sd(x,y)
}
      }
   }

   
   #cuiwl: change it back to 1 if threaded!
    # Compile in thread support?
    if {![catch {$config GetValue oommf_threads} _] && $_} {
       append porth {
/* Build in thread (multi-processing) support */
#define OOMMF_THREADS 0
} 
       if {![catch {$config GetValue \
          program_compiler_c++_property_init_thread_fpu_control_word} _] \
           && $_} {
          append porth {
/* Child threads need to have floating point control word explicitly set. */
#define OC_CHILD_COPY_FPU_CONTROL_WORD 1
}      } else {
          append porth {
/* Child threads don't need to have floating point control word explicitly set. */
#define OC_CHILD_COPY_FPU_CONTROL_WORD 0
}}  } else {
       append porth {
/* Don't provide thread (multi-processing) support */
#define OOMMF_THREADS 0
}   }


    # NUMA?  BTW, no threads ==> no NUMA
    if {(![catch {$config GetValue oommf_threads} _] && $_) &&
        (![catch {$config GetValue use_numa} _] && $_)} {
       append porth {
/* Use NUMA (non-uniform memory access) libraries */
#define OC_USE_NUMA 1
}       } else {
        append porth {
/* Don't use NUMA (non-uniform memory access) libraries */
#define OC_USE_NUMA 0
}}

    # No threads in Tcl prior to 8.1, and so no void definitions in the
    # header files either.  Provide void definitions if necessary.
    if {[$config GetValue TCL_MAJOR_VERSION] < 8 ||
        ([$config GetValue TCL_MAJOR_VERSION] == 8 &&
         [$config GetValue TCL_MINOR_VERSION] == 0)} {
       append porth {
/* No thread primitives in tcl.h, so provide dummy definitions */
#undef  TCL_DECLARE_MUTEX
#define TCL_DECLARE_MUTEX(name)
#undef  Tcl_MutexLock
#define Tcl_MutexLock(mutexPtr)
#undef  Tcl_MutexUnlock
#define Tcl_MutexUnlock(mutexPtr)
#undef  Tcl_MutexFinalize
#define Tcl_MutexFinalize(mutexPtr)
#undef  Tcl_ConditionNotify
#define Tcl_ConditionNotify(condPtr)
#undef  Tcl_ConditionWait
#define Tcl_ConditionWait(condPtr, mutexPtr, timePtr)
#undef  Tcl_ConditionFinalize
#define Tcl_ConditionFinalize(condPtr)
typedef int Tcl_ThreadDataKey;
#if defined __WIN32__
#   define Tcl_ThreadCreateType		unsigned __stdcall
#   define TCL_THREAD_CREATE_RETURN	return 0
#else
#   define Tcl_ThreadCreateType		void
#   define TCL_THREAD_CREATE_RETURN
#endif

}}

    # Tcl_GetString() first appears in Tcl 8.1.
    if {[$config GetValue TCL_MAJOR_VERSION] == 8 &&
        [$config GetValue TCL_MINOR_VERSION] == 0} {
       append porth {
/* Tcl_GetStringFromObj appears in Tcl 8.0, but Tcl_GetString in Tcl 8.1 */
#define Tcl_GetString(foobj) Tcl_GetStringFromObj((foobj),NULL)
}
    }

    # Memory layout
    append porth "
#define OC_PAGESIZE\
 [$config GetValue program_compiler_c++_property_pagesize]\
 /* Natural system memory blocksize, in bytes. */
#define OC_CACHE_LINESIZE\
 [$config GetValue program_compiler_c++_property_cache_linesize]\
 /* L1 data cache line size, in bytes. */
"

    # Machine platform types
    # NOTE: A duplicate of this table is used in oc.cc for setting
    #       "compiletime" features in RunPlatform.  Any changes here
    #       should be echoed there.
    append porth {
/* System type info */
#define OC_UNIX 1
#define OC_WINDOWS 2
#define OC_VANILLA 4
#define OC_DARWIN 5
#define OC_CYGWIN 6
#define OC_WINNT 7
}
    # Note: Local variable "systemtype" is set at top of this proc
    # Note 2: Some brain-damaged compilers bitch if the
    #         OC_SYSTEM_SUBTYPE macro is left undefined.
    if {[string compare $systemtype "unix"] == 0} {
       append porth "#define OC_SYSTEM_TYPE OC_UNIX\n"
    } elseif {[string compare $systemtype "windows"] == 0} {
       append porth "#define OC_SYSTEM_TYPE OC_WINDOWS\n"
    } else {
       error "Unrecognized system type: $systemtype"
    }

    set system_subtype OC_VANILLA
    if {[string compare $systemsubtype "darwin"] == 0} {
       set system_subtype OC_DARWIN
    } elseif {[string compare $systemsubtype "winnt"] == 0} {
       set system_subtype OC_WINNT
    } elseif {[string compare $systemsubtype "cygwin"] == 0} {
       set system_subtype OC_CYGWIN
    }
    append porth "#define OC_SYSTEM_SUBTYPE $system_subtype\n"

    # The system Tcl type is used to distinguish the Tcl variant
    # being used in the Cygwin environment.
    if {[string compare unix $tcl_platform(platform)]==0} {
       set system_tcltype OC_UNIX
    } elseif {[string compare windows $tcl_platform(platform)==0]} {
       set system_tcltype OC_WINDOWS
    } else {
       error "Unrecognized or unsupported Tcl platform:\
              $tcl_platform(platform)"
    }
    append porth "#define OC_TCL_TYPE $system_tcltype\n"

    set OS [string tolower [string trim $tcl_platform(os)]]
    set OSVERSION [string tolower [string trim $tcl_platform(osVersion)]]
    set OSMAJOR $OSVERSION
    regexp {([^.]*).*} $OSVERSION match OSMAJOR

# Random number generator protoypes
#   NOTE: The default random number generator comes from Oc_Random,
#         which has a max value of Oc_Random::MaxValue().

        append porth {
/* Random number generator.  Default is Oc_Random, which is an   */
/* implementation of the GLIBC random() function with default    */
/* state size.  You can replace this with your own random number */
/* generator if desired.                                         */
/* NB: Any code that uses the default macro settings (involving  */
/*     Oc_Random), must also #include "oc.h" in order to get the */
/*     definition of the Oc_Random class.                        */
#define OMF_SRANDOM(seed)  Oc_Random::Srandom(seed)
#define OMF_RANDOM()       Oc_Random::Random()
#define OMF_RANDOM_MAX     Oc_Random::MaxValue()
}

    if {[string match sunos $OS] && $OSMAJOR < 5} {
        append porth {
/* Signal handler prototype, to work around some non-ANSI header files */
extern "C" {
/* typedef void(*omf_sighandler)(int);  */ /* ANSI version */
typedef void(*omf_sighandler)(int, ...);   /* Not ANSI */
}
}
    } else {
        append porth {
/* Signal handler prototype, to work around some non-ANSI header files */
extern "C" {
typedef void(*omf_sighandler)(int);             /* ANSI version */
/* typedef void(*omf_sighandler)(int, ...); */  /* Not ANSI */
}
}
    }


    # Windows vs. Unix-isms
    if {[string compare $systemtype unix] == 0} {
        append porth {
/* For unix */
/* NICE_DEFAULT is the value passed to nice() inside  */
/* MakeNice().                                        */
#define NICE_DEFAULT 9
/* If your system doesn't have nice, uncomment the next line, or put */
/* in a suitable replacement macro (using, perhaps, setpriority()?). */
/*#define nice(x) 0 */

/* Directory path separator; Unix uses '/', DOS uses '\'. */
#define DIRSEPCHAR '/'
#define DIRSEPSTR  "/"
#define PATHSPLITSTR ":"
}
    }
    if {[string compare $systemtype windows] == 0} {
        append porth {
/* For Windows */
#define rint(x) (floor(x+0.5))
 
/* NICE_DEFAULT is the priority level passed to SetPriorityClass() */
/* inside MakeNice().                                              */
#define NICE_DEFAULT IDLE_PRIORITY_CLASS
#define NICE_THREAD_DEFAULT THREAD_PRIORITY_NORMAL

/* Directory path separator; Unix uses '/', DOS uses '\'. */
#define DIRSEPCHAR '\\'
#define DIRSEPSTR  "\\"
#define PATHSPLITSTR ";"
}
   }

   # Time API
   if {[string compare $systemtype windows] != 0 \
	   || [string compare $systemsubtype winnt] != 0} {
       append porth {
/* Definitions for Oc_TimeVal class */
typedef unsigned long OC_TIMEVAL_TICK_TYPE;
#define OC_TIMEVAL_TO_DOUBLE(x) double(x)
}
   } else {
       append porth {
/* Definitions for Oc_TimeVal class.
 *  NOTE: For most compilers, DWORDLONG is defined in Windows header
 *        files as unsigned __int64.  Adjust as necessary to get
 *        unsigned 64 bit integer.  Alternatively, don't define the
 *        HAS_GETPROCESSTIMES to use the clock() call, and set
 *        OC_TIMEVAL_TICK_TYPE to unsigned long.
 */
#define HAS_GETPROCESSTIMES
typedef DWORDLONG OC_TIMEVAL_TICK_TYPE;}

       if {[catch {
          $config GetValue program_compiler_c++_uint64_to_double
       } does_uint64_to_double]} {
          set does_uint64_to_double 1
       }
       if {$does_uint64_to_double} {
          append porth {
#define OC_TIMEVAL_TO_DOUBLE(x) double(x)
}
       } else {
          append porth {
#define OC_TIMEVAL_TO_DOUBLE(x) (double(__int64(x/2))*2.+double(__int64(x&1)))
}
      }
}
	
   # Start-up entry point
   if {[catch {
      $config GetValue program_compiler_c++_missing_startup_entry
   } missing_startup_entry]} {
      set missing_startup_entry 0
   }
   if {$missing_startup_entry} {
       append porth {
/* System is missing or has non-standard startup entry point.
 * An attempted fix is made in oommf/pkg/oc/oc.cc, which see
 * for details.
 */
#define OC_MISSING_STARTUP_ENTRY 1
}
   }

   # Dump trailer
   append porth "\n#endif /* _OC_PORT_H */"

    # Open output file
    if { [string compare $outfile stdout] == 0 } {
	set fileid stdout
    } else {
	if {[catch {open $outfile w} fileid]} {
	    puts stderr \
		    "Unable to open machine header file $outfile for writing"
	    return 0
	}
    }
    puts $fileid $porth
    if { [string compare $outfile stdout] != 0 } {
	close $fileid
    }
}

proc Oc_MakeTclIndex {dir args} {
    puts "Updating [file join [pwd] $dir tclIndex] ..."
    eval [list auto_mkindex $dir] $args
    global errorCode errorInfo
    set oldDir [pwd]
    cd $dir
    set dir [pwd]
    append index "# The following lines were appended to this file by the\n"
    append index "# command 'Oc_MakeTclIndex' called by pimake .  They\n"
    append index "# provide entries in the auto_index array to support the\n"
    append index "# auto-loading of Oc_Classes.\n\n"
    if {$args == ""} {
        set args *.tcl
    }
    foreach file [eval glob $args] {
        set f ""
        set error [catch {
            set f [open $file]
            while {[gets $f line] >= 0} {
                if {[regexp {^Oc_Class[         ]+([^   ]*)} $line match className]} {
                    append index "set [list auto_index($className)]"
                    # Should this be at global scope?
                    append index " \[list uplevel #0 \[list source \[file join \$dir [list $file]\]\]\]\n"
                }
            }
            close $f
        } msg]
        if {$error} {
            set code $errorCode
            set info $errorInfo
            catch {close $f}
            cd $oldDir
            error $msg $info $code
        }
    }
    set f ""
    set error [catch {
        set f [open tclIndex a]
        puts -nonewline $f $index
        close $f
        cd $oldDir
    } msg]
    if {$error} {
        set code $errorCode
        set info $errorInfo
        catch {close $f}
        cd $oldDir
        error $msg $info $code
    }
}

# Tk bindings are susceptible to reentrancy problems.  For example,
# suppose a user double clicks on a single click binding.  Two
# single clicks events get entered into the event loop.  Now, if
# during the processing of the binding on the first click, the
# event loop were to be re-entered, then this would cause a second
# call to the binding (from the second click) before the first call
# has finished processing.  In particular, one should be aware that
# anytime the binding proc accesses a global variable, that global
# variable may have a trace on it, and that trace may make an update,
# tkwait, or similar request...against the better interests of the
# binding proc.
#   To protect against reentrancy, use the following proc,
# OMF_ThreadSafe.  You will probably want to use a unique "thread_id"
# for each protected code segment.  Ideally, "script" should just be
# a proc call, in which case the name of the proc is a good choice for
# the thread_id.  (Note: The "wait" retry time is in milliseconds.)
#   Sample usage:
#
#    button $w.wFDokbtn -text "OK" \
#           -command "OMF_ThreadSafe OMD_FDokcmd,$w \{OMF_FDokcmd $w\}"
#
# Here I want to protect against reentrancy from the same window, $w,
# so I append that as an additional identifier on thread_id.
#
# NOTE 1: The script is executed at global scope
# NOTE 2: In the current implementation, order of processing of delayed
#         calls may not be preserved.
proc Oc_ThreadSafe { thread_id script {wait 500}} {
    global omfThreadLock errorInfo errorCode
    if { [info exists omfThreadLock($thread_id)] \
            && $omfThreadLock($thread_id)==1 } {
        # Thread locked.  Put script on shelf and try again later
        after $wait [list Oc_ThreadSafe $thread_id $script]
    } else {
        # Otherwise, process the script now
        set omfThreadLock($thread_id) 1
        set errcode [catch { uplevel #0 $script } errmsg]
        set omfThreadLock($thread_id) 0
        if { $errcode != 0 } {
            error $errmsg $errorInfo $errorCode
        }
    }    
}

# Some event bindings, in particular mouse drag events, can generate a
# nearly continuous stream of events in response to user input.  If the
# handler for these events is slow, then these events can pile up in the
# event loop.  It is often the case in this situation that all
# intermediate events can be ignored, and only the last one processed.
# This proc takes a standard bind command, and puts a wrapper around it
# to implement this behavior.  The returned string can be directly bound
# to an event.  NOTE: This routine does not protect against re-entrancy
# on $cmd.  If $cmd services the event loop, then there is the
# possibility of event re-ordering.  In this circumstance, use
# Oc_SafeSkipWrap instead.
set _oc_skipwrap(count) 0
proc Oc_SkipWrap { cmd {wait 3}} {
    global _oc_skipwrap   ;# Wrapper state
    set id $_oc_skipwrap(count) ;# Elt to hold pending event id
    incr _oc_skipwrap(count)
    set _oc_skipwrap($id) {} ;# Initialize
    set newcmd [format {
        global _oc_skipwrap
        after cancel $_oc_skipwrap(%s)
        set _oc_skipwrap(%s) [after %s {%s}]
    } $id $id $wait $cmd]  ;# Create wrapped command
    return $newcmd
}

# Oc_SafeSkipWrap is a version of Oc_SkipWrap that puts a
# semaphore-style lock around the wrapped command.  Before a command is
# executed, a check is made to see if a lock is set.  If so, the command
# goes back onto the event queue.  If not, the lock is set, the command
# is run, and then the lock is unset.  Three points: 1) It is important
# that the command run to completion.  If it should fail, then the lock
# would never be reset and all future commands on this binding would be
# locked out. Because of this, it is not a bad idea to wrap $cmd up
# inside a 'catch'.  2) The locking mechanism protects $cmd from
# reentrancy from _this_ binding.  For example, say $cmd is a call to
# proc foo.  This lock is placed outside of foo, so even if the lock is
# set as a result of this binding, that does not lock out calls to foo
# from a different event binding.  If you need absolute reentrancy
# protection, use Oc_ThreadSafe. 3) Because only 1 event is queued up at
# a time (unprocessed events are thrown away when a new one is
# generated), event order _is_ preserved on execution.
set _oc_safeskipwrap(count) 0
proc Oc_SafeSkipWrap { cmd {wait 3}} {
    global _oc_safeskipwrap   ;# Wrapper state
    set id $_oc_safeskipwrap(count) ;# Elt to hold pending event id
    incr _oc_safeskipwrap(count)
    set _oc_safeskipwrap($id) {} ;# Initialize
    set _oc_safeskipwrap(lock$id) {}
    set newcmd [format {
        global _oc_safeskipwrap
        after cancel $_oc_safeskipwrap(%s)
        set _oc_safeskipwrap(%s) \
                [after %s {Oc_SafeSkipLock %s {%s} %s}]
    } $id $id $wait $id $cmd $wait]  ;# Create wrapped command
    return $newcmd
}
proc Oc_SafeSkipLock { id cmd wait } {
    # Processes ripened events
    global _oc_safeskipwrap
    switch {} $_oc_safeskipwrap(lock$id) {
        set _oc_safeskipwrap(lock$id) 1
        eval $cmd
        set _oc_safeskipwrap(lock$id) {}
    } default {
        set _oc_safeskipwrap($id) \
                [after $wait "Oc_SafeSkipLock $id \{$cmd\} $wait"]
    }
}

proc Oc_OpenUniqueFile {args} {
    array set opts {
	-pfx	""
	-sfx	""
	-sep1	""
	-sep2	""
	-start	0
    }
    array set opts $args
    foreach var {pfx sfx sep1 sep2 start} {
	set $var $opts(-$var)
    }
    if {[file isdirectory $pfx]} {
	append pfx /
	set sep1 ""
    }

    set fn $pfx$sfx
    set code [catch {open $fn {CREAT EXCL RDWR}} handle]
    if {$code == 0} {
	return [list $handle $fn 0]
    }

    set N 4	;# number of digits in serial number.  Edit this
		;# if you need a larger range of unique file names
    set i [expr {$N+1}]
    set max 1
    while {[incr i -1]} {
	append max 0
    }

    if {$start >= $max} {
	set start 0
    }
    set i $start
    set serial [format -%0${N}d $i]
    set fn $pfx$sep1$serial$sep2$sfx
    while {[set code [catch {open $fn {CREAT EXCL RDWR}} handle]]} {
	incr i
	if {$i >= $max} {
	    set i 0
	}
	if {$i == $start} {
	    set msg "Can't open unique file name matching: $pfx$sep1"
	    incr max -1
	    regsub -all 9 $max ? max
	    append msg $max$sep2$sfx
	    error $msg $msg
	}
	set serial [format -%0${N}d $i]
	set fn $pfx$sep1$serial$sep2$sfx
    }
    return [list $handle $fn [incr i]]
}

proc Oc_TempName { {baseprefix {_}} {suffix {}} {basedir {}} } {
    if  {[string match {} $basedir]} {
        Oc_TempFile New f -stem $baseprefix -extension $suffix
    } else {
        Oc_TempFile New f -stem $baseprefix -extension $suffix \
                -directory $basedir
    }
    set retval [$f AbsoluteName]
    $f Claim
    $f Delete
    return $retval
}

proc Oc_StackTrace {} {
    set history {}
    for {set n [expr {[info level]-1}]} {$n>0} {incr n -1} {
        append history "LEVEL $n: [info level $n]\n\n"
    }
    return $history
}

# The rest of the procs in this file are only defined
# conditionally based on whether or not the commands
# defined in the C portion of the Oc extension are available.
# However, the usual indenting rules are not followed because
# we want all these procs to start in the first column so
# they will have entries in tclIndex.

if {[llength [info commands Oc_IgnoreSignal]]} {
proc Oc_IgnoreInteractiveSignals {} {
   catch {Oc_IgnoreSignal  SIGINT}  ;# Ctrl-C generates SIGINT
   catch {Oc_IgnoreSignal  SIGQUIT} ;# Ctrl-\ generates SIGQUIT
   catch {Oc_IgnoreSignal  SIGTSTP} ;# Ctrl-Z generates SIGTSTP

   # 2006-02-17: Disabled the ignoring of the SIGCHLD signal because
   # that appears to interfere with the expectations of the "Condor"
   # batch job control system that is often used to control OOMMF
   # simulations.  This means that programs that launch child processes
   # that are expected to terminate before the parent should set up
   # to reap those children to avoid zombies.
   #
   #catch {Oc_IgnoreSignal  SIGCHLD} ;# Invoke automatic child reaping.
                                    ## This may not work on all platforms.
   # Unix signals list:
   #  1) SIGHUP       2) SIGINT       3) SIGQUIT      4) SIGILL
   #  5) SIGTRAP      6) SIGIOT       7) SIGEMT       8) SIGFPE
   #  9) SIGKILL     10) SIGBUS      11) SIGSEGV     12) SIGSYS
   # 13) SIGPIPE     14) SIGALRM     15) SIGTERM     16) SIGURG
   # 17) SIGSTOP     18) SIGTSTP     19) SIGCONT     20) SIGCHLD
   # 21) SIGTTIN     22) SIGTTOU     23) SIGIO       24) SIGXCPU
   # 25) SIGXFSZ     26) SIGVTALRM   27) SIGPROF     28) SIGWINCH
   # 29) SIGPWR      30) SIGUSR1     31) SIGUSR2

   # Under Windows, only the following set is apparently available:
   #    SIGINT, SIGILL, SIGFPE, SIGSEGV, SIGTERM, SIGBREAK, SIGABRT.
   # The first five match the Unix assignments; the last two are unique
   # to Windows, with defined values SIGBREAK=21 and SIGABRT=22.
   #
   # NOTE: It has not been observed, but it seems possible that the
   # number<->name matching could be system dependent.  Therefore
   # the Oc_IgnoreSignal interface takes a string which is converted
   # inside the C++ code to an integer value, using the macros from
   # the signal.h header file.
}

proc Oc_IgnoreTermLoss {} {
    # Try to ignore the loss of controlling tty.
    catch {Oc_IgnoreSignal  SIGHUP} ;# Closing tty's generate SIGHUP
    catch {Oc_IgnoreSignal SIGPIPE} ;# Broken pipe
    catch {Oc_IgnoreSignal SIGTTIN} ;# Tty input for background process
    catch {Oc_IgnoreSignal SIGTTOU} ;# Tty onput for background process
}
}

# Bugs in some printf "%g" format routines round values out
# of valid double floating point range.  Detect and correct.
# Import eps is smallest double such that 1+eps!=eps.
# Import x is value to check.
# Return is valid double value.
# An error is raised if the import x is wrong format.
proc Oc_FixupBadDoubles { x eps } {
    if {[catch {expr {1.0*$x}}]} {
	# Bad value
	if {[regexp {^([0-9.]+)e([0-9+-]*)$} $x dum man exp]!=1} {
	    return -code error "Bad floating point value: $x"
	}
	set adj [expr {$eps*$man}]
	if {$exp>0} {
	    # Presumably mantissa value is too large
	    set adj [expr {-1*$adj}]
	}
	for {set n 1} {$n<256} {incr n} {
	    set y [expr {$man+($n*$adj)}]
	    append y "e$exp"
	    if {![catch {expr {1.0*$y}} y]} {
		set x $y
		break   ;# Good value found
	    }
	}
    }
    return $x
}
proc Oc_FixupConfigBadDouble { value } {
    set config [Oc_Config RunPlatform]
    set eps [$config GetValue compiletime_dbl_epsilon]
    set x [$config GetValue $value]
    set y [Oc_FixupBadDoubles $x $eps]
    if {[string compare $x $y]!=0} {
	$config SetValue $value $y
    }
}



if {![llength [info commands Oc_SetPanicHeader]]} {
proc Oc_SetPanicHeader {msg} {}
}
