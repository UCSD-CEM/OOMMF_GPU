# FILE: avf2ps.tcl
#
# This file must be evaluated by mmDispSh

package require Oc 1.2.0.4		;# [Oc_OpenUniqueFile]
package require Nb 1.2.0.4
package require Mmdispcmds 1.1.1.0
Oc_ForceStderrDefaultMessage
catch {wm withdraw .}

Oc_Main SetAppName avf2ps
Oc_Main SetVersion 1.2.0.5

Oc_CommandLine Option console {} {}

Oc_CommandLine Option v {
        {level {regexp {^[0-9]+$} $level} {is a decimal integer}}
    } {
        global verbosity;  scan $level %d verbosity
} {Verbosity level (default = 1)}
set verbosity 1

Oc_CommandLine Option f {} {
        global forceOutput; set forceOutput 1
} {Force overwrite of output files}
set forceOutput 0

# Build list of configuration files.  Note: The local/avf2ps.config
# file, if it exists, is automatically source by the main avf2ps.config
# file.
set configFiles [list [file join [file dirname \
        [Oc_DirectPathname [info script]]] avf2ps.config]]
Oc_CommandLine Option config {file} {
        global configFiles; lappend configFiles $file
} "Append file to list of configuration files"

Oc_CommandLine Option ipat {
        {pattern {} {is in glob-style}}
    } {
        global inputPattern; set inputPattern $pattern
} {Pattern specifying input files}

Oc_CommandLine Option opatexp {
        {regexp {} {is a regular expression}}
    } {
        global opatexp;  set opatexp $regexp
} {Pattern: part of input filename to replace}
set opatexp {(\.[^.]?[^.]?[^.]?(.gz)?$|$)}

Oc_CommandLine Option opatsub {sub} {
        global opatsub;  set opatsub $sub
} {Replacement in input filename to get output filename}

Oc_CommandLine Option filter {
        {pipeline {} {is an output pipeline}}
    } {
        global filter;  set filter $pipeline
} {Post-processing programs to apply to output}

Oc_CommandLine Option [Oc_CommandLine Switch] {
        {{file list} {} {Input vector field file(s)}}
    } {
        global infile; set infile $file
} {End of options; next argument is file}
Oc_CommandLine Parse $argv

# Get the configuration settings from config file list
foreach c $configFiles {
    source $c
}

# Turn -ipat pattern into list of input files
set inGlob [list]
if {[info exists inputPattern]} {
    set inGlob [lsort -dictionary [glob -nocomplain -- $inputPattern]]
} else {
    set inputPattern ""
}

# Set default output filename substitution
if {![info exists opatsub]} {
    set opatsub .eps
}

if {$verbosity >= 3} {
    puts stderr "##############"
    puts stderr "Input  files: $infile $inGlob"
    puts stderr "##############"
}
if {$verbosity >= 2} {
    puts stderr "ipat: $inputPattern"
    puts stderr "opatexp: $opatexp"
    puts stderr "opatsub: $opatsub"
    if {[info exists filter]} {
        puts stderr "Filter command: $filter"
    }
    set centerstr {}
    if {[info exists plot_config(misc,relcenterpt)]} {
	append centerstr "\n Relative center pt:"
	append centerstr " $plot_config(misc,relcenterpt)"
    }
    if {[info exists plot_config(misc,centerpt)]} {
	append centerstr "\n Absolute center pt:"
	append centerstr " $plot_config(misc,centerpt)"
    }
    puts stderr "Arrow Configuration---
 Display: $plot_config(arrow,status)
 Colormap name: $plot_config(arrow,colormap)
 Colormap size: $plot_config(arrow,colorcount)
 Quantity: $plot_config(arrow,quantity)
 Color phase: $plot_config(arrow,colorphase)
 Color reverse: $plot_config(arrow,colorreverse)
 Autosample: $plot_config(arrow,autosample)
 Subsample: $plot_config(arrow,subsample)
 Size: $plot_config(arrow,size)
 Antialias: $plot_config(arrow,antialias)"
    puts stderr "Pixel Configuration---
 Display: $plot_config(pixel,status)
 Colormap name: $plot_config(pixel,colormap)
 Colormap size: $plot_config(pixel,colorcount)
 Opaque: $plot_config(pixel,opaque)
 Quantity: $plot_config(pixel,quantity)
 Color phase: $plot_config(pixel,colorphase)
 Color reverse: $plot_config(pixel,colorreverse)
 Autosample: $plot_config(pixel,autosample)
 Subsample: $plot_config(pixel,subsample)
 Size: $plot_config(pixel,size)"
    puts stderr "Misc Configuration----
 Background color: $plot_config(misc,background)
 Draw boundary: $plot_config(misc,drawboundary)
 Margin: $plot_config(misc,margin)
 Dimensions: $plot_config(misc,width) x $plot_config(misc,height)
 Zoom: $plot_config(misc,zoom)
 Crop: $plot_config(misc,crop)
 CropToView: $print_config(croptoview)
 Rotation: $plot_config(misc,rotation)
 Datascale: $plot_config(misc,datascale)$centerstr"
    set axis [string index $plot_config(viewaxis) end]
    set centerstr {}
    if {[info exists plot_config(viewaxis,center)]} {
	append centerstr "\n    Axis center:"
	append centerstr " $plot_config(viewaxis,center)"
    }
    puts stderr "View axis Configuration----
 View axis: $plot_config(viewaxis)$centerstr
    Arrow  span: $plot_config(viewaxis,${axis}arrowspan)
    Pixel  span: $plot_config(viewaxis,${axis}pixelspan)"
}

########################################################################
### Support variables and procs ########################################

# Initialize view_transform array, which is used for out-of-plane
# rotations.
InitViewTransform

proc ConvertToEng { z sigfigs } {
    # Note: This routine differs from the similarly named routine
    #  in mmdisp.tcl in that trailing zeros in the mantissa are
    #  truncated.
    set sigfigs [expr int(round($sigfigs))]
    if {$sigfigs<1} { set sigfigs 1 }
    set z [format "%.*g" $sigfigs $z] ;# Remove extra precision

    if {$z==0} {
        return 0
    }
    set exp1 [expr {int(floor(log10(abs($z))))}]
    set exp2 [expr {int(round(3*floor($exp1/3.)))}]
    set prec [expr int(round($sigfigs-($exp1-$exp2+1)))]
    if { $prec<0 } { set prec 0 }
    set mantissa [format "%.*f" $prec [expr {$z*pow(10.,-1*$exp2)}]]
    regsub -- {0*$} $mantissa {} mantissa
    regsub -- {\.$} $mantissa {} mantissa
    if {$exp2==0} {return $mantissa}
    return [format "%se%d" $mantissa $exp2]
}

proc ReadFile { filename } {
    global plot_config print_config view_transform verbosity

    if {![file exists $filename]} {
        return -code error "Input file \"$filename\" does not exist."
    }
    if {![file readable $filename]} {
        return -code error "Input file \"$filename\" is not readable."
    }

    # Apply filter (based on filename extension), if applicable.
    set tempfile [Nb_InputFilter FilterFile $filename decompress ovf]
    if { [string match {} $tempfile] } {
        # No filter match
        set workname $filename
    } else {
        # Filter match
        set workname $tempfile
    }

    # Read file
    set viewaxis $plot_config(viewaxis)
    if {[string length $viewaxis] == 1} {
        set viewaxis "+$viewaxis"
        set plot_config(viewaxis) $viewaxis
    }
    set rot $plot_config(misc,rotation)
    set rot [expr {(round($rot/90.) % 4)*90}]
    set width $plot_config(misc,width)
    set height $plot_config(misc,height)
    set zoom $plot_config(misc,zoom)
    ChangeMesh $workname $width $height $rot $zoom

    if {$verbosity >= 1} {
        foreach i [GetMeshRange] j {xmin ymin zmin xmax ymax zmax} {
            set $j [ConvertToEng $i 6]
        }
        puts stderr \
           "Bounding box: ($xmin,$ymin,$zmin) x ($xmax,$ymax,$zmax)"
    }

    # Delete tempfile, if any
    if { ![string match {} $tempfile] } {
        catch { file delete $tempfile }
    }

    # Adjust data scaling
    SetDataValueScaling $plot_config(misc,datascale)

    # Mesh loads in with +z forward, so transform
    # mesh to show desired axis
    CopyMesh 0 0 0 $view_transform(+z,$viewaxis)
    SelectActiveMesh 0
}

########################################################################

# Loop through input files
foreach in [concat $infile $inGlob] {

    if {[regsub -- $opatexp $in $opatsub out] != 1} {
        set out $in$opatsub
    }
    if {!$forceOutput} {
	set next 0
        foreach {outChan out next} \
                [Oc_OpenUniqueFile -start $next -pfx $out -sep1 -] break
    } else {
        set outChan [open $out w]
    }
    if {[info exists filter]} {
	# Note: $filter might itself be a pipeline, or a command
	# with arguments.  Either way, we need the open eval to
	# interpret $filter as multiple command words, i.e., we
	# don't want to wrap $filter up inside a list.
        set outChan [open "| 2>@stderr $filter >@ $outChan" w]
    }
    if {$verbosity >= 1} {
        puts -nonewline stderr "$in --> "
        if {[info exists filter]} {
            puts -nonewline stderr "$filter --> "
        }
        puts stderr $out
    }
    fconfigure $outChan -buffersize 10000

    # Read input file into frame
    ReadFile $in

    # Dump PostScript
    PSWriteMesh $outChan

    if {[catch {close $outChan} msg]} {
        return -code error $msg
    }
}

exit 0
