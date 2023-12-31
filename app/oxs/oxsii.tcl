# FILE: oxsii.tcl
#
# The OOMMF eXtensible Solver Interative Interface

# Support libraries
package require Oc 1.1
package require Oxs 1.2
package require Net 1.2.0.3

Oc_IgnoreTermLoss  ;# Try to keep going, even if controlling terminal
## goes down.

# Application description boilerplate
Oc_Main SetAppName Oxsii
Oc_Main SetVersion 1.2.0.5
regexp \\\044Date:(.*)\\\044 {$Date: 2012-09-25 17:11:59 $} _ date
Oc_Main SetDate [string trim $date]
Oc_Main SetAuthor [Oc_Person Lookup dgp]
Oc_Main SetHelpURL [Oc_Url FromFilename [file join [file dirname \
        [file dirname [file dirname [Oc_DirectPathname [info \
        script]]]]] doc userguide userguide\
        OOMMF_eXtensible_Solver_Int.html]]

# Command line options
Oc_CommandLine ActivateOptionSet Net
Oc_CommandLine Option restart {
	{flag {expr {$flag==0 || $flag==1}} {= 0 (default) or 1}}
    } {
	global restart_flag; set restart_flag $flag
} {1 => use <basename>.restart file to restart simulation}
set restart_flag 0
Oxs_SetRestartFlag $restart_flag
trace variable restart_flag w {Oxs_SetRestartFlag $restart_flag ;# }

Oc_CommandLine Option nocrccheck {
	{flag {expr {$flag==0 || $flag==1}} {= 0 (default) or 1}}
    } {
	global nocrccheck_flag; set nocrccheck_flag $flag
} {1 => Disable CRC check on simulation restarts}
set nocrccheck_flag 0
Oxs_SetRestartCrcCheck [expr {!$nocrccheck_flag}]
trace variable nocrccheck_flag w \
	{Oxs_SetRestartCrcCheck [expr {!$nocrccheck_flag}] ;# }

Oc_CommandLine Option parameters \
{{params {expr {([llength $params]%2)==0}} { is a list of name+value pairs}}} \
{global MIF_params; set MIF_params $params} \
{Set MIF file parameters}
set MIF_params {}

Oc_CommandLine Option exitondone {
	{flag {expr {![catch {expr {$flag && $flag}}]}} {= 0 (default) or 1}}
    } {
	global exitondone; set exitondone $flag
} {1 => Exit when problem solved}
set exitondone 0

Oc_CommandLine Option pause {
	{flag {expr {![catch {expr {$flag && $flag}}]}} {= 0 or 1 (default)}}
    } {
	global autorun_pause; set autorun_pause $flag
} {1 => Pause after autorun initialization}
#cuiwl
set autorun_pause 0

Oc_CommandLine Option nice {
	{flag {expr {![catch {expr {$flag && $flag}}]}} {= 0 or 1 (default)}}
    } {
	global nice; set nice $flag
} {1 => Drop priority after starting}
set nice 1

Oc_CommandLine Option loglevel {
      {level {expr {[regexp {^[0-9]+$} $level]}}}
   } {
      global loglevel;  set loglevel $level
} {Level of log detail to oommf/boxsi.errors (default is 1)}
set loglevel 1

# Multi-thread support
if {[Oc_HaveThreads]} {
   set threadcount_request [Oc_GetMaxThreadCount]
   Oc_CommandLine Option threads {
      {number {expr {[regexp {^[0-9]+$} $number] && $number>0}}}
   } {
      global threadcount_request;  set threadcount_request $number
   } [subst {Number of concurrent threads (default is $threadcount_request)}]
} else {
   set threadcount_request 1  ;# Safety
}

# NUMA (non-uniform memory access) support
set numanode_request none
if {[Oc_NumaAvailable]} {
   if {[info exists env(OOMMF_NUMANODES)]} {
      set nodes $env(OOMMF_NUMANODES)
      if {![regexp {^([0-9 ,]*|auto|none)$} $nodes]} {
         puts stderr "\n************************************************"
         puts stderr "ERROR: Bad environment variable setting:\
                   OOMMF_NUMANODES=$nodes"
         puts stderr "   Overriding to \"$numanode_request\""
         puts stderr "************************************************"
      } else {
         set numanode_request $nodes
      }
   }
   Oc_CommandLine Option numanodes {
      {nodes {regexp {^([0-9 ,]*|auto|none)$} $nodes}}
   } {
      global numanode_request;
      set numanode_request $nodes
   } [subst {NUMA memory and run nodes (or "auto" or "none")\
                (default is "$numanode_request")}]
}

set autorun 0
Oc_CommandLine Option [Oc_CommandLine Switch] {
    {{filename {optional}} {} {
	Optional input MIF 2.1 problem file to load and run.}}
} {
    if {![string match {} $filename]} {
	global problem autorun
	set autorun 1
	set problem $filename
    }
} {End of options}

##########################################################################
# Parse commandline and initialize threading
##########################################################################
Oc_CommandLine Parse $argv

proc SetupThreads {} {
   if {[Oc_HaveThreads]} {
      global threadcount_request
      if {$threadcount_request<1} {set threadcount_request 1}
      Oc_SetMaxThreadCount $threadcount_request
      set aboutinfo "Number of threads: $threadcount_request"
   } else {
      set aboutinfo "Single threaded build"
   }

   if {[Oc_HaveThreads] && [Oc_NumaAvailable]} {
      global numanode_request
      if {[string match auto $numanode_request]} {
         set nodes {}
      } elseif {[string match none $numanode_request]} {
         set nodes none
      } else {
         set nodes [split $numanode_request " ,"]
      }
      if {![string match "none" $nodes]} {
         Oc_NumaInit $threadcount_request $nodes
      } else {
         Oc_NumaDisable
      }
      append aboutinfo "\nNUMA: $numanode_request"
   }

   Oc_Main SetExtraInfo $aboutinfo
   global update_extra_info
   set update_extra_info $aboutinfo
}

# Initialize extra "About" info.  This may be changed
# by future calls to SetupThreads
if {[Oc_HaveThreads]} {
   set aboutinfo "Number of threads: $threadcount_request"
} else {
   set aboutinfo "Single threaded build"
}
if {[Oc_NumaAvailable]} {
   append aboutinfo "\nNUMA: $numanode_request"
}
Oc_Main SetExtraInfo $aboutinfo
set update_extra_info $aboutinfo
unset aboutinfo

##########################################################################
# Define the GUI of this app to be displayed remotely by clients of the
# Net_GeneralInterface protocol.  Return $gui in response to the
# GetGui message.
##########################################################################
set gui {
    package require Oc 1.1
    package require Tk
    package require Ow 1.2.0.4
    wm withdraw .

   Oc_Log SetLogHandler [list Ow_BkgdLogger Log] panic Oc_Log
   Oc_Log SetLogHandler [list Ow_BkgdLogger Log] error Oc_Log
   Oc_Log SetLogHandler [list Ow_BkgdLogger Log] warning Oc_Log
   Oc_Log SetLogHandler [list Ow_BkgdLogger Log] info Oc_Log
   Oc_Log SetLogHandler [list Ow_BkgdLogger Log] panic
   Oc_Log SetLogHandler [list Ow_BkgdLogger Log] error
   Oc_Log SetLogHandler [list Ow_BkgdLogger Log] warning
   Oc_Log SetLogHandler [list Ow_BkgdLogger Log] info
}
append gui "[list Oc_Main SetAppName [Oc_Main GetAppName]]\n"
append gui "[list Oc_Main SetVersion [Oc_Main GetVersion]]\n"
append gui "[list Oc_Main SetExtraInfo [Oc_Main GetExtraInfo]]\n"
append gui {
share update_extra_info
trace variable update_extra_info w { Oc_Main SetExtraInfo $update_extra_info ;# }
}
append gui "[list Oc_Main SetPid [pid]]\n"
append gui {

regexp \\\044Date:(.*)\\\044 {$Date: 2012-09-25 17:11:59 $} _ date
Oc_Main SetDate [string trim $date]

# This won't cross different OOMMF installations nicely
Oc_Main SetAuthor [Oc_Person Lookup donahue]

}

# This might not cross nicely either:
#   Originally the filename of the Oxsii help HTML documentation section
# was constructed on the solver side and passed to the interface side in
# the gui string.  However, if the solver and interface are running on
# two different machines, then there are potentially two problems with
# that setup.  The Help menu item launches a documentation viewer on the
# interface side.  If the solver and interface machines don't share a
# common filesystem, then it is likely that the Help filename passed
# across will be wrong; in this case mmHelp will display a File not
# found Retrieval Error.  Even worse, the 'Oc_Url FromFilename' call
# raises an error if the filename passed to it does not look like an an
# absolute pathname.  This will happen if, for example, one of the
# machines is running Windows and the other is Unix, since absolute
# paths on Windows look like 'C:/foo/bar.html' but Unix wants
# '/foo/bar.html'.  If this error is not caught, then the interface
# won't even come up.  Definitely not very user friendly.
#   To bypass these problems, the code below uses 'Oc_Main
# GetOOMMFRootDir' to construct the Help filename on the interface side.
# Only down side to this that I see is if the interface is running out
# of a different OOMMF version than the solver; but in that case it
# would not be clear which documentation to display anyway, since we
# don't know if the user is asking for help on the interface or on the
# solver internals.  Regardless, we don't currently have a general way
# to make the solver-side documentation available, so we might as well
# serve up the interface-side docs.
#   A catch is put around this code is just to make doubly sure that any
# problems with setting up the Help URL don't prevent the user
# interfacing from being displayed.  If the Help URL is not explicitly
# set, then as a fallback Oc_Main returns a pointer to the documentation
# front page, built by indexing off the directory holding the
# interface-side Oc_Main script.  Note that this is the same method used
# by 'Oc_Main GetOOMMFRootDir'.  -mjd, 29-July-2002
append gui {
catch {
   set oxsii_doc_section [list [file join \
                          [Oc_Main GetOOMMFRootDir] doc userguide userguide \
                          OOMMF_eXtensible_Solver_Int.html]]
   Oc_Main SetHelpURL [Oc_Url FromFilename $oxsii_doc_section]
   proc SetOidCallback {code result args} {
      if {$code == 0} {Oc_Main SetOid $result}
      rename SetOidCallback {}
   }
   remote -callback SetOidCallback serveroid
   if {[Ow_IsAqua]} {
      # Add some padding to allow space for Aqua window resize
      # control in lower righthand corner
      . configure -relief flat -borderwidth 15
   }
}
}

append gui {
proc ConsoleInfo {m t s} {
   set timestamp [clock format [clock seconds] -format %T]
   puts stderr "\[$timestamp\] $m"
}
Oc_Log SetLogHandler [list ConsoleInfo] info
Oc_Log AddType infolog ;# Record in log file only

set menubar .mb
foreach {fmenu omenu hmenu} [Ow_MakeMenubar . $menubar File Options Help] break
$fmenu add command -label "Load..." \
	-command [list LoadButton $fmenu "Load..."] -underline 0
$fmenu add command -label "Show Console" -command { console show } -underline 0
$fmenu add command -label "Close Interface" -command closeGui -underline 0
$fmenu add separator
$fmenu add command -label "Exit [Oc_Main GetAppName]" -command exit -underline 1
$omenu add command -label "Clear Schedule" -underline 0 -command ClearSchedule
$omenu add separator
$omenu add checkbutton -label "Restart flag" -underline 0 \
    -variable restart_flag
share restart_flag
share threadcount_request  ;# Set from command line or File|Load dialog
share numanode_request     ;# Ditto
share MIF_params

Ow_StdHelpMenu $hmenu
set SmartDialogs 1
proc LoadButton { btn item } {
    if { [string match disabled [$btn entrycget $item -state]] } {
        return
    }
    # NOTE: File checking is disabled at the widget level
    #  (-file_must_exist 0 -dir_must_exist 0) in order to allow
    #  the oxsii interface to be used to load problems across a
    #  port on machines with disparate filesystems. -mjd, 5-Nov-2001
    global SmartDialogs
    Ow_FileDlg New dialog -callback LoadCallback \
	    -dialog_title "[Oc_Main GetTitle]: Load Problem" \
            -allow_browse 1 \
            -optionbox_setup LoadOptionBoxSetup \
            -selection_title "Load MIF File..." \
            -select_action_id LOAD \
	    -filter [list *.mif *.mif2] \
	    -dir_must_exist 0 \
	    -file_must_exist 0 \
	    -menu_data [list $btn $item] \
	    -optbox_position bottom \
	    -smart_menu $SmartDialogs

   # Set icon
   Ow_SetIcon [$dialog Cget -winpath]
}

proc LoadOptionBoxSetup { widget frame } {
    global restart_flag loadopt_restart_flag
    global MIF_params loadopt_params_widget

    frame $frame.top

    set loadopt_restart_flag $restart_flag
    checkbutton $frame.top.restart -text "Restart" \
       -variable loadopt_restart_flag \
       -relief flat -padx 1m -pady 1m \
       -offvalue 0 -onvalue 1
    # Don't save restart value directly in restart_flag, but wait until
    # the user selects Ok/Apply.  If the user selects Close, then
    # restart_flag is not set from loadopt_restart_flag, but remains at
    # whatever its previous value was.

    set dothreads [Oc_HaveThreads]
    set donuma [Oc_NumaAvailable]
    if {$dothreads} {
       global loadopt_threadcount threadcount_request
       set loadopt_threadcount $threadcount_request
       Ow_EntryBox New loadopt_threads_widget $frame.top.threads \
          -label "Threads:" -autoundo 0  \
          -valuewidth 4 -valuetype posint \
          -coloredits 0 -writethrough 1 \
          -outer_frame_options "-bd 0" \
          -variable loadopt_threadcount
       if {$donuma} {
          global loadopt_numanodes numanode_request
          set loadopt_numanodes $numanode_request
          Ow_EntryBox New loadopt_numanodes_widget $frame.top.numanodes \
             -label "NUMA:" -autoundo 0  \
             -valuewidth 8 -valuetype text \
             -coloredits 0 -writethrough 1 \
             -outer_frame_options "-bd 0" \
             -variable loadopt_numanodes
       }
    }

    Ow_EntryBox New loadopt_params_widget $frame.params \
       -label "Params:" -autoundo 0  \
       -valuewidth 0 -valuetype text \
       -coloredits 0 -writethrough 1 \
       -outer_frame_options "-bd 0"
    $loadopt_params_widget Set $MIF_params
    # Don't tie MIF_params directly to loadopt_params_widget, because
    # MIF_params is "shared" with the backend; if a user edits the
    # Params entry too fast for the backend to keep up (which can easily
    # happen if the backend is busy running a simulation and only
    # updating the event loop every few seconds), then the editing
    # cursor may get reset to the end of the entry at seemingly
    # arbitrary intervals.  Instead, just copy the value from the widget
    # into MIF_params when the user selects OK/Apply.  This has the
    # added benefit of allowing the user to close and reopen the dialog
    # to reset the Params value to the preceding (unsaved) value.

    pack $frame.top -fill x
    pack $frame.top.restart -side left -anchor w
    if {$dothreads} {
       if {!$donuma} {
          pack $frame.top.threads -side right
       } else {
          pack $frame.top.numanodes $frame.top.threads \
             -side right -padx 5
       }
    }
    pack $frame.params -fill x -anchor w

    if {[Ow_IsAqua]} {
       # Add some padding to allow space for Aqua window resize
       # control in lower righthand corner
       $frame configure -bd 4 -relief ridge
       grid columnconfigure $frame 3 -minsize 10
    } else {
       $frame configure -relief raised
    }
}

proc LoadCallback { widget actionid args } {
    if {[string match DELETE $actionid]} {
	# This really is the way it has to be!  The quoting on $args
	# is just plain weird.
        eval [join $args]
        return
    }
    if {[string match LOAD $actionid]} {
       global problem
       global restart_flag loadopt_restart_flag
       global threadcount_request loadopt_threadcount
       global numanode_request loadopt_numanodes
       global MIF_params loadopt_params_widget
       set restart_flag $loadopt_restart_flag
       if {[info exists loadopt_threadcount]} {
          set threadcount_request $loadopt_threadcount
       }
       if {[info exists loadopt_numanodes]} {
          set numanode_request $loadopt_numanodes
       }
       set MIF_params [$loadopt_params_widget ReadEntryString]
       set problem [$widget GetFilename] ;# NOTE: Variable "problem"
       ## is a "shared" variable with the backend.  Writing to this
       ## variable triggers the loading of a new problem, so it
       ## should be written last.
    } else {
        return "ERROR (proc LoadCallBack): Invalid actionid: $actionid"
    }
    # The following [return] must be here so the dialog will close
    return
}

trace variable problem w { Ow_BkgdLogger Reset ;# }

set menuwidth [Ow_GuessMenuWidth $menubar]
set brace [canvas .brace -width $menuwidth -height 0 -borderwidth 0 \
        -highlightthickness 0]
pack $brace -side top

if {[package vcompare [package provide Tk] 8] >= 0 \
        && [string match windows $tcl_platform(platform)]} {
    # Windows doesn't size Tcl 8.0 menubar cleanly
    Oc_DisableAutoSize .
    wm geometry . "${menuwidth}x0"
    update
    wm geometry . {}
    Oc_EnableAutoSize .
    update
}

share problem

set interface_state disabled
share interface_state

set bf [frame .buttons]
pack [button $bf.reload -text Reload -command {set problem $problem} \
	-state $interface_state] -fill x -side left
pack [button $bf.reset -text Reset -command {set status Initializing...} \
	-state $interface_state] -fill x -side left
pack [button $bf.run -text Run -command {set status Run} \
	-state $interface_state] -fill x -side left
pack [button $bf.relax -text Relax -command {set status Relax} \
	-state $interface_state] -fill x -side left
pack [button $bf.step -text Step -command {set status Step} \
	-state $interface_state] -fill x -side left
pack [button $bf.pause -text Pause -command {set status Pause} \
	-state $interface_state] -fill x -side left

pack $bf -side top -fill x

set probframe [frame .problem ]
pack [label $probframe.l -text Problem: -anchor w] -side left
pack [label $probframe.val -textvariable problem -anchor e] -side left
pack $probframe -side top -fill x

set mf [frame .monitor]

share status
set statframe [frame $mf.status]
pack [label $statframe.l -text " Status:" -anchor w] -side left
pack [label $statframe.val -textvariable status -anchor e] -side left
pack $statframe -side top -fill x

share stagerequest
share number_of_stages
if {![info exists number_of_stages]} {
    set number_of_stages 100 ;# Dummy init value
}
Ow_EntryScale New stageES $mf.stage \
	-label "  Stage:" \
	-variable stagerequest \
	-valuewidth 4 \
	-valuetype posint \
	-rangemax [expr {$number_of_stages-1}] \
	-scalestep 1 \
	-outer_frame_options "-bd 0 -relief flat" \
	-foreground Black -disabledforeground #a3a3a3 \
	-state $interface_state
pack [$stageES Cget -winpath] -side top -fill x

pack $mf -side top -fill x

trace variable number_of_stages w [format {
    %s Configure -rangemax [uplevel #0 expr {$number_of_stages-1}]
    ;# } $stageES]

trace variable interface_state w [format {
    foreach _ [winfo children .buttons] {
	if {[string match Reload [$_ cget -text]] \
		&& [info exists problem] \
		&& ![string match {} $problem]} {
	    $_ configure -state normal
	} else {
	    $_ configure -state [uplevel #0 set interface_state];
	}
    }
    %s Configure -state [uplevel #0 set interface_state] ;# } $stageES]

set oframe [frame .sopanel]

set opframe [frame $oframe.outputs]
pack [label $opframe.l -text " Output " -relief groove] -side top -fill x
Ow_ListBox New oplb $opframe.lb -height 4 -variable oplbsel
# Note: At this time (Feb-2006), tk list boxes on Mac OS X/Darwin don't
# scroll properly if the height is smaller than 4.
pack [$oplb Cget -winpath] -side top -fill both -expand 1
share opAssocList
trace variable opAssocList w { uplevel #0 {
    catch {unset opArray}
    array set opArray $opAssocList
    set opList [lsort [array names opArray]]
    UpdateSchedule } ;# }
trace variable opList w {
    $oplb Remove 0 end
    eval [list $oplb Insert 0] $opList ;# }
trace variable oplbsel w {uplevel #0 {
    trace vdelete destList w {SetDestinationList; UpdateSchedule ;#}
    if {[llength $oplbsel]} {
	upvar 0 $opArray([lindex $oplbsel 0])List destList
	SetDestinationList
    }
    trace variable destList w {SetDestinationList; UpdateSchedule ;#}
} ;# }
pack $opframe -side left -fill both -expand 1

# The destination lists (Find a way to construct dynamically, yet
# efficiently):
share vectorFieldList
share scalarFieldList
share DataTableList
set vectorFieldList [list]
set scalarFieldList [list]
set DataTableList [list]

set dframe [frame $oframe.destinations]
pack [label $dframe.l -text " Destination " -relief groove] -side top -fill x
Ow_ListBox New dlb $dframe.lb -height 4 -variable dlbsel
# Note: At this time (Feb-2006), tk list boxes on Mac OS X/Darwin don't
# scroll properly if the height is smaller than 4.
pack [$dlb Cget -winpath] -side top -fill both -expand 1
proc SetDestinationList {} {
    global dlb destList
    $dlb Remove 0 end
    eval [list $dlb Insert 0] $destList
}
trace variable destList w {SetDestinationList; UpdateSchedule ;#}
pack $dframe -side left -fill both -expand 1
	
set sframe [frame $oframe.schedule]
grid [label $sframe.l -text Schedule -relief groove] - -sticky new
grid columnconfigure $sframe 0 -pad 15
grid columnconfigure $sframe 1 -weight 1
grid [button $sframe.send -command \
	{remote Oxs_Output Send [lindex $oplbsel 0] [lindex $dlbsel 0]} \
	-text Send] - -sticky new
#
# FOR NOW JUST HACK IN THE EVENTS WE SUPPORT
#	eventually this may be driver-dependent
set events [list Step Stage]
set schedwidgets [list $sframe.send]
foreach event $events {
    set active [checkbutton $sframe.a$event -text $event -anchor w \
	    -variable Schedule---activeA($event)]
    $active configure -command [subst -nocommands \
		{Schedule Active [$active cget -variable] $event}]
    Ow_EntryBox New frequency $sframe.f$event -label every \
	    -autoundo 0 -valuewidth 4 \
	    -variable Schedule---frequencyA($event) \
	    -callback [list EntryCallback $event] \
	    -foreground Black -disabledforeground #a3a3a3 \
	    -valuetype posint -coloredits 1 -writethrough 0 \
	    -outer_frame_options "-bd 0"
    grid $active [$frequency Cget -winpath] -sticky nw
    lappend schedwidgets $active $frequency
}

grid rowconfigure $sframe [expr {[lindex [grid size $sframe] 1] - 1}] -weight 1
pack $sframe -side left -fill y

proc EntryCallback {e w args} {
    upvar #0 [$w Cget -variable] var
    set var [$w Cget -value]
    Schedule Frequency [$w Cget -variable] $e
}
proc Schedule {x v e} {
    global oplbsel dlbsel
    upvar #0 $v value
    remote Oxs_Schedule Set [lindex $oplbsel 0] [lindex $dlbsel 0] $x $e $value
}

trace variable oplbsel w "UpdateSchedule ;#"
trace variable dlbsel w "UpdateSchedule ;#"
Oc_EventHandler Bindtags UpdateSchedule UpdateSchedule
proc UpdateSchedule {} {
    global oplbsel dlbsel schedwidgets opArray
    Oc_EventHandler Generate UpdateSchedule Reset
    set os [lindex $oplbsel 0]
    set ds [lindex $dlbsel 0]
    set state disabled
    if {[info exists opArray($os)]} {
	upvar #0 $opArray($os)List destList
	if {[lsearch -exact $destList $ds] >= 0} {
	    set state normal
	}
    }
    foreach _ $schedwidgets {
	if {[string match *EntryBox* $_]} {
#Oc_Log Log "$_ Configure -state $state" warning
	    $_ Configure -state $state
	} else {
	    $_ configure -state $state
	}
    }
    if {[string compare normal $state]} {return}
    regsub -all {:+} $os : os
    regsub -all "\[ \r\t\n]+" $os _ os
    regsub -all {:+} $ds : ds
    regsub -all "\[ \r\t\n]+" $ds _ ds
    global Schedule-$os-$ds-active
    global Schedule-$os-$ds-frequency
    global Schedule---frequencyA

    if {![info exists Schedule-$os-$ds-active]} {
	share Schedule-$os-$ds-active
	trace variable Schedule-$os-$ds-active w [subst { uplevel #0 {
	    catch {[list unset Schedule-$os-$ds-activeA]}
	    [list array set Schedule-$os-$ds-activeA] \[[list set \
	    Schedule-$os-$ds-active]] } ;# }]
    }

    if {![info exists Schedule-$os-$ds-frequency]} {
	share Schedule-$os-$ds-frequency
    } else {
	array set Schedule---frequencyA [set Schedule-$os-$ds-frequency]
    }

    # Reconfigure Schedule widgets
    foreach _ $schedwidgets {
	if {[regexp {.+[.]a([^.]+)$} $_ -> e]} {
	    $_ configure -variable Schedule-$os-$ds-activeA($e)
	}
    }

    # When entry boxes commit - write change to shared variable.
    trace variable Schedule---frequencyA w [subst { uplevel #0 {
	    [list set Schedule-$os-$ds-frequency] \
	    \[[list array get Schedule---frequencyA]]} ;# }]

    Oc_EventHandler New _ UpdateSchedule Reset \
	    [subst {trace vdelete Schedule---frequencyA w { uplevel #0 {
	    [list set Schedule-$os-$ds-frequency] \
	    \[[list array get Schedule---frequencyA]]} ;# }}] \
	    -oneshot 1

    # When shared variable changes - write change to entry box
    trace variable Schedule-$os-$ds-frequency w [subst { uplevel #0 {
	    [list array set Schedule---frequencyA] \[[list set \
	    Schedule-$os-$ds-frequency]] } ;# }]

    Oc_EventHandler New _ UpdateSchedule Reset [subst \
	    {[list trace vdelete Schedule-$os-$ds-frequency w] { uplevel #0 {
	    [list array set Schedule---frequencyA] \[[list set \
	    Schedule-$os-$ds-frequency]] } ;# }}] \
	    -oneshot 1
}
UpdateSchedule
proc ClearSchedule {} {
    global opList opArray

    # Loop over all the outputs and destinations
    foreach o $opList {
	upvar #0 $opArray($o)List destList
	foreach d $destList {
	    remote Oxs_Schedule Set $o $d Active Stage 0
	    remote Oxs_Schedule Set $o $d Active Step 0
	}
    }
}

pack $oframe -side top -fill both -expand 1

update idletasks ;# Help interface display at natural size

}
##########################################################################
# End of GUI script.
##########################################################################

##########################################################################
# Handle Tk
##########################################################################
if {[Oc_Main HasTk]} {
    wm withdraw .        ;# "." is just an empty window.
    package require Ow
    Ow_SetIcon .

    # Evaluate $gui?  (in a slave interp?) so that when this app
    # is run with Tk it presents locally the same interface it
    # otherwise exports to be displayed remotely?
}

# Set up to write log messages to oommf/oxsii.errors.
FileLogger SetFile [file join [Oc_Main GetOOMMFRootDir] oxsii.errors]
Oc_Log SetLogHandler [list FileLogger Log] panic Oc_Log
Oc_Log SetLogHandler [list FileLogger Log] error Oc_Log
Oc_Log SetLogHandler [list FileLogger Log] warning Oc_Log
Oc_Log SetLogHandler [list FileLogger Log] info Oc_Log
Oc_Log SetLogHandler [list FileLogger Log] panic
Oc_Log SetLogHandler [list FileLogger Log] error
Oc_Log SetLogHandler [list FileLogger Log] warning
Oc_Log SetLogHandler [list FileLogger Log] info

Oc_Log AddType infolog ;# Record in log file only
if {$loglevel>0} {
   Oc_Log SetLogHandler [list FileLogger Log] infolog
}

# Create a new Oxs_Destination for each Net_Thread that becomes Ready
Oc_EventHandler New _ Net_Thread Ready [list Oxs_Destination New _ %object]

##########################################################################
# Here's the guts of OXSII -- a switchboard between interactive GUI events
# and the set of Tcl commands provided by OXS
##########################################################################
# OXS provides the following classes:
#	FileLogger
#	Oxs_Mif
# and the following operations:
#	Oxs_ProbInit $file $params	: Calls Oxs_ProbRelease.  Reads
#					: $file into an Oxs_Mif object
#                                       : with $params parameter list.
#					: Creates Ext and Output objects
#					: Resets objects (driver)
#	Oxs_ProbReset			: Resets objects (driver)
#       Oxs_SetRestartFlag $flag        : Determines whether the next
#                                       : call to Oxs_ProbInit generates
#                                       : a fresh run, or if instead a
#                                       : checkpoint file is used to
#                                       : restart a previously suspended
#                                       : run.
#	Oxs_Run				: driver takes steps, may generate
#					: Step, Stage, Done events
#	Oxs_ProbRelease			: Destroys Ext and Output objects
#	Oxs_ListRegisteredExtClasses	: returns names
#	Oxs_ListExtObjects		: returns names
#	Oxs_ListEnergyObjects		: returns names
#	Oxs_ListOutputObjects		: returns output tokens
#	Oxs_OutputGet $outputToken $args : return or write output value
#	Oxs_OutputNames $outputToken	: returns name/type/units of output
#
# Some of these commands only make sense when the solver is in a
# particular state.  For example, [Oxs_Run] can only succeed if a
# problem is loaded.  So, this script has to keep track of the
# state of the solver.
##########################################################################

# Any changes to status get channeled through [ChangeStatus]
set status UNINITIALIZED
trace variable status w [list ChangeStatus $status]
proc ChangeStatus {old args} {
    global status interface_state problem
    if {[string match $old $status]} {return}
    if {[string match Done $old]} {
       # Limit stage changes available from Done state
       if {![string match "Loading..." $status] \
           && ![string match "Initializing..." $status]} {
          set status Done
          return
       }
    }
    trace vdelete status w [list ChangeStatus $old]
    trace variable status w [list ChangeStatus $status]
    Oc_EventHandler DeleteGroup ChangeStatus
    switch -exact -- $status {
	"" {
	    # The initial state -- no problem loaded.
	    # Also the state after any problem load fails, or
	    # a problem is released.
	    set interface_state disabled
	}
	Loading... {
	    set interface_state disabled
	    # Let interface get updated with above changes, then
	    # call ProblemLoad
	    after idle LoadProblem problem
	}
        Initializing... {
	    set interface_state disabled
	    after idle Reset
	}
	Pause {
	    # A problem is loaded, but not running.
	    set interface_state normal
	}
	Run {
	    after idle Loop Run
	}
	Relax {
	    Oc_EventHandler New _ Oxs Stage [list set status Pause] \
		    -oneshot 1 -groups [list ChangeStatus]
	    after idle Loop Relax
	}
        Step {
            Oc_EventHandler New _ Oxs Step [list set status Pause] \
                    -oneshot 1 -groups [list ChangeStatus]
            after idle Loop Step
        }
	Done {
           global problem exitondone
           if {[info exists problem]} {
              Oc_Log Log "Done \"[file tail $problem]\"" infolog
           }
           if {$exitondone} {
              exit	;# will trigger ReleaseProblem, etc.
           } else {
              # do nothing ?
           }
	}
	default {error "Status: $status not yet implemented"}
    }
}
# Initialize status to "" -- no problem loaded.
set status ""

# Routine to flush pending log messages.  Used for cleanup
proc FlushLog {} {
    foreach id [after info] {
	foreach {script type} [after info $id] break
	if {[string match idle $type] && [string match *Oc_Log* $script]} {
	    uplevel #0 $script
	}
    }
}

# Be sure any loaded problem gets release on application exit
Oc_EventHandler New _ Oc_Main Shutdown ReleaseProblem -oneshot 1
Oc_EventHandler New _ Oc_Main Shutdown FlushLog
proc ReleaseProblem {} {
    global status

    # We're about to release any loaded problem.  Spread the word.
    Oc_EventHandler Generate Oxs Release
    if {[catch {
	Oxs_ProbRelease
    } msg]} {
	# This is really bad.  Kill the solver.
	#
	# ...but first flush any pending log messages to the
	# error log file.
	FlushLog
	Oc_Log Log "Oxs_ProbRelease FAILED:\n\t$msg" panic
	exit
    }
    global problem
    if {[info exists problem]} {
       Oc_Log Log "End \"[file tail $problem]\"" infolog
    }
    set status ""
}

proc MeshGeometry {} {
   # Returns a tweaked version of the mesh geometry string
   set geostr [Oxs_MeshGeometryString]
   if {[regexp {([0-9]+) cells$} $geostr dummy cellcount]} {
      while {[regsub {([0-9])([0-9][0-9][0-9])( |$)} \
                 $cellcount {\1 \2} cellcount]} {}
      regsub {([0-9]+) cells$} $geostr "$cellcount cells" geostr
   }
   return $geostr
}

set workdir [Oc_DirectPathname "."]  ;# Initial working directory
Oc_Log AddSource LoadProblem
proc LoadProblem {fname} {
   # Side effects: If problem $f is successfully loaded, then the
   # working directory is changed to the directory holding $f.
   # Also, f is changed to a direct (i.e., absolute) pathname.
   # This way, the file can be loaded again later without worrying
   # about changes to the working directory.
   #    Also, the "About" info may be changed by the SetupThreads
   # call.

   global status step autorun autorun_pause workdir
   global stage stagerequest number_of_stages
   upvar 1 $fname f

   # We're about to release any loaded problem.  Spread the word.
   Oc_EventHandler Generate Oxs Release

   set f [Oc_DirectPathname $f]
   set newdir [file dirname $f]
   set origdir [Oc_DirectPathname "."]

   set msg "not readable"
   if {![file readable $f] || [catch {
      cd $newdir
      set workdir $newdir
   } msg] || [catch {
      global MIF_params
      if {![catch {Oxs_GetMif} mif]} {
         set pf [file tail [$mif GetFilename]]
         Oc_Log Log "End \"$pf\"" infolog
      }
      if {[llength $MIF_params]} {
         set ps "Params: $MIF_params"
      } else {
         set ps "no params"
      }
      Oc_Log Log "Start \"$f\", $ps" infolog
      Oxs_ProbRelease
      SetupThreads
      Oxs_ProbInit $f $MIF_params
      global update_extra_info
      append update_extra_info "\nMesh geometry: [MeshGeometry]"
      Oc_Main SetExtraInfo $update_extra_info
   } msg] || [catch {
      foreach o [Oxs_ListOutputObjects] {
         Oxs_Output New _ $o
      }
   } msg]} {
      # Error; the problem has been released
      global errorInfo errorCode
      after idle [subst {[list set errorInfo $errorInfo]; \
                            [list set errorCode $errorCode]; [list \
           Oc_Log Log "Error loading $f:\n\t$msg" error LoadProblem]}]
      set status ""
      cd $origdir  ;# Make certain cwd is same as on entry
      set workdir $origdir
   } else {
      set mif [Oxs_GetMif]
      foreach {crc fsize} [$mif GetCrc] break
      Oc_Log Log "Loaded \"[file tail $f]\", CRC: $crc, $fsize bytes " infolog
      foreach {step stage number_of_stages} [Oxs_GetRunStateCounts] break
      set stagerequest $stage
      set script {set status Pause}
	  puts [list $autorun $autorun_pause]
      if {$autorun && !$autorun_pause} {
         append script {; after 1 {set status Run}}
         ## The 'after 1' is to allow the solver console
         ## (as opposed to the interface console), if any,
         ## an opportunity to initialize and display before
         ## the solver begins chewing CPU cycles.  Otherwise,
         ## the console never(?) displays and the interface
         ## can't be brought up either.
      }
      set autorun 0 ;# Autorun at most once
      after idle [list SetupInitialSchedule $script]
   }
}

proc SetupInitialSchedule {script} {
    # Need account server; HACK(?) - we know it's in [global a]
    # need MIF object; HACK(?) - retrieve it from OXS
    upvar #0 a acct
    if {![$acct Ready]} {
        Oc_EventHandler New _ $acct Ready [info level 0] -oneshot 1
	return
    }
    set mif [Oxs_GetMif]
    Oc_EventHandler New _ $mif ScheduleReady $script -oneshot 1 \
	    -groups [list $mif]
    $mif SetupSchedule $acct
}

Oc_Log AddSource Reset
proc Reset {} {
    global status stage step stagerequest
    set status Pause
    Oc_EventHandler Generate Oxs Cleanup
    if {[catch {
	Oxs_ProbReset
    } msg]} {
	global errorInfo errorCode
	after idle [subst {[list set errorInfo $errorInfo]; \
		[list set errorCode $errorCode]; [list \
		Oc_Log Log "Reset error:\n\t$msg" error Reset]}]
	ReleaseProblem
    } else {
        foreach {step stage number_of_stages} [Oxs_GetRunStateCounts] break
        set stagerequest $stage
	# Should the schedule be reset too?
	# No, we decided that a Reset during interactive operations should
	# keep whatever schedule has been set up interactively.  For batch
	# use, resets will be rare, and if it turns out that in that case
	# we should reset the schedule, we'll add it when that becomes clear.
    }
}

# Keep taking steps as long as the status is unchanged,
# remaining one of Run, Relax.
Oc_Log AddSource Loop
proc Loop {type} {
    global status
    if {![string match $type $status]} {return}
    if {[catch {Oxs_Run} msg]} {
	global errorInfo errorCode
	after idle [subst {[list set errorInfo $errorInfo]; \
		[list set errorCode $errorCode]; \
		[list Oc_Log Log $msg error Loop]}]
	ReleaseProblem
    } else {
	;# Fire event handlers
	foreach ev $msg {
	    # $ev is a 4 item list: <event_type state_id stage step>
	    set event [lindex $ev 0]
	    switch -exact -- $event {
		STEP {
		    Oc_EventHandler Generate Oxs Step \
			    -stage [lindex $ev 2] \
			    -step [lindex $ev 3]
		}
		STAGE_DONE {
		    Oc_EventHandler Generate Oxs Stage
		}
		RUN_DONE {
		    Oc_EventHandler Generate Oxs Done
		}
		default {
		    after idle [list Oc_Log Log \
			    "Unrecognized event: $event" error Loop]
		    ReleaseProblem
		}
	    }
	}
    }
    after idle [info level 0]
}

# When the Load... menu item chooses a problem,
# $problem is the file to load.
trace variable problem w {uplevel #0 set status Loading... ;#}

# Update Stage and Step counts for Oxs_Schedule
Oc_EventHandler New _ Oxs Step [list set step %step]
Oc_EventHandler New _ Oxs Step [list set stage %stage]
Oc_EventHandler New _ Oxs Step {
    if {$stagerequest != %stage} {
		trace vdelete stagerequest w {
		    if {[info exists stage] && $stage != $stagerequest} {
			Oxs_SetStage $stagerequest
			set stage [Oxs_GetStage]
			if {$stage != $stagerequest} {
			    after idle [list set stagerequest $stage]
			}
		    } ;#}
	set stagerequest %stage
		trace variable stagerequest w {
		    if {[info exists stage] && $stage != $stagerequest} {
			Oxs_SetStage $stagerequest
			set stage [Oxs_GetStage]
			if {$stage != $stagerequest} {
			    after idle [list set stagerequest $stage]
			}
		    } ;#}
    }
}
		trace variable stagerequest w {
		    if {[info exists stage] && $stage != $stagerequest} {
			Oxs_SetStage $stagerequest
			set stage [Oxs_GetStage]
			if {$stage != $stagerequest} {
			    after idle [list set stagerequest $stage]
			}
		    } ;#}

# Terminate Loop when solver is Done
Oc_EventHandler New _ Oxs Done [list set status Done]

# All problem releases should request cleanup first
Oc_EventHandler New _ Oxs Release [list Oc_EventHandler Generate Oxs Cleanup]

##########################################################################
# Our server offers GeneralInterface protocol services -- start it
##########################################################################
Net_Server New server -alias [Oc_Main GetAppName] \
	-protocol [Net_GeneralInterfaceProtocol $gui {
		Oxs_Output Oxs_Schedule
	}]
$server Start 0

##########################################################################
# Track the threads known to the account server
#	code mostly cribbed from mmLaunch
#	good candidate for library code?
##########################################################################
# Get Thread info from account server:
proc Initialize {acct} {
   # Now that connections to servers are established, it's safe
   # to process options and possibly start computing.
   global problem
   if {[info exists problem]} {
      set problem $problem  ;# Fire trace
   }

   global nice
   if {$nice} {
      Oc_MakeNice
   }

   AccountReady $acct
}
proc AccountReady {acct} {
    set qid [$acct Send threads]
    Oc_EventHandler New _ $acct Reply$qid [list GetThreadsReply $acct] \
        -groups [list $acct]
    Oc_EventHandler New _ $acct Ready [list AccountReady $acct] -oneshot 1
}
proc GetThreadsReply { acct } {
    # Set up to receive NewThread messages, but only one handler per account
    Oc_EventHandler DeleteGroup GetThreadsReply-$acct
    Oc_EventHandler New _ $acct Readable [list HandleAccountTell $acct] \
            -groups [list $acct GetThreadsReply-$acct]
    set threads [$acct Get]
    Oc_Log Log "Received thread list: $threads" status
    # Create a Net_Thread instance for each element of the returned
    # thread list
    if {![lindex $threads 0]} {
        foreach triple [lrange $threads 1 end] {
	    NewThread $acct [lindex $triple 1]
        }
    }
}
# Detect and handle NewThread message from account server
proc HandleAccountTell { acct } {
    set message [$acct Get]
    switch -exact -- [lindex $message 0] {
        newthread {
            NewThread $acct [lindex $message 1]
        }
        deletethread {
            Net_Thread New t -hostname [$acct Cget -hostname] \
                    -pid [lindex $message 1]
            if {[$t Ready]} {
                $t Send bye
            } else {
                Oc_EventHandler New _ $t Ready [list $t Send bye]
            }
        }
        default {
          Oc_Log Log "Bad message from account $acct:\n\t$message" status
        }
    }
}
# There's a new thread with id $id, create corresponding local instance
proc NewThread { acct id } {

    # Do not keep any connections to yourself!
    if {[string match [$acct OID]:* $id]} {return}

    Net_Thread New _ -hostname [$acct Cget -hostname] \
            -accountname [$acct Cget -accountname] -pid $id
}
# Delay "nice" of this process until any children are spawned.
Net_Account New a
if {[$a Ready]} {
    Initialize $a
} else {
    Oc_EventHandler New _ $a Ready [list Initialize $a] -oneshot 1
}

vwait forever




