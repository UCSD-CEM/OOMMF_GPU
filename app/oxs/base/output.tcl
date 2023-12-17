##########################################################################
# Class representing script interface to an output
##########################################################################
Oc_Class Oxs_Output {
    private array common closers
    private array common index
    private array common directory
    # Map from the "type" of data provided by an Output to the protocol
    # needed in order to send that data type out.
    private array common protocol {
        "vector field"  vectorField
        "scalar field"  scalarField
        DataTable       DataTable
    }

    # We assume that all Oxs_Output objects existing at the same time
    # are provided by the same loaded problem, i.e. the same MIF,
    # so all scalars should be gathered into one DataTable, and one
    # basename is shared in common by all Oxs_Output objects.
    # If Oxs ever allows multiple problems to be loaded at once,
    # re-think this.
    private common scalars {}
    private common basename oxs
    private common driver

    # Formats for embedding into filenames
    public variable iter_fmt = "%07d"
    public variable stage_fmt = "%02d"

    const public variable handle
    const public variable name
    const private variable type
    const public variable units
    ClassConstructor {
        trace variable index wu [subst { uplevel #0 {
            set opAssocList \[[list array get [$class GlobalName index]]]
        } ;# }]
        array set index [list DataTable DataTable]
        array set directory [list DataTable DataTable]
        foreach d [Oxs_Destination Instances] {
            if {[string compare DataTable [$d Cget -protocol]] == 0} {
                Oxs_Schedule New _ -output DataTable \
                        -destination [$d Cget -name]
            }
        }
        # When the problem is released, output handles become invalid,
        # so Oxs_Output instances must be destroyed.
        Oc_EventHandler New _ Oxs Release [list $class DeleteAll]
    }
    proc DeleteAll {} {
        # remove trace on index so we don't send message for every
        # individual object destruction
        trace vdelete index wu [subst { uplevel #0 {
            set opAssocList \[[list array get [$class GlobalName index]]]
        } ;# }]
        foreach i [$class Instances] {
            $i Delete
        }
        global opAssocList
        set opAssocList [array get index]
        # re-establish trace on index
        trace variable index wu [subst { uplevel #0 {
            set opAssocList \[[list array get [$class GlobalName index]]]
        } ;# }]
    }
    Constructor {_handle} {
        set handle $_handle
        foreach {owner id type units} [Oxs_OutputNames $handle] {break}
        set name "$owner:$id"
        if {[info exists index($name)]} {
            return -code error "Duplicate output name: $name"
        }
        if {[string match scalar $type]} {
            if {[lsearch -exact $scalars $name] >= 0} {
                return -code error "Duplicate output name: $name"
            }
            lappend scalars $name
            set directory($name) $this
        } else {
            # Register directly with the output list
            set index($name) $protocol($type)
            set directory($name) $this
            foreach d [Oxs_Destination Instances] {
                if {[string compare $index($name) [$d Cget -protocol]] == 0} {
                    Oxs_Schedule New _ -output $name \
                            -destination [$d Cget -name]
                }
            }
        }
        if {![info exists driver]} {
            set driver [Oxs_DriverName]
            Oc_EventHandler New _ Oxs Release \
                    [list unset [$class GlobalName driver]] -oneshot 1
        }

        # Get filename base from director.  Convert to absolute
        # pathname if not already in that form.
        set basename [Oxs_MifOption basename]
        if {[string match relative [file pathtype $basename]]} {
            # The relative check may not be strictly necessary,
            # but may be helpful in case basename is volume
            # relative...in which case we punt and don't change it.
            global workdir
            set basename [file join $workdir $basename]
        }

        # Stage value format, for embedding into filenames
        foreach {step stage number_of_stages} [Oxs_GetRunStateCounts] break
        if {$number_of_stages>99} {
            set digits [expr {int(floor(log(double($number_of_stages))))+1}]
            set stage_fmt "%0${digits}d"
        }

    }
    Destructor {
        Oc_EventHandler Generate $this Delete
        Oc_EventHandler DeleteGroup $this
        set idx [lsearch -exact $scalars $name]
        if {$idx < 0} {
            unset index($name)
        } else {
            set scalars [lreplace $scalars $idx $idx]
        }
        catch {unset directory($name)}
    }
    proc Lookup {n} {
        return $directory($n)
    }
    private proc SendDataTable {thread extra} {
	array set custom $extra
	if {[info exists custom(-basename)]} {
            set triples [list [list @Filename:$custom(-basename).odt {} 0]]
	} else {
            set triples [list [list @Filename:$basename.odt {} 0]]
	}
        foreach on $scalars {
            set o $directory($on)
            set n [$o Cget -name]
            set u [$o Cget -units]
            set h [$o Cget -handle]
            set data [Oxs_OutputGet $h]
            ### KLUDGE KLUDGE KLUDGE ###
            if {[catch {expr {1.0*$data}}]} {
                # Assume this is an underflow bug
                set data 0.0
            }
            ### KLUDGE KLUDGE KLUDGE ###
            lappend triples [list $n $u $data]
        }
        # We're sending DataTable data.  Set up an event to guarantee
        # the table gets closed when the problem ends.
        if {![info exists closers($thread)]} {
            Oc_EventHandler New closers($thread) Oxs Cleanup \
                    "[list $thread Send DataTable [list [list @Filename: \
                    {} 0]]]; [list unset [$class GlobalName \
                    closers]($thread)]" -groups [list $thread] -oneshot 1
        }
        return [$thread Send DataTable $triples]
    }
    proc Send {n dest} {
        if {[catch {set directory($n)} output]} {
            return -code error -errorcode OxsUnknownOutput \
               "Unknown output: $n"
        }
        if {[catch {Oxs_Destination Thread $dest} thread]} {
            return -code error -errorcode OxsUnknownDestination \
               "Unknown destination: $dest"
        }

        # DataTable output gets special handling
        if {[string compare DataTable $output]==0} {
	    set extra [[Oxs_Destination Lookup $dest] Retrieve]
            return [$class SendDataTable $thread $extra]
        }
        return [$output Send $thread]
    }
    method Protocol {} {
        if {[info exists protocol($type)]} {
            return $protocol($type)
        }
        return ""
    }
    method Send {thread} {
        # Send my output to $thread.
        # Exact mechanism depends on the protocol
        if {![info exists protocol($type)]} {
            return -code error "No known protocol for sending output\
                    of type $type"
        }
        switch -exact -- $protocol($type) {
            vectorField {
                # Piece together a permanent file name
                set idx [lsearch -glob $scalars $driver:Iteration]
                set iter [format $iter_fmt [expr {int([Oxs_OutputGet \
                        [$directory([lindex $scalars $idx]) Cget -handle]])}]]
                set idx [lsearch -glob $scalars $driver:Stage]
                set stage [format $stage_fmt [expr {int([Oxs_OutputGet \
                        [$directory([lindex $scalars $idx]) Cget -handle]])}]]
                switch -exact -- $units {
                    T {set ext .obf ;# vector field in Tesla}
                    A/m {
                        switch -glob -- [string tolower $name] {
                            *field* {set ext .ohf}
                            *magnetization* {set ext .omf}
                            default {set ext .ovf}
                        }
                    }
                    {} { set ext .omf ;# spin (M/Ms) field }
                    default {set ext .ovf}
                }
                set filename $name-$stage-$iter$ext
                regsub -all {:+} $filename {-} filename
                regsub -all "\[ \r\n\t]+" $filename {_} filename
                regsub -all {/} $filename {_} filename
                set filename $basename-$filename

                # Piece together a temporary file name
		set temp [Oc_TempName [file tail $basename] $ext]
                Oxs_OutputGet $handle $temp
                return [$thread Send datafile [list $temp $filename]]
            }
            scalarField {
                # Piece together a permanent file name
                set idx [lsearch -glob $scalars $driver:Iteration]
                set iter [format $iter_fmt [expr {int([Oxs_OutputGet \
                        [$directory([lindex $scalars $idx]) Cget -handle]])}]]
                set idx [lsearch -glob $scalars $driver:Stage]
                set stage [format $stage_fmt [expr {int([Oxs_OutputGet \
                        [$directory([lindex $scalars $idx]) Cget -handle]])}]]
                switch -exact -- $units {
                   J/m^3  {set ext .oef ;# energy density field in Joules/meter^3 }
                   default {set ext .ovf}
                }
                set filename $name-$stage-$iter$ext
                regsub -all {:+} $filename {-} filename
                regsub -all "\[ \r\n\t]+" $filename {_} filename
                regsub -all {/} $filename {_} filename
                set filename $basename-$filename

                # Piece together a temporary file name
		set temp [Oc_TempName [file tail $basename] $ext]
                Oxs_OutputGet $handle $temp
                return [$thread Send datafile [list $temp $filename]]
            }
            default {
                return -code error "Don't know how to send output\
                        using \"$protocol($type)\# protocol"
            }
        }
    }
}

