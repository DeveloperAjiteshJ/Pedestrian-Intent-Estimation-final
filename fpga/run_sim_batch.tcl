set script_dir [file normalize [file dirname [info script]]]
set canonical_xpr [file normalize [file join $script_dir vivado_project tinymobilenet_fpga.xpr]]

if {[file exists $canonical_xpr]} {
    set project_xpr $canonical_xpr
} else {
    set proj_glob [glob -nocomplain -types f [file join $script_dir vivado_project* tinymobilenet_fpga.xpr]]
    if {[llength $proj_glob] == 0} {
        error "No tinymobilenet_fpga.xpr found under fpga/vivado_project*"
    }

    set project_xpr ""
    set latest_mtime -1
    foreach p $proj_glob {
        set mt [file mtime $p]
        if {$mt > $latest_mtime} {
            set latest_mtime $mt
            set project_xpr $p
        }
    }
}

puts "Using project: $project_xpr"
open_project $project_xpr
set_property top tinymobilenet_tb [get_filesets sim_1]
update_compile_order -fileset sim_1
launch_simulation -simset sim_1 -mode behavioral
run all
close_sim -force
close_project
puts "Simulation batch run completed."