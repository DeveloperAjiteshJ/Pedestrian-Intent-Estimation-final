# Force recreate a clean Vivado project with design sources and constraints
set script_dir [file normalize [file dirname [info script]]]
set project_root [file normalize [file join $script_dir vivado_project]]
set project_name "tinymobilenet_fpga"
set part "xc7a100tcsg324-1"

set repo_root [file normalize [file join $script_dir ..]]
set rtl_files [glob -nocomplain -types f [file join $script_dir rtl *.v]]
set xdc_file [file normalize [file join $script_dir constraints nexys_a7_100t.xdc]]
set mem_files [list \
    [file normalize [file join $repo_root fpga_test_vectors all_samples.mem]] \
    [file normalize [file join $repo_root fpga_test_vectors sample_labels.mem]] \
    [file normalize [file join $repo_root fpga_test_vectors all_samples_synth.mem]] \
    [file normalize [file join $repo_root fpga_test_vectors sample_labels_synth.mem]] \
    [file normalize [file join $repo_root fpga weights_mem weights_13.mem]] \
    [file normalize [file join $repo_root fpga weights_mem weights_14.mem]] \
]
set sim_tb_file [file normalize [file join $script_dir sim tinymobilenet_tb.v]]

if {[llength $rtl_files] == 0} {
    error "No RTL files found under [file join $script_dir rtl]"
}
if {![file exists $xdc_file]} {
    error "Constraint file not found: $xdc_file"
}
foreach mf $mem_files {
    if {![file exists $mf]} {
        error "Memory init file not found: $mf"
    }
}
if {![file exists $sim_tb_file]} {
    error "Simulation testbench not found: $sim_tb_file"
}

if {[llength [get_projects -quiet]] > 0} {
    close_project
}

set project_dir $project_root
set deleted_ok 1
if {[file exists $project_root]} {
    if {[catch {file delete -force $project_root} del_err]} {
        set deleted_ok 0
        puts "WARNING: Could not fully delete existing project dir (likely file lock): $del_err"
        set ts [clock format [clock seconds] -format "%Y%m%d_%H%M%S"]
        set project_dir [file normalize [file join $script_dir "vivado_project_$ts"]]
        puts "WARNING: Falling back to new project dir: $project_dir"
    }
}
if {![file exists $project_dir]} {
    file mkdir $project_dir
}

set project_xpr [file normalize [file join $project_dir ${project_name}.xpr]]

create_project $project_name $project_dir -part $part
set_property default_lib work [current_project]
catch {set_property board_part "digilentinc.com:nexys-a7-100t:part0:1.1" [current_project]}

import_files -fileset sources_1 -norecurse $rtl_files
add_files -fileset sources_1 -norecurse $mem_files
import_files -fileset sim_1 -norecurse $sim_tb_file
foreach mf $mem_files {
    set mf_obj [get_files -all -quiet $mf]
    if {[llength $mf_obj] > 0} {
        set_property file_type {Memory Initialization Files} $mf_obj
        set bn [file tail $mf]
        if {$bn eq "all_samples.mem" || $bn eq "sample_labels.mem"} {
            set_property used_in_synthesis false $mf_obj
            set_property used_in_simulation true $mf_obj
        } elseif {$bn eq "all_samples_synth.mem" || $bn eq "sample_labels_synth.mem"} {
            set_property used_in_synthesis true $mf_obj
            set_property used_in_simulation false $mf_obj
        } else {
            set_property used_in_synthesis true $mf_obj
            set_property used_in_simulation true $mf_obj
        }
    }
}
import_files -fileset constrs_1 -norecurse $xdc_file
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1
set_property top tinymobilenet_top [get_filesets sources_1]
set_property top tinymobilenet_tb [get_filesets sim_1]

set constr_files [get_files -of_objects [get_filesets constrs_1]]
if {[llength $constr_files] == 0} {
    error "No constraint files present in constrs_1 after import"
}
set_property used_in_synthesis true $constr_files
set_property used_in_implementation true $constr_files

puts "Created: $project_xpr"
puts "Sources in sources_1:"
foreach f [get_files -of_objects [get_filesets sources_1]] { puts "  - $f" }
puts "Constraints in constrs_1:"
foreach f [get_files -of_objects [get_filesets constrs_1]] { puts "  - $f" }

close_project
