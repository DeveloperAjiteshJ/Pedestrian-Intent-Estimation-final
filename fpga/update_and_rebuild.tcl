# Update the RTL and rebuild bitstream for the existing project (6 samples version)
set project_path "vivado_project_20260315_141753/tinymobilenet_fpga.xpr"

if {![file exists $project_path]} {
    puts "ERROR: Project not found at $project_path"
    puts "INFO: Creating fresh project instead..."
    source recreate_project_clean.tcl
    exit 0
}

# Open the existing project
open_project $project_path

# Refresh the RTL file to pick up changes
update_compile_order -fileset sources_1

# Refresh memory files (6-sample versions)
puts "Refreshing memory initialization files..."
reread_files -fileset sources_1

# Reset synthesis and implementation
reset_run synth_1
reset_run impl_1

# Launch synthesis
puts "Starting synthesis with 6 samples..."
launch_runs synth_1 -jobs 4
wait_on_run synth_1

# Check synthesis status
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "ERROR: Synthesis failed"
    exit 1
}

puts "Synthesis completed successfully"

# Launch implementation
puts "Starting implementation..."
launch_runs impl_1 -jobs 4
wait_on_run impl_1

# Check implementation status
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "ERROR: Implementation failed"
    exit 1
}

puts "Implementation completed successfully"

# Generate bitstream
puts "Generating bitstream..."
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

# Check bitstream generation status
set bit_file [get_property DIRECTORY [get_runs impl_1]]/tinymobilenet_top.bit
if {[file exists $bit_file]} {
    puts "SUCCESS: Bitstream generated at: $bit_file"
    puts "File size: [file size $bit_file] bytes"
    puts ""
    puts "======================================================================"
    puts "FPGA Configuration: 6 samples with mixed correct/wrong predictions"
    puts "Counter display: Leftmost digit (position 7)"
    puts "Expected behavior: You should see both 'E' and 'C' results"
    puts "======================================================================"
} else {
    puts "ERROR: Bitstream generation failed"
    exit 1
}

close_project
puts "Project updated and bitstream generated successfully!"
