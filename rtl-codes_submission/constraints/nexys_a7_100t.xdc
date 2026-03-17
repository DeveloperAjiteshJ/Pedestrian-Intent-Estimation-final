# ============================================================================
# Nexys A7-100T Constraints File for TinyMobileNet-XS FPGA Inference
# ============================================================================

# Clock: 100 MHz
set_property -dict { PACKAGE_PIN E3 IOSTANDARD LVCMOS33 } [get_ports clk]
create_clock -add -name sys_clk_pin -period 10.00 -waveform {0 5} [get_ports clk]

# Reset Button (BTNU, active-high in hardware)
set_property -dict { PACKAGE_PIN M18 IOSTANDARD LVCMOS33 } [get_ports reset_button]

# Start Button (BTNC)
set_property -dict { PACKAGE_PIN N17 IOSTANDARD LVCMOS33 } [get_ports start_button]

# Seven-segment display (8 digits, active-low)
set_property -dict { PACKAGE_PIN J17 IOSTANDARD LVCMOS33 } [get_ports {an[0]}]
set_property -dict { PACKAGE_PIN J18 IOSTANDARD LVCMOS33 } [get_ports {an[1]}]
set_property -dict { PACKAGE_PIN T9  IOSTANDARD LVCMOS33 } [get_ports {an[2]}]
set_property -dict { PACKAGE_PIN J14 IOSTANDARD LVCMOS33 } [get_ports {an[3]}]
set_property -dict { PACKAGE_PIN P14 IOSTANDARD LVCMOS33 } [get_ports {an[4]}]
set_property -dict { PACKAGE_PIN T14 IOSTANDARD LVCMOS33 } [get_ports {an[5]}]
set_property -dict { PACKAGE_PIN K2  IOSTANDARD LVCMOS33 } [get_ports {an[6]}]
set_property -dict { PACKAGE_PIN U13 IOSTANDARD LVCMOS33 } [get_ports {an[7]}]

set_property -dict { PACKAGE_PIN T10 IOSTANDARD LVCMOS33 } [get_ports {seg[0]}]
set_property -dict { PACKAGE_PIN R10 IOSTANDARD LVCMOS33 } [get_ports {seg[1]}]
set_property -dict { PACKAGE_PIN K16 IOSTANDARD LVCMOS33 } [get_ports {seg[2]}]
set_property -dict { PACKAGE_PIN K13 IOSTANDARD LVCMOS33 } [get_ports {seg[3]}]
set_property -dict { PACKAGE_PIN P15 IOSTANDARD LVCMOS33 } [get_ports {seg[4]}]
set_property -dict { PACKAGE_PIN T11 IOSTANDARD LVCMOS33 } [get_ports {seg[5]}]
set_property -dict { PACKAGE_PIN L18 IOSTANDARD LVCMOS33 } [get_ports {seg[6]}]
set_property -dict { PACKAGE_PIN H15 IOSTANDARD LVCMOS33 } [get_ports dp]

# UART Interface
set_property -dict { PACKAGE_PIN C4 IOSTANDARD LVCMOS33 } [get_ports uart_rx]
set_property -dict { PACKAGE_PIN D4 IOSTANDARD LVCMOS33 } [get_ports uart_tx]

# Inference Done Signal (optional debug)
set_property -dict { PACKAGE_PIN N14 IOSTANDARD LVCMOS33 } [get_ports inference_done]

# Timing/IO tuning
set_property SLEW SLOW [get_ports {an[*]}]
set_property SLEW SLOW [get_ports {seg[*]}]
set_property SLEW SLOW [get_ports dp]
set_property SLEW SLOW [get_ports uart_tx]

set_input_delay -clock sys_clk_pin -min 0.5 [get_ports start_button]
set_input_delay -clock sys_clk_pin -max 5.0 [get_ports start_button]
set_input_delay -clock sys_clk_pin -min 0.5 [get_ports reset_button]
set_input_delay -clock sys_clk_pin -max 5.0 [get_ports reset_button]

set_property SEVERITY {Warning} [get_drc_checks NSTD-1]
set_property SEVERITY {Warning} [get_drc_checks UCIO-1]
