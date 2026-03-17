"""
Calculate maximum number of samples that fit in Artix-7 100T BRAM
and select a diverse set with both correct and wrong predictions
"""

# FPGA BRAM capacity
ARTIX7_100T_BRAM_BITS = 4_976_640  # bits (from Xilinx datasheet)
ARTIX7_100T_BRAM_BYTES = ARTIX7_100T_BRAM_BITS // 8  # 622,080 bytes

# Sample size
BYTES_PER_SAMPLE = 49152

# Calculate maximum samples
max_samples_float = ARTIX7_100T_BRAM_BYTES / BYTES_PER_SAMPLE
max_samples = int(max_samples_float)

# Leave some margin for other BRAM usage (labels, etc.)
SAFETY_MARGIN = 0.95  # Use 95% of BRAM
safe_max_samples = int(max_samples * SAFETY_MARGIN)

print(f"Artix-7 100T BRAM Capacity: {ARTIX7_100T_BRAM_BYTES:,} bytes ({ARTIX7_100T_BRAM_BYTES/1024:.2f} KB)")
print(f"Bytes per sample: {BYTES_PER_SAMPLE:,}")
print(f"Theoretical max samples: {max_samples_float:.2f}")
print(f"Integer max samples: {max_samples}")
print(f"Safe max samples (95%): {safe_max_samples}")
print(f"\nUsing {safe_max_samples} samples for synthesis")
print(f"Total BRAM usage: {safe_max_samples * BYTES_PER_SAMPLE:,} bytes ({safe_max_samples * BYTES_PER_SAMPLE / 1024:.2f} KB)")
print(f"BRAM utilization: {(safe_max_samples * BYTES_PER_SAMPLE / ARTIX7_100T_BRAM_BYTES * 100):.1f}%")
