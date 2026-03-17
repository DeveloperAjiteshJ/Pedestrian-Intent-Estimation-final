# FPGA Configuration: Maximum 11 Samples with Optimized Display

## ✅ Changes Completed

### 1. Maximum BRAM Utilization
- **Artix-7 100T BRAM:** 622,080 bytes (607.5 KB)
- **Maximum samples:** 11 (with 95% safety margin)
- **Total usage:** 540,672 bytes (528 KB)
- **BRAM utilization:** 86.9%

### 2. Sample Selection (fpga/select_synth_samples.py)
- **Samples:** 11 (maximum possible)
- **Selected indices:** [2, 10, 15, 22, 35, 47, 58, 68, 81, 95, 109]
- **Labels:** ['0', '1', '0', '1', '1', '0', '1', '1', '0', '1', '0']
- **Mix:** 6 crossing (1) + 5 not-crossing (0)
- **Spread:** Indices from 2 to 109 for maximum diversity

### 3. RTL Updates (fpga/rtl/tinymobilenet_top.v)
- **NUM_SAMPLES:** 6 → **11** (synthesis mode)
- **Counter width:** 3 bits → **4 bits** (to count 0-10)
- **Counter position:** **RIGHTMOST digit** (position 0)
- **Display layout:** Reorganized with spaces

### 4. Seven-Segment Display Layout (Right to Left)

```
[Expected] [E/C] [Pred] [__] [Conf_Tens] [Conf_Ones] [__] [COUNTER]
    7        6      5     4        3           2        1       0
```

**Position 0 (RIGHTMOST):** Cycle counter (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, A=10)
**Position 1:** Blank space
**Position 2-3:** Confidence percentage (65-95%)
**Position 4:** Blank space
**Position 5:** Predicted class (0 or 1)
**Position 6:** Match indicator (E=correct, C=wrong)
**Position 7 (LEFTMOST):** Expected ground truth (0 or 1)

### 5. Counter Display Format
- Values 0-9: Show as digits 0-9
- Value 10: Shows as "A" (hex digit)
- Wraps back to 0 after sample 10 (the 11th sample)

### 6. Expected Behavior on FPGA
Press START button 11 times:
1. Counter=0, see result with E or C
2. Counter=1, see different result
3. Counter=2
4. Counter=3
5. Counter=4
6. Counter=5
7. Counter=6
8. Counter=7
9. Counter=8
10. Counter=9
11. Counter=A (hex for 10)
12. **Wraps to 0** - cycle repeats

You should see a **good mix of E (correct) and C (wrong)** across the 11 samples!

## Memory File Verification
```
all_samples_synth.mem:   540,672 lines (11 × 49,152 bytes)
sample_labels_synth.mem: 11 lines
```

## To Rebuild and Program FPGA

### Quick Method:
```cmd
cd fpga
run_update_rebuild.cmd
```

### Manual in Vivado GUI:
1. Open: `vivado vivado_project_20260315_141753\tinymobilenet_fpga.xpr`
2. Click "Generate Bitstream"
3. Wait ~10-15 minutes
4. Program Nexys A7-100T

### Or Create Fresh Project:
```cmd
cd fpga
vivado -mode batch -source recreate_project_clean.tcl
```

## Why 11 Samples?
- Theoretical max: 12.66 samples
- Safe max (95% BRAM): **11 samples**
- Leaves 5% margin for:
  - Label ROM
  - Other potential BRAM uses
  - Synthesis optimization headroom

## Display Reading Guide
Example display: `1 C 1  65  3`
- 1 (leftmost) = Expected class 1 (crossing)
- C = Mismatch (wrong prediction)
- 1 = Predicted class 1 (crossing)
- [blank space]
- 65 = 65% confidence
- [blank space]
- 3 (rightmost) = This is the 4th sample (counter=3)
