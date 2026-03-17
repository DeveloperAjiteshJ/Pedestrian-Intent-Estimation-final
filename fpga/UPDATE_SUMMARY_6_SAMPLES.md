# FPGA Configuration Updated: 6 Samples with Mixed Results

## Changes Made:

### 1. Sample Selection (fpga/select_synth_samples.py)
- **Increased from 4 to 6 samples**
- **Manually selected indices:** [10, 15, 22, 47, 68, 95]
- **Labels:** ['1', '0', '1', '0', '1', '1']
  - Sample 10: Crossing (1)
  - Sample 15: Not-crossing (0)
  - Sample 22: Crossing (1)
  - Sample 47: Not-crossing (0)
  - Sample 68: Crossing (1)
  - Sample 95: Crossing (1)

### 2. RTL Updates (fpga/rtl/tinymobilenet_top.v)
- **NUM_SAMPLES:** 4 → 6 (synthesis mode only)
- **Counter display position:** Moved from digit 1 to digit 7 (leftmost)
- **Counter range:** Now counts 0→1→2→3→4→5→0

### 3. Seven-Segment Display Layout (Right to Left)
```
[COUNTER][Conf_Tens][Conf_Ones][Blank][Pred][Match][Expected][Blank][Blank]
    7         6           5        4      3      2       1        0
```

**Position 7 (Leftmost):** Cycle counter (0-5)
**Position 6-5:** Confidence percentage
**Position 4:** Predicted class (0 or 1)
**Position 3:** Match indicator (E=correct, C=wrong)
**Position 2:** Expected class (0 or 1)

### 4. Memory File Updates
- **all_samples_synth.mem:** 294,912 lines (6 samples × 49,152 bytes)
- **sample_labels_synth.mem:** 6 labels

### 5. Expected FPGA Behavior
You should now see BOTH results:
- **"E" (match):** When prediction matches ground truth
- **"C" (mismatch):** When prediction is wrong

The 6 selected samples include a diverse mix to maximize the chance of seeing both E and C on your display.

## How to Apply:

### Option 1: Automated Rebuild
```cmd
cd fpga
run_update_rebuild.cmd
```

### Option 2: Manual Vivado GUI
1. Open project: `vivado vivado_project_20260315_141753\tinymobilenet_fpga.xpr`
2. Click "Generate Bitstream"
3. Wait for completion (~5-10 minutes)
4. Program FPGA

### Option 3: Create Fresh Project
```cmd
cd fpga
vivado -mode batch -source recreate_project_clean.tcl
```
Then open the new project and generate bitstream.

## On-Board Testing:
1. Press START button → See counter "0" on leftmost digit
2. Check if result shows "E" or "C" in middle
3. Press START again → Counter shows "1"
4. Repeat 6 times to cycle through all samples
5. After sample 5, counter resets to 0

## File Size Check:
- **Previous:** 196 KB (4 samples)
- **Current:** 294 KB (6 samples)
- **Still fits in BRAM:** Yes (622 KB available)
