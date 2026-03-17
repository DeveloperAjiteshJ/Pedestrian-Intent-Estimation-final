# FPGA Weight Export Guide

This guide explains how to export trained TinyMobileNet-XS weights to INT8 format for FPGA deployment.

## Overview

After training your model with `train.py`, you need to:
1. **Quantize** the model from FP32 to INT8 (8-bit integers)
2. **Export** the weights to a C header file for FPGA/RTL

## Files Created

### 1. `export_quantized_weights.py`
Main export script with full functionality:
- Loads trained model checkpoint
- Performs post-training quantization (PTQ)
- Exports INT8 weights to C header
- Optional validation of quantized model

### 2. `quick_export.py`
Simplified version for quick exports without user interaction.

### 3. `export_weights.bat`
Windows batch script for one-click export.

## Usage

### Method 1: Quick Export (Recommended)

**Windows:**
```bash
export_weights.bat
```

**Linux/Mac:**
```bash
python quick_export.py
```

This will automatically:
- Load the trained model from `checkpoints/best_model.pth`
- Quantize it using test set for calibration
- Export to `fpga_weights/tinymobilenet_xs_weights.h`

### Method 2: Full Export with Validation

```bash
python export_quantized_weights.py
```

This provides:
- Interactive validation prompt
- Accuracy comparison (FP32 vs INT8)
- Detailed quantization statistics

## Prerequisites

Make sure you have:
1. ✅ Trained model at `checkpoints/best_model.pth`
   - Run `python train.py` first if you don't have this
2. ✅ Test dataset at `PIE/data_cache/sequences/test_sequences.pkl`
3. ✅ PyTorch and dependencies installed

## Output Files

After running the export, you'll get:

```
fpga_weights/
├── tinymobilenet_xs_weights.h    ← INT8 weights (C header for FPGA)
└── quantization_config.json      ← Quantization scales & metadata
```

### `tinymobilenet_xs_weights.h`
C header file containing:
- INT8 weights for all layers
- Array declarations ready for FPGA/RTL
- Scale factors for dequantization

Example format:
```c
// Layer 0: conv1
// Shape: (8, 3, 3, 3)
// Scale: 45.2341
int8_t weights_0[216] = {
    12, -5, 3, 8, ...,
};
```

### `quantization_config.json`
JSON file with:
- Per-layer weight scales
- Per-layer activation scales
- Quantization metadata

Example:
```json
{
  "weight_scales": {
    "conv1.0": 45.234,
    "bottleneck_a.expand.0": 32.156,
    ...
  },
  "activation_scales": {
    "relu": 12.345,
    ...
  },
  "quantization_bits": 8
}
```

## Quantization Process

The export pipeline performs:

### 1. BatchNorm Folding
Folds BatchNorm parameters into Conv weights:
```
y = (x * w) * gamma/sqrt(var + eps) + beta
  ↓ fold ↓
y = x * (w * gamma/sqrt(var + eps)) + beta
```

### 2. Post-Training Quantization (PTQ)
- Uses test set for calibration (first 10 batches)
- Collects activation statistics (min/max)
- Computes optimal quantization scales
- Formula: `int8_value = round(fp32_value * scale / 127)`

### 3. INT8 Export
- Converts all weights to 8-bit integers [-128, 127]
- Generates C arrays for FPGA
- Saves scales for runtime dequantization

## Using Exported Weights in FPGA/RTL

### Step 1: Copy Header File
```bash
cp fpga_weights/tinymobilenet_xs_weights.h your_vivado_project/
```

### Step 2: Include in Your Design
```c
#include "tinymobilenet_xs_weights.h"

// Access weights
int8_t* conv1_weights = weights_0;  // First conv layer
```

### Step 3: Implement INT8 MAC
```verilog
// Example INT8 multiply-accumulate
wire signed [15:0] mac_result;
assign mac_result = input_int8 * weight_int8 + accumulator;
```

### Step 4: Apply Dequantization
```c
// After INT8 inference, dequantize using scales
float output = (int8_output / 127.0) / activation_scale;
```

## Troubleshooting

### Error: "Checkpoint not found"
```
❌ ERROR: Checkpoint not found at checkpoints/best_model.pth
```
**Solution:** Train the model first:
```bash
python train.py
```

### Error: "Calibration data not found"
```
❌ ERROR: Cannot find test_sequences.pkl
```
**Solution:** Run data preparation:
```bash
python generate_pie_database.py
python split_train_test.py
```

### Accuracy Drop > 5%
If quantization causes significant accuracy loss:
1. Use more calibration data (modify `batch_idx >= 10` in script)
2. Try Quantization-Aware Training (QAT) instead of PTQ
3. Use per-channel quantization (advanced)

## Advanced Options

### Custom Calibration Dataset
Edit `export_quantized_weights.py`:
```python
# Use training set instead of test set
CALIBRATION_PKL = os.path.join(PIE_ROOT, 'data_cache/sequences/train_sequences.pkl')
```

### More Calibration Batches
```python
# In quantize_model_post_training function
if batch_idx >= 50:  # Increased from 10
    break
```

### Skip Validation
```python
# In quick_export.py - validation is already skipped
# No changes needed
```

## Expected Results

Typical quantization results:
- **Weight size reduction:** 75% (FP32 → INT8)
- **Memory footprint:** ~7.2 KB → ~1.8 KB
- **Accuracy drop:** < 3% (typically 1-2%)
- **FPGA inference speedup:** 2-4x vs FP32

Example output:
```
📊 Accuracy Comparison:
  FP32 model:  81.25% (13/16)
  INT8 model:  81.25% (13/16)
  Accuracy drop: 0.00%

📊 Output Difference:
  Max logit difference: 0.0234
  Avg logit difference: 0.0089

✅ Quantization successful! Accuracy drop < 5%
```

## Next Steps

After exporting weights:
1. ✅ Open `fpga_weights/tinymobilenet_xs_weights.h`
2. ✅ Copy to your Vivado/Verilog project
3. ✅ Implement INT8 inference engine in RTL
4. ✅ Use scales from `quantization_config.json`
5. ✅ Test on FPGA board

## Additional Resources

- **Quantization tutorial:** https://pytorch.org/docs/stable/quantization.html
- **INT8 inference:** https://arxiv.org/abs/1712.05877
- **FPGA deployment:** https://github.com/pytorch/glow

---

**Questions?** Check the main README or contact the developer.
