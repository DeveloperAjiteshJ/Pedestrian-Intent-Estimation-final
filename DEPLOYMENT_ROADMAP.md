"""
================================================================================
PEDESTRIAN INTENTION ESTIMATION - FPGA HARDWARE ACCELERATOR
EXECUTIVE SUMMARY & DEPLOYMENT ROADMAP
================================================================================

PROJECT SCOPE
=============
Target:     Binary classification (Crossing vs Not-Crossing) for pedestrians
Dataset:    PIE (Pedestrian Intention Estimation) — 40+ videos, balanced classes
Platform:   Artix-7 XC7A100T (Nexys DDR4) — Entry-level FPGA
Goal:       Real-time inference on video streams (620 FPS with 32 DSPs, 77 FPS with 8 DSPs)

================================================================================
DATABASE SUMMARY
================================================================================

✅ Training sequences generated:
   • Total samples: ~1000+ pedestrian tracks
   • Sequence length: Variable (8–32 frames median ~16)
   • Classification: Binary (K=2)
     - Class 0: Not crossing (~60%)
     - Class 1: Crossing (~40%)
   • Input per frame: 64×64×3 RGB (after resize from original)
   • Output: 2 logits (softmax → class + confidence)

Recommendation: Use T=8 for optimal accuracy/resource tradeoff
  (Your proposed T=4 is conservative; T=8 adds 5.2M MACs still fits easily)

================================================================================
ARCHITECTURE OVERVIEW
================================================================================

Model: TinyMobileNet-XS (Modified for T=8)

Backbone (Per-Frame):
  Layer 0: Conv 3×3 stride=2, 3→8 channels           (32×32×8)
  Layer 1: MobileBottleneck (DW 3×3), 8 channels     (32×32×8)
  Layer 2: MobileBottleneck (DW 3×3 stride=2, expan×4), 12 channels  (16×16×12)
  Layer 3: MobileBottleneck ×2 (stride=2, then stride=1), 16 channels (8×8×16)
           ✨ TSM inserted here (0 MACs, 4 KB BRAM)
  Layer 4: Conv 1×1, 16→48 channels                  (8×8×48)
  Layer 5: Global Average Pool                       (1×1×48)

Temporal Fusion:
  Option A: TSM + temporal average (CHOSEN — 0 extra MACs)
  Option B: 1D depthwise temporal conv (alternative, higher accuracy)

Classification Head:
  FC1: 48 → 32 (ReLU)
  FC2: 32 → 2 (logits)

Resource Summary:
  • Total parameters: 7,232 (7.2 KB)
  • Per-frame MACs: 1.29M
  • Full sequence MACs (T=8): 10.32M
  • Quantization: 8-bit signed integer (weights + activations)
  • Activation footprint: ~50 KB BRAM (worst case)

Performance Estimates (100 MHz clock):
  With 32 DSPs:  3.2 ms latency (~310 FPS)
  With 64 DSPs:  1.6 ms latency (~620 FPS)
  With 8 DSPs:  12.9 ms latency (~77 FPS)
  Actual: +15–25% overhead due to control/memory stalls

================================================================================
IMPLEMENTATION CHECKLIST
================================================================================

PHASE 1: MODEL TRAINING (CPU/GPU)
  ☐ Implement TinyMobileNet-XS in PyTorch/TensorFlow
  ☐ Train on PIE dataset with T=8 input
  ☐ Validate: target >88% accuracy on test set
  ☐ Calibrate quantization (collect activation statistics)
  ☐ Export weights as int8 C arrays + scale factors

PHASE 2: QUANTIZATION & VALIDATION
  ☐ Apply post-training quantization (PTQ) or QAT
  ☐ Verify quantized accuracy: >86% (loss <2%)
  ☐ Export golden test vectors (PyTorch → C arrays)
  ☐ Generate reference outputs for HLS verification

PHASE 3: HLS KERNEL DEVELOPMENT
  ☐ Implement Conv2D kernels (Conv1, BottleneckDW, BottleneckPW)
  ☐ Implement TSM circular buffer (BRAM-based)
  ☐ Implement GlobalAvgPool
  ☐ Implement FC layers
  ☐ Connect layers with hls::stream (DATAFLOW pragma)
  ☐ Test C simulation against golden vectors
  ☐ Verify II=1 (pipeline initiation interval)

PHASE 4: HIGH-LEVEL SYNTHESIS
  ☐ Set target clock: 100 MHz
  ☐ Synthesize with -DFPGA_TARGET=ARTIX7
  ☐ Check resource estimates:
     - LUTs: <3,000 (20% of 15,850)
     - DSPs: 32–80 (target 33–66%)
     - BRAM: <20/270 (target <10%)
     - FF: <2,000 (13% of 15,850)
  ☐ Place & Route, verify timing closure
  ☐ Generate .xdc constraints for DDR, clock

PHASE 5: HARDWARE INTEGRATION
  ☐ Create Vivado project (XC7A100T)
  ☐ Instantiate HLS core
  ☐ Add AXI interconnect (input/output streams)
  ☐ Add DDR3 controller (for weights + intermediate buffers)
  ☐ Add AXI-PCIE (if USB/PCIe host interface needed)
  ☐ Add Microblaze/PicoBlaze for control logic
  ☐ Implement preprocessing: resize 1920×1080 → 64×64

PHASE 6: FIRMWARE & VALIDATION
  ☐ Write C firmware for Microblaze:
     - Video input handler (frame capture)
     - Preprocessing (resize, quantize)
     - Softmax computation (post-network)
     - Output formatting (class + confidence + latency)
  ☐ Implement input/output handlers
  ☐ Test end-to-end on real video
  ☐ Measure latency via cycle counters
  ☐ Validate accuracy matches GPU baseline

PHASE 7: DEPLOYMENT & OPTIMIZATION
  ☐ Generate bitstream
  ☐ Test on Nexys DDR4 hardware
  ☐ Profile power consumption (use Vivado power analyzer)
  ☐ Profile thermal behavior (monitor FPGA temp via XADC)
  ☐ Optimize memory bandwidth (reduce DDR stalls)
  ☐ Fine-tune DSP allocation for latency requirements

================================================================================
FILE STRUCTURE
================================================================================

PIE/
├── generate_pie_database.py          [✓ COMPLETED] Image extraction + DB generation
├── check_database.py                 [✓] Verify dataset structure
├── verify_database_structure.py       [✓] Analyze database for FPGA requirements
├── FPGA_Architecture_Spec.py          [✓] This document (detailed specs)
│
├── [TO CREATE - TRAINING]
│   ├── models/
│   │   ├── tinymobilenet_xs.py       [Model definition]
│   │   ├── quantization.py           [PTQ/QAT utilities]
│   │   └── training.py               [Training loop]
│   ├── train.py                      [Entry point]
│   └── export_weights.py             [Export to C arrays]
│
├── [TO CREATE - FPGA/HLS]
│   ├── hls/
│   │   ├── network_kernels.cpp       [✓ TEMPLATE PROVIDED]
│   │   ├── preprocessing.cpp         [Resize, quantize]
│   │   ├── postprocessing.cpp        [Softmax, output format]
│   │   ├── top.cpp                   [Top-level wrapper]
│   │   └── weights.h                 [Generated int8 weights]
│   ├── vivado/
│   │   ├── tinymobilenet.tcl         [Create project script]
│   │   ├── constraints.xdc           [Pin assignments]
│   │   └── bitstream.bit             [Generated binary]
│   └── firmware/
│       ├── main.c                    [Microblaze control]
│       ├── input_handler.c           [Video input]
│       └── output_handler.c          [Results output]

================================================================================
QUANTIZATION STRATEGY
================================================================================

Training → INT8 Conversion:

1. FLOAT32 TRAINING (PyTorch/TF)
   model = TinyMobileNet_XS()
   train(model, pie_dataset, epochs=100, lr=1e-3)
   accuracy_float = validate(model)  # Target: >88%

2. CALIBRATION (Collect activation statistics)
   with torch.no_grad():
       for frames, labels in calibration_set:  # ~100 samples
           activations = collect_layer_activations(model, frames)
           # Compute min/max for each layer
           stats[layer_name] = {
               'act_min': activation_min,
               'act_max': activation_max,
               'weight_min': weight_min,
               'weight_max': weight_max
           }

3. QUANTIZATION (8-bit signed)
   For each layer:
       scale = 127 / max(|activation|)
       q_activation = round(activation * scale) / scale
   
   Per-layer scale factors exported as float32 (used in HLS)

4. VALIDATION (Quantized Model)
   q_model = quantize_model(model, stats)
   accuracy_int8 = validate(q_model)  # Target: >86%
   
   If accuracy_int8 < 86%:
       → Use Quantization-Aware Training (QAT)
       → Simulate quantization during training
       → Expected: >87% (loss <1%)

5. EXPORT (C Arrays + Headers)
   export_weights_to_c(q_model, 'weights.h')
   # Generates:
   int8_t conv1_weights[3][3][3][8] = { ... };
   float conv1_scales[8] = { ... };
   // ... all layers

================================================================================
EXPECTED ACCURACY & LATENCY
================================================================================

Training Accuracy (Float32):        88–92%
Quantized Accuracy (INT8 PTQ):      86–90% (−2% from float)
Quantized Accuracy (INT8 QAT):      87–91% (−1% from float)

Latency breakdown (32 DSP configuration):
  Layer 0 (Conv1):             0.2 ms
  Layer 1 (Bottleneck A):      0.1 ms
  Layer 2 (Bottleneck B):      0.4 ms
  Layer 3 (Bottleneck C ×2):   0.3 ms
  Layer 4 (Conv1×1):           0.05 ms
  Layer 5 (GAP):               0.01 ms
  Temporal fusion:             0.01 ms
  FC layers:                   0.01 ms
  ─────────────────────────────
  Total compute:              ~1.1 ms
  + Control overhead:         ~0.5 ms
  + Memory stalls:            ~1.6 ms
  ─────────────────────────────
  Total per-inference:        ~3.2 ms
  
  Throughput: 1000 / 3.2 ≈ 312 FPS

With 64 DSPs: ~1.6 ms per inference (625 FPS)
With 8 DSPs: ~12.9 ms per inference (77 FPS)

Streaming throughput (video):
  At 30 FPS input: Easy (requires ~0.1% of chip)
  At 120 FPS input: Easy (requires ~0.4% of chip)
  At 600 FPS input: Requires buffering or temporal downsampling

================================================================================
RESOURCE UTILIZATION (XC7A100T)
================================================================================

Device Capacity:
  Logic Cells:  15,850 LUTs
  Flip-Flops:   31,700 FFs
  DSP48Es:      240
  BRAM36:       270 (4.86 MB total)
  I/O Pins:     210

This Network (Estimated):
  LUTs:   ~2,000 (13% capacity) ✓
  FFs:    ~1,500 (5% capacity) ✓
  DSPs:   32–80 (13–33% capacity) ✓
  BRAM:   ~20 blocks (7% capacity) ✓

Headroom:
  87% of LUTs available for:
    - Control logic
    - Preprocessing (resize)
    - Postprocessing (softmax)
    - DDR3 controller
    - Video input/output
    - Monitoring/debugging

================================================================================
NEXT STEPS
================================================================================

1. VERIFY DATABASE (Run check_database.py):
   ✓ Confirm T_actual and class distribution
   ✓ Identify any data format issues

2. IMPLEMENT TRAINING PIPELINE:
   ├─ Define model in PyTorch
   ├─ Train with DISTributed Data Parallel
   ├─ Validate accuracy >88%
   └─ Export weights

3. QUANTIZE & EXPORT:
   ├─ Calibrate on validation set
   ├─ Generate C arrays (int8 weights + scales)
   └─ Create golden test vectors

4. DEVELOP HLS KERNELS:
   ├─ Use network_kernels.cpp template
   ├─ Implement preprocessing/postprocessing
   ├─ Test C simulation
   └─ Synthesize for XC7A100T

5. HARDWARE INTEGRATION:
   ├─ Create Vivado project
   ├─ Integrate HLS core + DDR3
   ├─ Implement firmware
   └─ Deploy on Nexys DDR4

6. VALIDATION:
   ├─ Test on real video
   ├─ Measure latency
   ├─ Verify accuracy matches GPU
   └─ Optimize performance

================================================================================
CONTACT & REVISION HISTORY
================================================================================

Document Version: 1.0
Created: 2025
Architecture: TinyMobileNet-XS (T=8, K=2)
Target Device: Artix-7 XC7A100T
Status: Ready for Implementation

Questions?
  • Dataset structure → See check_database.py
  • Architecture details → See FPGA_Architecture_Spec.py
  • HLS template → See network_kernels.cpp
  • Training → See training.py (to be created)

================================================================================
"""

print(__doc__)
