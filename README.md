# Pedestrian Intention Estimation — FPGA Accelerated Inference

> Real-time pedestrian crossing-intention classification using a TinyMobileNet-XS decision-tree classifier, implemented as a pure RTL hardware accelerator on the **Nexys A7-100T** (Xilinx Artix-7 XC7A100T).

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Dataset Download](#3-dataset-download)
4. [Environment Setup](#4-environment-setup)
5. [Frame Extraction from Videos](#5-frame-extraction-from-videos)
6. [Database Generation](#6-database-generation)
7. [Model Training](#7-model-training)
8. [Model Evaluation](#8-model-evaluation)
9. [FPGA Test Vector Generation](#9-fpga-test-vector-generation)
10. [FPGA Implementation (Vivado)](#10-fpga-implementation-vivado)
11. [Running Simulation (XSIM)](#11-running-simulation-xsim)
12. [Programming the Board](#12-programming-the-board)
13. [UART Host Monitor](#13-uart-host-monitor)
14. [Resource Utilization & Results](#14-resource-utilization--results)

---

## 1. Project Overview

This project implements a **pedestrian crossing-intention estimator** that classifies whether a pedestrian will cross the road or not, using a sequence of image frames from the [PIE dataset](https://github.com/aras62/PIE).

### Pipeline Summary

```
PIE Videos  →  Frame Extraction  →  PyTorch Model Training  →  Decision Tree Extraction
     ↓
FPGA Test Vectors (.mem)  →  Vivado RTL Synthesis  →  Nexys A7-100T FPGA  →  Real-time Inference
```

### Key Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 95.54% (107/112 test samples) |
| **FPGA Clock** | 100 MHz (WNS +0.454 ns) |
| **End-to-end Latency** | ≈ 1,099 µs/sample |
| **Throughput** | ≈ 910 samples/second |
| **Total On-chip Power** | 0.108 W |
| **LUT Utilization** | 0.77% (490/63,400) |
| **BRAM Utilization** | 53.33% (72/135 RAMB36) |

---

## 2. Repository Structure

```
.
├── tinymobilenet_xs.py             # TinyMobileNet-XS PyTorch model definition
├── train.py                        # Training script
├── generate_fpga_test_vectors.py   # Generates .mem files from trained model
├── export_quantized_weights.py     # Exports quantized weights
├── quantize_and_compare_sets.py    # PTQ analysis and INT8 comparison
├── fpga_uart_monitor.py            # Host-side UART result monitor
├── fpga_host_interface.py          # FPGA host communication interface
├── split_train_test.py             # Train/test dataset splitter
├── comprehensive_evaluation.py     # Full accuracy evaluation
├── generate_pie_database.py        # PIE annotation → PKL database builder
├── requirements.txt                # Python dependencies
│
├── fpga/
│   ├── rtl/
│   │   └── tinymobilenet_top.v     # Top-level RTL (FSM + inference core + tree classifier)
│   ├── sim/
│   │   └── tinymobilenet_tb.v      # Self-checking testbench (112 samples)
│   ├── constraints/
│   │   └── nexys_a7_100t.xdc       # Nexys A7-100T pin + timing constraints
│   ├── recreate_project_clean.tcl  # Vivado project creation script
│   ├── update_and_rebuild.tcl      # Incremental rebuild + bitstream script
│   ├── run_sim_batch.tcl           # Batch simulation script
│   ├── run_update_rebuild.cmd      # Windows launcher for rebuild
│   ├── run_uart_monitor.cmd        # Windows launcher for UART monitor
│   ├── select_synth_samples.py     # Selects 6 BRAM-safe synthesis samples
│   ├── calculate_max_samples.py    # BRAM capacity calculator
│   └── prepare_weight_mems.py      # Prepares weight .mem files
│
├── fpga_test_vectors/
│   ├── all_samples.mem             # 112-sample hex dump (simulation)
│   ├── all_samples_synth.mem       # 6-sample hex dump (synthesis / BRAM)
│   ├── sample_labels.mem           # 112 ground-truth labels (binary)
│   ├── sample_labels_synth.mem     # 6 ground-truth labels (binary)
│   └── test_summary.json           # Inference metadata per sample
│
├── fpga_weights/
│   ├── tinymobilenet_xs_weights.h  # C/H weight header for firmware use
│   └── quantization_config.json    # INT8 quantization configuration
│
├── checkpoints/
│   └── best_model.pth              # Trained PyTorch model weights (0.2 MB)
│
├── rtl-codes_submission/           # Clean standalone RTL submission folder
│   ├── rtl/tinymobilenet_top.v
│   ├── sim/tinymobilenet_tb.v
│   ├── constraints/nexys_a7_100t.xdc
│   └── mem/                        # Simulation + synthesis .mem files
│
├── utilities/                      # PIE dataset utilities (from official repo)
│   ├── pie_data.py                 # PIE data loader
│   ├── jaad_data.py                # JAAD data loader
│   ├── data_gen_utils.py           # Annotation processing helpers
│   └── configs.yaml                # Dataset path configuration
│
├── RTL_ARCHITECTURE_SPECIFICATION.txt
├── FPGA_IMPLEMENTATION_RESULTS.md
├── FPGA_IMPLEMENTATION_GUIDE.md
└── waveform_final_simulation.wcfg  # Vivado waveform configuration
```

---

## 3. Dataset Download

This project uses the **PIE (Pedestrian Intention Estimation)** dataset.

### 3.1 Download PIE Dataset

1. Visit the official PIE repository: **https://github.com/aras62/PIE**
2. Read and accept the dataset license agreement on the project page.
3. Request access by emailing the authors (link on the PIE repo) — you will receive a download link.
4. Download the following:
   - **Video clips:** `set01/`, `set02/`, `set05/` (≈ 10 GB total)
   - **Annotations:** Available directly from this repo under `annotations/`

### 3.2 Place Dataset Files

After downloading, place the video folders in the project root:

```
PIEs/
├── set01/
│   ├── video_0001.mp4
│   ├── video_0002.mp4
│   └── ...
├── set02/
│   └── ...
└── set05/
    └── ...
```

### 3.3 Configure Dataset Paths

Edit `utilities/configs.yaml` and set the root path to your local project directory:

```yaml
pie_path: 'C:/path/to/PIEs'   # Windows
# or
pie_path: '/home/user/PIEs'   # Linux/Mac
```

---

## 4. Environment Setup

### Requirements

- Python 3.10+
- PyTorch 2.x (CPU is sufficient; CUDA optional)
- Xilinx Vivado 2024.1 (for FPGA implementation)

### Install Python Dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` covers: `torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`, `Pillow`, `tqdm`, `matplotlib`, `pyserial`.

---

## 5. Frame Extraction from Videos

The PIE dataset is distributed as `.mp4` video clips. You must extract individual frames before training.

### 5.1 Extract Set02 Frames (Primary Training Set)

```bash
python extract_set02_frames.py
```

This reads `set02/video_*.mp4`, extracts frames at the annotated timestamps, resizes each pedestrian bounding-box crop to **64 × 64 RGB**, and saves them under `images/set02/`.

For faster extraction using GPU-accelerated decoding:

```bash
python extract_set02_gpu.py
```

### 5.2 Extract Set05 Frames (Test Set)

```bash
python extract_set05.py
```

### 5.3 Verify Extraction

```bash
python check_extraction_status.py
```

Expected output: counts of extracted frames per set, with no missing sequences.

### 5.4 Frame Format

Each extracted frame is:
- **Size:** 64 × 64 pixels
- **Channels:** RGB (3 channels)
- **Format:** JPEG or PNG under `images/<set>/<video_id>/<pedestrian_id>/<frame_id>.jpg`

---

## 6. Database Generation

The PIE annotations are in XML format. This step converts them to a serialised Python PKL database used by the model loader.

```bash
python generate_pie_database.py
```

This calls `utilities/pie_data.py` internally and produces `data_cache/pie_database.pkl`.

### Verify Database Structure

```bash
python check_database.py
```

Expected output:
```
Total sequences  : 112 (test split)
Class distribution: 0=Not-Crossing (56), 1=Crossing (56)
Input shape      : (T, 64, 64, 3)  T = variable (8–32 frames)
```

---

## 7. Model Training

### 7.1 Architecture — TinyMobileNet-XS

Defined in `tinymobilenet_xs.py`:

| Layer | Operation | Input → Output |
|-------|-----------|---------------|
| 0 | Conv 3×3 stride-2 | 64×64×3 → 32×32×8 |
| 1 | Bottleneck-A (DW) | 32×32×8 |
| 2 | Bottleneck-B (DW stride-2, expand×4) | 16×16×12 |
| 3 | Bottleneck-C ×2 + **TSM** | 8×8×16 |
| 4 | Conv 1×1 | 8×8×48 |
| 5 | Global Avg Pool | 48-dim vector |
| FC1 | Linear + ReLU | 48 → 32 |
| FC2 | Linear | 32 → 2 logits |

**Total parameters:** 7,232 | **MACs/frame:** 1.29 M | **Quantization:** INT8

### 7.2 Train

```bash
python train.py
```

Default configuration:
- **Optimizer:** Adam, lr=1e-3
- **Epochs:** 50
- **Batch size:** 32
- **Sequence length (T):** 4 frames
- **Input:** 64×64×3 RGB, normalised to [0, 1]

Best checkpoint saved to `checkpoints/best_model.pth`.

### 7.3 Monitor Training

```bash
python training_monitor.py
```

### 7.4 Hardware-Optimised Decision Tree (for FPGA)

The full CNN model is too complex for a lightweight RTL accelerator. Instead, a **decision tree classifier** is extracted from the trained model:

```bash
python tmp_tree_hw.py
```

This extracts 8 statistical features (`se`, `so`, `q0–q3`, `hi`, `lo`) from the raw pixel data and trains a scikit-learn `DecisionTreeClassifier`. The resulting tree (thresholds as INT32 values) is then manually transcribed into the combinational `tree_classifier` module in `fpga/rtl/tinymobilenet_top.v`.

---

## 8. Model Evaluation

### Evaluate on Full Test Set (112 samples)

```bash
python comprehensive_evaluation.py
```

Expected: **95.54% accuracy** (107/112).

### Quantization Analysis

```bash
python quantize_and_compare_sets.py
```

Compares FP32 vs INT8 post-training quantization. Note: naive INT8 PTQ degrades accuracy to ~18% — use QAT (quantization-aware training) for production.

---

## 9. FPGA Test Vector Generation

This step converts the raw 64×64 RGB test sequences into flat hex memory files (`.mem`) that Vivado loads into BRAM.

```bash
python generate_fpga_test_vectors.py
```

**Outputs** (written to `fpga_test_vectors/`):

| File | Description | Size |
|------|-------------|------|
| `all_samples.mem` | 112 samples hex dump (simulation) | 21 MB |
| `all_samples_synth.mem` | 6 samples hex dump (synthesis) | 1.1 MB |
| `sample_labels.mem` | 112 binary ground-truth labels | 336 B |
| `sample_labels_synth.mem` | 6 binary ground-truth labels | 18 B |
| `test_summary.json` | Per-sample confidence metadata | 59 KB |

### Select Synthesis Samples (BRAM-safe subset)

The Artix-7 100T has 135 RAMB36 blocks. Storing all 112 samples (5.5 MB) would require 272 blocks — exceeding capacity. The script selects 6 well-balanced samples that fit within 53% BRAM:

```bash
python fpga/select_synth_samples.py
```

---

## 10. FPGA Implementation (Vivado)

### Requirements

- **Xilinx Vivado Design Suite 2024.1** (free WebPACK edition is sufficient)
- Windows or Linux host
- Nexys A7-100T board

### 10.1 Install Vivado

Download from: **https://www.xilinx.com/support/download.html**  
Choose "Vivado ML Edition" → version 2024.1 → WebPACK (free, no licence required for Artix-7).

### 10.2 Create the Vivado Project (first time)

```cmd
cd fpga
vivado -mode batch -source recreate_project_clean.tcl
```

This creates `fpga/vivado_project_<timestamp>/tinymobilenet_fpga.xpr` with all sources, constraints, and simulation sets pre-configured.

### 10.3 Rebuild / Generate Bitstream (after any RTL change)

```cmd
cd fpga
run_update_rebuild.cmd
```

Or manually:

```cmd
vivado -mode batch -source update_and_rebuild.tcl
```

This runs synthesis → implementation → bitstream generation. Takes **10–15 minutes** on a modern laptop.

### 10.4 Source Files Required by Vivado

| File | Role |
|------|------|
| `fpga/rtl/tinymobilenet_top.v` | RTL design (all modules in one file) |
| `fpga/constraints/nexys_a7_100t.xdc` | Pin assignments + 100 MHz clock constraint |
| `fpga_test_vectors/all_samples_synth.mem` | BRAM initialisation (synthesis) |
| `fpga_test_vectors/sample_labels_synth.mem` | Label ROM initialisation |

> **Important:** Set the Vivado project's working directory (or simulation run directory) to the folder containing the `.mem` files so `$readmemh()` / `$readmemb()` can locate them.

### 10.5 RTL Module Hierarchy

```
tinymobilenet_top          ← Top-level FSM, BRAM ROM, display, UART (idle)
  └── inference_core       ← Byte-streaming accumulator
        └── tree_classifier ← Combinational INT32 decision tree
```

### 10.6 Seven-Segment Display Layout

```
Digit position:  [7]    [6]   [5]    [4]    [3][2]   [1]   [0]
                 Exp   E/C   Pred  blank   Conf %   blank  Count
```

- **Exp** — Ground-truth label (0 = Not-Crossing, 1 = Crossing)
- **E/C** — `C` = Correct, `E` = Error (mismatch)
- **Pred** — Predicted class
- **Conf %** — Confidence score (00–99)
- **Count** — Sample cycle index (0–5, wraps)

Press **BTNC** to run next inference. Press **BTNU** to reset.

---

## 11. Running Simulation (XSIM)

### Batch Simulation (all 112 samples)

```cmd
cd fpga
vivado -mode batch -source run_sim_batch.tcl
```

Runs the self-checking testbench `fpga/sim/tinymobilenet_tb.v` which:
- Loads `all_samples.mem` (112 × 49,152 bytes)
- Triggers each inference via button pulse
- Compares predicted vs ground-truth for every sample
- Prints a final accuracy summary

Expected output:
```
FINAL RESULT: TOTAL=112 CORRECT=107 WRONG=5 ACC=95.54%
```

### Open in Vivado GUI (waveforms)

1. Open `fpga/vivado_project_<timestamp>/tinymobilenet_fpga.xpr`
2. Flow Navigator → Run Simulation → Run Behavioral Simulation
3. Load `waveform_final_simulation.wcfg` for pre-configured signal groups

---

## 12. Programming the Board

1. Connect Nexys A7-100T via USB-JTAG cable.
2. Open Vivado → Hardware Manager → Open Target → Auto Connect.
3. Click **Program Device** → select `tinymobilenet_top.bit` from:
   ```
   fpga/vivado_project_<timestamp>/tinymobilenet_fpga.runs/impl_1/
   ```
4. Press **BTNU** once to reset, then **BTNC** to start each inference cycle.

---

## 13. UART Host Monitor

Results are streamed over UART at 115,200 baud (USB-UART via CP2102 on Nexys A7).

```cmd
cd fpga
run_uart_monitor.cmd
```

Or directly:

```bash
python fpga_uart_monitor.py --port COM3 --baud 115200
```

Replace `COM3` with the actual COM port assigned to your board (check Device Manager on Windows).

> **Note:** The UART back-protocol (reset notification packet) has been removed from the current RTL. `uart_tx` is tied idle. Only the on-board seven-segment display provides real-time output. UART monitoring is available if the protocol is re-enabled in `uart_tx_module`.

---

## 14. Resource Utilization & Results

### Post-Implementation (Vivado 2024.1, XC7A100T-1CSG324C)

| Resource | Used | Available | Util % |
|----------|------|-----------|--------|
| Slice LUTs | 490 | 63,400 | **0.77%** |
| Slice Registers (FFs) | 339 | 126,800 | **0.27%** |
| RAMB36 | 72 | 135 | **53.33%** |
| DSP48E1 | 1 | 240 | **0.42%** |
| IOBs | 21 | 210 | **10.00%** |

### Timing

| Metric | Value |
|--------|-------|
| Clock | 100 MHz |
| WNS | +0.454 ns ✅ |
| All endpoints met | ✅ |

### Power (Vivado estimate)

| Component | Power |
|-----------|-------|
| Total on-chip | **0.108 W** |
| Dynamic | 0.008 W |
| Static (leakage) | 0.100 W |

### Performance vs Software

| | SW (CPU, single-thread) | **FPGA Hardware** | Speedup |
|--|------------------------|-------------------|---------|
| Latency | 15–25 ms | **≈ 1,099 µs** | **~20×** |
| Throughput | 40–65 samples/s | **≈ 910 samples/s** | **~14×** |
| Power | ~65 W | **0.108 W** | **~600× lower** |
| Accuracy | 95.54% | **95.54%** | — |

---
