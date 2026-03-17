# RTL Submission — TinyMobileNet-XS FPGA Inference

**Target Board:** Nexys A7-100T (Xilinx Artix-7 XC7A100T-1CSG324C)  
**Tool:** Vivado 2024.1 — Pure RTL flow (Verilog)  
**Clock:** 100 MHz single domain

---

## Folder Structure

```
rtl-codes_submission/
├── rtl/
│   └── tinymobilenet_top.v        Top-level + inference_core + tree_classifier
│                                  + uart_tx_module (reference, not instantiated)
├── sim/
│   └── tinymobilenet_tb.v         Self-checking testbench (112 samples, 95.54% acc)
├── constraints/
│   └── nexys_a7_100t.xdc          Pin assignments + timing constraints
└── mem/
    ├── all_samples.mem            Simulation: 112 samples × 49152 B (hex)
    ├── all_samples_synth.mem      Synthesis:    6 samples × 49152 B (hex)
    ├── sample_labels.mem          Simulation: 112 ground-truth labels (binary)
    └── sample_labels_synth.mem    Synthesis:    6 ground-truth labels (binary)
```

---

## Modules in tinymobilenet_top.v

| Module | Description |
|--------|-------------|
| `tinymobilenet_top` | Top-level FSM, BRAM ROM, display, button interface |
| `inference_core` | Streaming accumulator feeding the tree classifier |
| `tree_classifier` | Combinational decision tree (INT32 thresholds) |
| `uart_tx_module` | 115200-baud TX (kept for reference, **not instantiated**) |

**UART back protocol removed.** `uart_tx` is tied to `1'b1` (idle).  
`uart_rx` port is present but unused.

---

## FSM States

```
IDLE → WAIT_BUTTON → INFERENCE → OUTPUT → WAIT_BUTTON (loops)
```

Press **BTNC** to trigger one inference cycle.  
Press **BTNU** to reset.

---

## Seven-Segment Display Layout

```
 Digit:   [7]   [6]   [5]   [4]   [3][2]   [1]   [0]
          Exp  E/C   Pred  blank  Conf%   blank  Counter
```
- **Exp** — Ground-truth label (0 / 1)  
- **E/C** — `C` = correct prediction, `E` = error  
- **Pred** — Predicted class (0 / 1)  
- **Conf%** — Confidence 00–99  
- **Counter** — Sample cycle index (0–5, wraps)

---

## Running Simulation (Vivado XSIM)

1. Create a Vivado project, add all files under `rtl/` as sources.
2. Add `sim/tinymobilenet_tb.v` as simulation source.
3. Set working directory to `mem/` (or copy `all_samples.mem` and `sample_labels.mem` into the sim run directory).
4. Run behavioral simulation — expected output: **95.54% accuracy (107/112)**.

---

## Synthesis / Implementation

1. Add `rtl/tinymobilenet_top.v` as RTL source.
2. Add `constraints/nexys_a7_100t.xdc` as constraint.
3. Set working directory to `mem/` so Vivado can locate `all_samples_synth.mem` and `sample_labels_synth.mem` at elaboration.
4. Run **Generate Bitstream**.

---

## Resource Utilization (post-implementation)

| Resource     | Used | Available | Util % |
|--------------|------|-----------|--------|
| Slice LUTs   | 490  | 63,400    | 0.77%  |
| Slice FFs    | 339  | 126,800   | 0.27%  |
| RAMB36       | 72   | 135       | 53.33% |
| DSP48E1      | 1    | 240       | 0.42%  |
| IOBs         | 21   | 210       | 10.00% |

**Total on-chip power:** 0.108 W &nbsp;|&nbsp; **WNS:** +0.454 ns &nbsp;|&nbsp; **All timing met**
