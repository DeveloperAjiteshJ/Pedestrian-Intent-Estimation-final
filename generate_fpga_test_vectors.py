"""
Generate INT8 test vectors from trained PyTorch model
Exports test data in both Python pickle and Verilog hex format for FPGA BRAM
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from pathlib import Path
import json
from tinymobilenet_xs import create_model

# Configuration
CHECKPOINT_PATH = './checkpoints/best_model.pth'
QUANT_CONFIG_PATH = './fpga_weights/quantization_config.json'
TEST_PKL_FILES = [
    './data_cache/sequences/set02_sequences.pkl',
    './data_cache/sequences/test_sequences.pkl'
]
TESTSET_NAME = 'set02+set05'
MAX_SAMPLES = None
DECISION_THRESHOLD = 0.27
PIE_ROOT = '.'
OUTPUT_DIR = './fpga_test_vectors'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 1. Load quantization config
# ============================================================================

print("="*80)
print("GENERATING FPGA TEST VECTORS")
print("="*80)

with open(QUANT_CONFIG_PATH, 'r') as f:
    quant_config = json.load(f)

print(f"\n✅ Loaded quantization config from {QUANT_CONFIG_PATH}")
print(f"   Weight scales: {len(quant_config['weight_scales'])} layers")
print(f"   Activation scales: {len(quant_config['activation_scales'])} layers")
print(f"   Decision threshold (class 1): {DECISION_THRESHOLD:.2f}")

# ============================================================================
# 2. Load trained model and prepare for inference
# ============================================================================

print("\nLoading trained model...")
model = create_model(num_classes=2, t_frames=4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Model loaded from {CHECKPOINT_PATH}")
print(f"   Best validation accuracy: {checkpoint['best_acc']:.2f}%")

# ============================================================================
# 3. Load test dataset
# ============================================================================

print("\nLoading test dataset...")

class TestDataset(torch.utils.data.Dataset):
    """Minimal dataset loader for test samples"""
    def __init__(self, pkl_files, pie_root, max_frames=4, split='test', max_samples=None):
        self.pkl_files = pkl_files if isinstance(pkl_files, (list, tuple)) else [pkl_files]
        self.pie_root = pie_root
        self.max_frames = max_frames

        self.image_sequences = []
        self.labels = []
        for pkl_file in self.pkl_files:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            self.image_sequences.extend(data['image'])
            self.labels.extend(data['intention_binary'])

        if max_samples is not None:
            self.image_sequences = self.image_sequences[:max_samples]
            self.labels = self.labels[:max_samples]
        
        self.failed = 0
        self.real_images = 0
        self.total = 0
        
    def __len__(self):
        return len(self.image_sequences)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        image_paths = self.image_sequences[idx][:self.max_frames]
        label = int(self.labels[idx][0][0])
        
        frames = []
        for path in image_paths:
            try:
                if not os.path.exists(path):
                    # Try to reconstruct path
                    filename = Path(path).name
                    for root, dirs, files in os.walk(os.path.join(self.pie_root, 'images')):
                        if filename in files:
                            path = os.path.join(root, filename)
                            break
                
                img = Image.open(path).convert('RGB')
                if img.size != (64, 64):
                    img = img.resize((64, 64), Image.BILINEAR)
                
                img_array = np.array(img, dtype=np.float32) / 255.0
                self.real_images += 1
                frames.append(img_array)
            except:
                frames.append(np.zeros((64, 64, 3), dtype=np.float32))
                self.failed += 1
            
            self.total += 1
        
        # Pad if needed
        while len(frames) < self.max_frames:
            frames.append(np.zeros((64, 64, 3), dtype=np.float32))
        
        frames = np.stack(frames, axis=0)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
        
        return frames, label, idx

# Load dataset
dataset = TestDataset(TEST_PKL_FILES, PIE_ROOT, max_frames=4, split='test', max_samples=MAX_SAMPLES)
label_list = [int(x[0][0]) for x in dataset.labels]
label_counts = {0: int(sum(l == 0 for l in label_list)), 1: int(sum(l == 1 for l in label_list))}
print(f"✅ Loaded {len(dataset)} test samples from {TESTSET_NAME}")
for p in TEST_PKL_FILES:
    print(f"   - {p}")
print(f"   Label distribution: {label_counts}")

# ============================================================================
# 4. Generate FP32 and INT8 test vectors
# ============================================================================

print("\nGenerating test vectors...")
print("-" * 80)

test_vectors_fp32 = []
test_vectors_int8 = []
predictions_fp32 = []
predictions_int8 = []
ground_truth = []
sample_indices = []

# Activation scale for quantization
activation_scales = quant_config.get('activation_scales', {})
weight_scales = quant_config.get('weight_scales', {})
gap_scale = float(activation_scales.get('gap', 1.0))
relu_scale = float(activation_scales.get('relu', 1.0))
fc2_scale = float(weight_scales.get('fc2', 1.0))

captured_activations = {'gap': None}

def gap_hook(module, inp, out):
    captured_activations['gap'] = out.detach()

gap_handle = model.gap.register_forward_hook(gap_hook)

with torch.no_grad():
    for idx in range(len(dataset)):
        frames, label, sample_idx = dataset[idx]
        frames = frames.unsqueeze(0).to(device)  # Add batch dimension
        
        # FP32 inference
        output_fp32 = model(frames)
        logits_fp32 = output_fp32[0].cpu().numpy()

        # Softmax to get probabilities
        probs_fp32 = np.exp(logits_fp32) / np.sum(np.exp(logits_fp32))

        # Get prediction using tuned binary threshold on class-1 probability
        pred_fp32 = int(probs_fp32[1] >= DECISION_THRESHOLD)
        conf_fp32 = float(probs_fp32[1] if pred_fp32 == 1 else probs_fp32[0])
        
        # Quantize input test vector for FPGA
        frames_uint8 = (frames.cpu().numpy() * 255).astype(np.uint8)

        # Simulated INT8 head inference with activation scales
        gap_out = captured_activations['gap']
        if gap_out is not None:
            gap_feat = gap_out.view(1, model.t_frames, 48).mean(dim=1)
            gap_int8 = torch.clamp(torch.round(gap_feat * gap_scale), -128, 127)
            gap_dequant = gap_int8 / gap_scale

            fc1_out = model.fc1(gap_dequant)
            relu_out = torch.relu(fc1_out)
            relu_int8 = torch.clamp(torch.round(relu_out * relu_scale), -128, 127)
            relu_dequant = relu_int8 / relu_scale

            logits_sim = model.fc2(relu_dequant)[0].cpu().numpy()
        else:
            logits_sim = logits_fp32

        # Final FC2 quantization (correct formula: q = round(fp32 * scale))
        logits_int8 = np.clip(np.round(logits_sim * fc2_scale), -128, 127).astype(np.int8)
        logits_int8_dequant = logits_int8.astype(np.float32) / fc2_scale
        logits_int8_shift = logits_int8_dequant - np.max(logits_int8_dequant)
        probs_int8 = np.exp(logits_int8_shift) / np.sum(np.exp(logits_int8_shift))
        pred_int8 = int(probs_int8[1] >= DECISION_THRESHOLD)
        conf_int8_pct = int(np.clip(np.round(100.0 * (probs_int8[1] if pred_int8 == 1 else probs_int8[0])), 0, 99))

        # Store results
        test_vectors_fp32.append(frames[0].cpu().numpy())  # (4, 3, 64, 64)
        test_vectors_int8.append(frames_uint8[0])  # (4, 3, 64, 64)
        
        predictions_fp32.append({
            'logits': logits_fp32.tolist(),
            'probs': probs_fp32.tolist(),
            'pred': int(pred_fp32),
            'confidence': conf_fp32
        })
        
        predictions_int8.append({
            'logits': logits_int8.tolist(),
            'logits_dequant': logits_int8_dequant.tolist(),
            'pred': int(pred_int8),
            'confidence_pct': int(conf_int8_pct)
        })
        
        ground_truth.append(int(label))
        sample_indices.append(int(sample_idx))
        
        # Print progress
        if (idx + 1) % 5 == 0:
            match_fp32 = "✅" if pred_fp32 == label else "❌"
            match_int8 = "✅" if pred_int8 == label else "❌"
            print(f"Sample {idx+1:2d}/{len(dataset)}: GT={label} | FP32={pred_fp32}{match_fp32} | INT8={pred_int8}{match_int8}")

gap_handle.remove()
# ============================================================================
# 5. Calculate accuracy metrics
# ============================================================================

print("\n" + "="*80)
print("ACCURACY METRICS")
print("="*80)

acc_fp32 = 100 * np.mean(np.array([p['pred'] for p in predictions_fp32]) == np.array(ground_truth))
acc_int8 = 100 * np.mean(np.array([p['pred'] for p in predictions_int8]) == np.array(ground_truth))

print(f"\nFP32 Accuracy: {acc_fp32:.2f}%")
print(f"INT8 Accuracy: {acc_int8:.2f}%")
print(f"Accuracy Drop: {acc_fp32 - acc_int8:.2f}%")

# Per-class accuracy
print("\nPer-class accuracy (FP32):")
for class_id in [0, 1]:
    mask = np.array(ground_truth) == class_id
    if np.sum(mask) > 0:
        acc = 100 * np.mean(np.array([p['pred'] for p in predictions_fp32])[mask] == class_id)
        count = np.sum(mask)
        print(f"  Class {class_id} (count={count}): {acc:.2f}%")

# ============================================================================
# 6. Export to Python pickle
# ============================================================================

print("\n" + "="*80)
print("EXPORTING TEST VECTORS")
print("="*80)

# Save comprehensive pickle
test_data = {
    'test_vectors_fp32': np.array(test_vectors_fp32),  # (N, 4, 3, 64, 64)
    'test_vectors_int8': np.array(test_vectors_int8),  # (N, 4, 3, 64, 64)
    'predictions_fp32': predictions_fp32,
    'predictions_int8': predictions_int8,
    'ground_truth': ground_truth,
    'sample_indices': sample_indices,
    'accuracy_fp32': float(acc_fp32),
    'accuracy_int8': float(acc_int8),
    'accuracy_drop': float(acc_fp32 - acc_int8)
}

pickle_file = os.path.join(OUTPUT_DIR, 'fpga_test_vectors.pkl')
with open(pickle_file, 'wb') as f:
    pickle.dump(test_data, f)

print(f"✅ Saved to {pickle_file}")
print(f"   {len(test_vectors_fp32)} test samples")

# ============================================================================
# 7. Export to Verilog HEX format (for BRAM initialization)
# ============================================================================

print("\nExporting to Verilog HEX format...")

# For each test sample, create a HEX file
for sample_idx in range(len(test_vectors_int8)):
    sample_data = test_vectors_int8[sample_idx]  # (4, 3, 64, 64)

    # Flatten to 1D: (4 * 3 * 64 * 64,) = (49152,)
    flat_data = sample_data.flatten()

    # Convert to hex string (2 hex chars per byte)
    hex_data = ''.join([f'{int(b):02x}' for b in flat_data.astype(np.uint8)])

    # Save as Verilog .mem file (for simulation)
    mem_file = os.path.join(OUTPUT_DIR, f'test_sample_{sample_idx:02d}.mem')
    with open(mem_file, 'w') as f:
        # Write as 32-bit words (4 bytes per line)
        for i in range(0, len(hex_data), 8):
            f.write(hex_data[i:i+8].upper() + '\n')

    print(f"  ✅ test_sample_{sample_idx:02d}.mem ({len(flat_data)} bytes)")

# Combined sample ROM (one byte per line, all samples contiguous)
all_samples_mem = os.path.join(OUTPUT_DIR, 'all_samples.mem')
with open(all_samples_mem, 'w') as f:
    for sample_data in test_vectors_int8:
        flat_data = sample_data.flatten().astype(np.uint8)
        for b in flat_data:
            f.write(f'{int(b):02X}\n')
print(f"  ✅ all_samples.mem ({len(test_vectors_int8)} samples contiguous)")

# Ground-truth labels ROM (binary, one label per line)
labels_mem = os.path.join(OUTPUT_DIR, 'sample_labels.mem')
with open(labels_mem, 'w') as f:
    for lbl in ground_truth:
        f.write(f'{int(lbl) & 1:b}\n')
print(f"  ✅ sample_labels.mem ({len(ground_truth)} labels)")

# Predicted labels ROM from INT8 evaluation (binary, one prediction per line)
preds_mem = os.path.join(OUTPUT_DIR, 'sample_predictions.mem')
with open(preds_mem, 'w') as f:
    for pred in predictions_int8:
        f.write(f"{int(pred['pred']) & 1:b}\n")
print(f"  ✅ sample_predictions.mem ({len(predictions_int8)} predictions)")

# Confidence ROM from INT8 evaluation (hex byte, one confidence per line)
conf_mem = os.path.join(OUTPUT_DIR, 'sample_confidence.mem')
with open(conf_mem, 'w') as f:
    for pred in predictions_int8:
        f.write(f"{int(pred['confidence_pct']) & 0xFF:02X}\n")
print(f"  ✅ sample_confidence.mem ({len(predictions_int8)} confidences)")

# ============================================================================
# 8. Export C header for reference
# ============================================================================

print("\nExporting C header...")

c_header = """// AUTO-GENERATED FPGA TEST VECTORS
// TinyMobileNet-XS Inference Test Data
// Generated from trained PyTorch model

#ifndef __FPGA_TEST_VECTORS_H__
#define __FPGA_TEST_VECTORS_H__

#include <stdint.h>

// Test dataset statistics
#define NUM_TEST_SAMPLES {}
#define FRAME_WIDTH 64
#define FRAME_HEIGHT 64
#define FRAME_CHANNELS 3
#define FRAMES_PER_SAMPLE 4
#define BYTES_PER_SAMPLE {} // 4 * 3 * 64 * 64

// Ground truth labels
static const uint8_t test_ground_truth[NUM_TEST_SAMPLES] = {{
{}
}};

// Expected FP32 predictions
static const uint8_t test_predictions_fp32[NUM_TEST_SAMPLES] = {{
{}
}};

// Expected INT8 predictions
static const uint8_t test_predictions_int8[NUM_TEST_SAMPLES] = {{
{}
}};

// Accuracy metrics
#define TEST_ACCURACY_FP32 {:.2f}
#define TEST_ACCURACY_INT8 {:.2f}
#define ACCURACY_DROP {:.2f}

#endif // __FPGA_TEST_VECTORS_H__
""".format(
    len(ground_truth),
    4 * 3 * 64 * 64,
    ', '.join(map(str, ground_truth)),
    ', '.join([str(p['pred']) for p in predictions_fp32]),
    ', '.join([str(p['pred']) for p in predictions_int8]),
    acc_fp32,
    acc_int8,
    acc_fp32 - acc_int8
)

header_file = os.path.join(OUTPUT_DIR, 'fpga_test_vectors.h')
with open(header_file, 'w') as f:
    f.write(c_header)

print(f"✅ Saved to {header_file}")

# ============================================================================
# 9. Export test summary JSON
# ============================================================================

summary = {
    'dataset_name': TESTSET_NAME,
    'dataset_path': TEST_PKL_FILES,
    'decision_threshold_class1': DECISION_THRESHOLD,
    'num_samples': len(ground_truth),
    'label_counts': label_counts,
    'accuracy_fp32': float(acc_fp32),
    'accuracy_int8': float(acc_int8),
    'accuracy_drop': float(acc_fp32 - acc_int8),
    'quantization_config': quant_config,
    'predictions': {
        'fp32': predictions_fp32,
        'int8': predictions_int8
    },
    'ground_truth': ground_truth
}

summary_file = os.path.join(OUTPUT_DIR, 'test_summary.json')
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✅ Saved summary to {summary_file}")

# ============================================================================
# 10. Print final summary
# ============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"\n📊 Test Vectors Generated:")
print(f"   Dataset: {TESTSET_NAME}")
print(f"   Decision threshold: {DECISION_THRESHOLD:.2f}")
print(f"   Total samples: {len(ground_truth)}")
print(f"   Label distribution: {label_counts}")
print(f"   FP32 Accuracy: {acc_fp32:.2f}%")
print(f"   INT8 Accuracy: {acc_int8:.2f}%")
print(f"   Accuracy Drop: {acc_fp32 - acc_int8:.2f}%")

print(f"\n📁 Output files saved to {OUTPUT_DIR}/:")
print(f"   ✅ fpga_test_vectors.pkl - All test data + predictions")
print(f"   ✅ fpga_test_vectors.h - C header with ground truth")
print(f"   ✅ test_summary.json - Summary statistics")
print(f"   ✅ test_sample_*.mem - Individual samples (Verilog format)")
print(f"   ✅ all_samples.mem - Contiguous byte ROM for RTL")
print(f"   ✅ sample_labels.mem - Ground-truth label ROM for RTL")
print(f"   ✅ sample_predictions.mem - Predicted class ROM for RTL")
print(f"   ✅ sample_confidence.mem - Confidence ROM for RTL")

print("\n✅ Test vectors ready for FPGA deployment!")

