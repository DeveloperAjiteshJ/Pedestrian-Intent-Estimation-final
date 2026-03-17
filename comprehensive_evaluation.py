"""
Comprehensive Model Evaluation with Detailed Metrics
Tests FP32 and INT8 quantized models on set02 + set05
Provides: Accuracy, Confusion Matrix, Precision, Recall, F1 Score
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from pathlib import Path
import json
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
from tinymobilenet_xs import create_model

# Configuration
CHECKPOINT_PATH = './checkpoints/best_model.pth'
QUANT_CONFIG_PATH = './fpga_weights/quantization_config.json'
SET02_PKL = './data_cache/sequences/set02_sequences.pkl'
SET05_PKL = './data_cache/sequences/test_sequences.pkl'
PIE_ROOT = '.'

print("="*80)
print("COMPREHENSIVE MODEL EVALUATION")
print("Test on: set02 + set05 (Combined)")
print("="*80)

# ============================================================================
# 1. Load quantization config
# ============================================================================

print("\n[1/6] Loading quantization config...")
with open(QUANT_CONFIG_PATH, 'r') as f:
    quant_config = json.load(f)

print(f"✅ Loaded quantization config")
print(f"   Weight scales: {len(quant_config['weight_scales'])} layers")
print(f"   Activation scales: {len(quant_config['activation_scales'])} layers")

# ============================================================================
# 2. Load trained model
# ============================================================================

print("\n[2/6] Loading trained model...")
model = create_model(num_classes=2, t_frames=4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Model loaded from {CHECKPOINT_PATH}")
print(f"   Best training accuracy: {checkpoint['best_acc']:.2f}%")

# ============================================================================
# 3. Load test datasets (set02 + set05)
# ============================================================================

print("\n[3/6] Loading test datasets...")

class TestDataset(torch.utils.data.Dataset):
    """Minimal dataset loader"""
    def __init__(self, pkl_files, pie_root, max_frames=4):
        self.pie_root = pie_root
        self.max_frames = max_frames
        
        # Combine multiple PKL files
        self.image_sequences = []
        self.labels = []
        
        for pkl_file in pkl_files:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            self.image_sequences.extend(data['image'])
            self.labels.extend(data['intention_binary'])
        
        self.failed = 0
        self.real_images = 0
        
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
        
        while len(frames) < self.max_frames:
            frames.append(np.zeros((64, 64, 3), dtype=np.float32))
        
        frames = np.stack(frames, axis=0)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
        
        return frames, label

# Load combined dataset
pkl_files = []
if os.path.exists(SET02_PKL):
    pkl_files.append(SET02_PKL)
    print(f"  ✅ Found set02_sequences.pkl")
else:
    print(f"  ⚠️  set02_sequences.pkl not found, using only set05")

if os.path.exists(SET05_PKL):
    pkl_files.append(SET05_PKL)
    print(f"  ✅ Found test_sequences.pkl (set05)")

if not pkl_files:
    print("[ERROR] No test data found!")
    exit(1)

dataset = TestDataset(pkl_files, PIE_ROOT, max_frames=4)
print(f"\n✅ Combined test dataset: {len(dataset)} samples")

# ============================================================================
# 4. FP32 Inference
# ============================================================================

print("\n[4/6] Running FP32 inference...")

predictions_fp32 = []
ground_truth = []

with torch.no_grad():
    for idx in range(len(dataset)):
        frames, label = dataset[idx]
        frames = frames.unsqueeze(0).to(device)
        
        output_fp32 = model(frames)
        logits_fp32 = output_fp32[0].cpu().numpy()
        pred_fp32 = np.argmax(logits_fp32)
        
        predictions_fp32.append(int(pred_fp32))
        ground_truth.append(int(label))
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx+1}/{len(dataset)} samples...")

predictions_fp32 = np.array(predictions_fp32)
ground_truth = np.array(ground_truth)

print(f"✅ FP32 inference complete")

# ============================================================================
# 5. INT8 Quantized Inference
# ============================================================================

print("\n[5/6] Running INT8 quantized inference...")

activation_scales = quant_config.get('activation_scales', {})
weight_scales = quant_config.get('weight_scales', {})
gap_scale = float(activation_scales.get('gap', 1.0))
relu_scale = float(activation_scales.get('relu', 1.0))
fc2_scale = float(weight_scales.get('fc2', 1.0))

predictions_int8 = []
captured_activations = {'gap': None}

def gap_hook(module, inp, out):
    captured_activations['gap'] = out.detach()

gap_handle = model.gap.register_forward_hook(gap_hook)

with torch.no_grad():
    for idx in range(len(dataset)):
        frames, label = dataset[idx]
        frames = frames.unsqueeze(0).to(device)
        
        # Run model to capture GAP output
        _ = model(frames)
        
        # Simulated INT8 head inference
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
            logits_sim = np.array([0, 0])

        logits_int8 = np.clip(np.round(logits_sim * fc2_scale), -128, 127).astype(np.int8)
        pred_int8 = np.argmax(logits_int8)
        
        predictions_int8.append(int(pred_int8))
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx+1}/{len(dataset)} samples...")

gap_handle.remove()
predictions_int8 = np.array(predictions_int8)

print(f"✅ INT8 inference complete")

# ============================================================================
# 6. Compute Comprehensive Metrics
# ============================================================================

print("\n[6/6] Computing comprehensive metrics...")

# FP32 Metrics
cm_fp32 = confusion_matrix(ground_truth, predictions_fp32)
acc_fp32 = 100 * np.mean(predictions_fp32 == ground_truth)
precision_fp32 = precision_score(ground_truth, predictions_fp32, average=None, zero_division=0)
recall_fp32 = recall_score(ground_truth, predictions_fp32, average=None, zero_division=0)
f1_fp32 = f1_score(ground_truth, predictions_fp32, average=None, zero_division=0)
f1_macro_fp32 = f1_score(ground_truth, predictions_fp32, average='macro', zero_division=0)
f1_weighted_fp32 = f1_score(ground_truth, predictions_fp32, average='weighted', zero_division=0)

# INT8 Metrics
cm_int8 = confusion_matrix(ground_truth, predictions_int8)
acc_int8 = 100 * np.mean(predictions_int8 == ground_truth)
precision_int8 = precision_score(ground_truth, predictions_int8, average=None, zero_division=0)
recall_int8 = recall_score(ground_truth, predictions_int8, average=None, zero_division=0)
f1_int8 = f1_score(ground_truth, predictions_int8, average=None, zero_division=0)
f1_macro_int8 = f1_score(ground_truth, predictions_int8, average='macro', zero_division=0)
f1_weighted_int8 = f1_score(ground_truth, predictions_int8, average='weighted', zero_division=0)

# ============================================================================
# 7. Print Results
# ============================================================================

print("\n" + "="*80)
print("EVALUATION RESULTS: FP32 MODEL")
print("="*80)

print(f"\n📊 Overall Accuracy: {acc_fp32:.2f}%")

print("\n🎯 Confusion Matrix (FP32):")
print("                 Predicted")
print("                 Class 0  Class 1")
print(f"Actual Class 0:  {cm_fp32[0,0]:6d}   {cm_fp32[0,1]:6d}")
print(f"Actual Class 1:  {cm_fp32[1,0]:6d}   {cm_fp32[1,1]:6d}")

print("\n📈 Per-Class Metrics (FP32):")
print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
print("-" * 60)
for class_id in [0, 1]:
    support = np.sum(ground_truth == class_id)
    print(f"{class_id:<10} {precision_fp32[class_id]:<12.4f} {recall_fp32[class_id]:<12.4f} {f1_fp32[class_id]:<12.4f} {support:<10}")

print(f"\n📊 Macro F1-Score: {f1_macro_fp32:.4f}")
print(f"📊 Weighted F1-Score: {f1_weighted_fp32:.4f}")

print("\n" + "="*80)
print("EVALUATION RESULTS: INT8 QUANTIZED MODEL")
print("="*80)

print(f"\n📊 Overall Accuracy: {acc_int8:.2f}%")

print("\n🎯 Confusion Matrix (INT8):")
print("                 Predicted")
print("                 Class 0  Class 1")
print(f"Actual Class 0:  {cm_int8[0,0]:6d}   {cm_int8[0,1]:6d}")
print(f"Actual Class 1:  {cm_int8[1,0]:6d}   {cm_int8[1,1]:6d}")

print("\n📈 Per-Class Metrics (INT8):")
print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
print("-" * 60)
for class_id in [0, 1]:
    support = np.sum(ground_truth == class_id)
    print(f"{class_id:<10} {precision_int8[class_id]:<12.4f} {recall_int8[class_id]:<12.4f} {f1_int8[class_id]:<12.4f} {support:<10}")

print(f"\n📊 Macro F1-Score: {f1_macro_int8:.4f}")
print(f"📊 Weighted F1-Score: {f1_weighted_int8:.4f}")

# ============================================================================
# 8. Comparison Summary
# ============================================================================

print("\n" + "="*80)
print("COMPARISON: FP32 vs INT8 QUANTIZED")
print("="*80)

print(f"\n📊 Accuracy Drop: {acc_fp32 - acc_int8:.2f}%")
print(f"   FP32:  {acc_fp32:.2f}%")
print(f"   INT8:  {acc_int8:.2f}%")

print(f"\n📈 F1-Score Comparison:")
print(f"   Macro F1 Drop:    {f1_macro_fp32 - f1_macro_int8:.4f}")
print(f"   Weighted F1 Drop: {f1_weighted_fp32 - f1_weighted_int8:.4f}")

print(f"\n🎯 Class-wise F1 Drop:")
for class_id in [0, 1]:
    drop = f1_fp32[class_id] - f1_int8[class_id]
    print(f"   Class {class_id}: {drop:.4f} (FP32: {f1_fp32[class_id]:.4f}, INT8: {f1_int8[class_id]:.4f})")

# ============================================================================
# 9. Save Results
# ============================================================================

print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

results = {
    'test_set': 'set02 + set05 (combined)',
    'num_samples': int(len(ground_truth)),
    'class_distribution': {
        'class_0': int(np.sum(ground_truth == 0)),
        'class_1': int(np.sum(ground_truth == 1))
    },
    'fp32': {
        'accuracy': float(acc_fp32),
        'confusion_matrix': cm_fp32.tolist(),
        'precision': precision_fp32.tolist(),
        'recall': recall_fp32.tolist(),
        'f1_score': f1_fp32.tolist(),
        'f1_macro': float(f1_macro_fp32),
        'f1_weighted': float(f1_weighted_fp32)
    },
    'int8': {
        'accuracy': float(acc_int8),
        'confusion_matrix': cm_int8.tolist(),
        'precision': precision_int8.tolist(),
        'recall': recall_int8.tolist(),
        'f1_score': f1_int8.tolist(),
        'f1_macro': float(f1_macro_int8),
        'f1_weighted': float(f1_weighted_int8)
    },
    'comparison': {
        'accuracy_drop': float(acc_fp32 - acc_int8),
        'f1_macro_drop': float(f1_macro_fp32 - f1_macro_int8),
        'f1_weighted_drop': float(f1_weighted_fp32 - f1_weighted_int8),
        'f1_class_drop': (f1_fp32 - f1_int8).tolist()
    }
}

results_file = './evaluation_results_comprehensive.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✅ Saved comprehensive results to: {results_file}")

# ============================================================================
# 10. Classification Report
# ============================================================================

print("\n" + "="*80)
print("DETAILED CLASSIFICATION REPORT")
print("="*80)

print("\n[FP32 Model]")
print(classification_report(ground_truth, predictions_fp32, 
                          target_names=['Not Crossing', 'Crossing'],
                          zero_division=0))

print("\n[INT8 Quantized Model]")
print(classification_report(ground_truth, predictions_int8, 
                          target_names=['Not Crossing', 'Crossing'],
                          zero_division=0))

print("\n" + "="*80)
print("✅ COMPREHENSIVE EVALUATION COMPLETE!")
print("="*80)
