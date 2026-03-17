"""
Export Quantized Weights for FPGA Deployment
Loads trained TinyMobileNet-XS model, quantizes it, and exports INT8 weights
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

from tinymobilenet_xs import (
    create_model,
    fold_batch_norm,
    quantize_model_post_training,
    export_weights_to_int8
)

from train import PIEIntentionDataset


def load_trained_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}...")
    
    # Create model architecture
    model = create_model(num_classes=2, t_frames=4)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Loaded model from epoch {checkpoint['epoch']}")
    print(f"✓ Best accuracy: {checkpoint['best_acc']:.2f}%")
    
    model.eval()
    model = model.to(device)
    
    return model


def quantize_and_export_weights(
    model,
    calibration_loader,
    device='cuda',
    output_dir='./fpga_weights'
):
    """
    Full quantization and export pipeline
    
    1. Fold BatchNorm into Conv weights
    2. Quantize using post-training quantization
    3. Export to INT8 C header for FPGA
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("QUANTIZATION & EXPORT PIPELINE")
    print("=" * 80)
    
    # Step 1: Fold BatchNorm
    print("\n[1/3] Folding BatchNorm into Conv weights...")
    print("      (Required for INT8 inference on FPGA)")
    model_fused = fold_batch_norm(model)
    model_fused.eval()
    
    # Step 2: Post-Training Quantization
    print("\n[2/3] Quantizing model (Post-Training Quantization)...")
    print("      Using calibration data to determine optimal scales...")
    model_int8, quant_config = quantize_model_post_training(
        model_fused,
        calibration_loader,
        device=device
    )
    
    # Step 3: Export INT8 weights to C header
    print("\n[3/3] Exporting INT8 weights to C header file...")
    output_file = os.path.join(output_dir, 'tinymobilenet_xs_weights.h')
    export_weights_to_int8(model_int8, quant_config, output_file=output_file)
    
    # Save quantization config as JSON
    import json
    config_file = os.path.join(output_dir, 'quantization_config.json')
    
    # Convert scales to serializable format
    weight_scales_serializable = {k: float(v) for k, v in quant_config.weight_scales.items()}
    activation_scales_serializable = {k: float(v) for k, v in quant_config.activation_scales.items()}
    
    with open(config_file, 'w') as f:
        json.dump({
            'weight_scales': weight_scales_serializable,
            'activation_scales': activation_scales_serializable,
            'quantization_bits': 8,
            'note': 'Scales are used to convert FP32 to INT8: int8_value = round(fp32_value * scale)'
        }, f, indent=2)
    
    print("\n" + "=" * 80)
    print("✓ EXPORT COMPLETE!")
    print("=" * 80)
    print(f"Output directory: {os.path.abspath(output_dir)}/")
    print(f"  📁 tinymobilenet_xs_weights.h  - INT8 weights for FPGA (C header)")
    print(f"  📁 quantization_config.json    - Quantization scales and metadata")
    print()
    print("Next steps:")
    print("  1. Copy tinymobilenet_xs_weights.h to your Vivado/Verilog project")
    print("  2. Use the quantization scales in your RTL inference engine")
    print("  3. Implement INT8 MAC operations in hardware")
    print("=" * 80)
    
    return model_int8, quant_config


def validate_quantized_model(model_fp32, model_int8, test_loader, device='cuda'):
    """
    Compare FP32 vs INT8 model accuracy
    """
    print("\n" + "=" * 80)
    print("VALIDATING QUANTIZED MODEL")
    print("=" * 80)
    
    model_fp32.eval()
    model_int8.eval()
    
    correct_fp32 = 0
    correct_int8 = 0
    total = 0
    
    max_diff = 0
    avg_diff = 0
    
    with torch.no_grad():
        for frames, labels in tqdm(test_loader, desc="Validating"):
            frames = frames.to(device)
            labels = labels.to(device)
            
            # FP32 inference
            logits_fp32 = model_fp32(frames)
            pred_fp32 = logits_fp32.argmax(dim=1)
            correct_fp32 += (pred_fp32 == labels).sum().item()
            
            # INT8 inference (still in FP32 framework, but quantized weights)
            logits_int8 = model_int8(frames)
            pred_int8 = logits_int8.argmax(dim=1)
            correct_int8 += (pred_int8 == labels).sum().item()
            
            # Compute output difference
            diff = torch.abs(logits_fp32 - logits_int8).max().item()
            max_diff = max(max_diff, diff)
            avg_diff += torch.abs(logits_fp32 - logits_int8).mean().item()
            
            total += labels.size(0)
    
    acc_fp32 = 100 * correct_fp32 / total
    acc_int8 = 100 * correct_int8 / total
    avg_diff /= len(test_loader)
    
    print(f"\n📊 Accuracy Comparison:")
    print(f"  FP32 model:  {acc_fp32:.2f}% ({correct_fp32}/{total})")
    print(f"  INT8 model:  {acc_int8:.2f}% ({correct_int8}/{total})")
    print(f"  Accuracy drop: {acc_fp32 - acc_int8:.2f}%")
    print(f"\n📊 Output Difference:")
    print(f"  Max logit difference: {max_diff:.4f}")
    print(f"  Avg logit difference: {avg_diff:.4f}")
    
    if abs(acc_fp32 - acc_int8) < 5.0:
        print("\n✅ Quantization successful! Accuracy drop < 5%")
    else:
        print("\n⚠️  Warning: Accuracy drop > 5%. Consider fine-tuning or QAT.")
    
    print("=" * 80)


def main():
    """Main export entry point"""
    
    # Configuration
    PIE_ROOT = r'C:\Users\jajit\OneDrive\Desktop\fpga_hackathon\PIE'
    CHECKPOINT_PATH = './checkpoints/best_model.pth'
    CALIBRATION_PKL = os.path.join(PIE_ROOT, 'data_cache/sequences/test_sequences.pkl')
    OUTPUT_DIR = './fpga_weights'
    
    BATCH_SIZE = 16
    MAX_FRAMES = 4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 80)
    print("TinyMobileNet-XS Weight Export for FPGA")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)
    
    # Check if checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\n❌ ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        print("Please train the model first using train.py")
        return
    
    # Load trained model
    model = load_trained_model(CHECKPOINT_PATH, device=DEVICE)
    
    # Create calibration dataset (using test set for calibration)
    print("\nLoading calibration dataset...")
    calibration_dataset = PIEIntentionDataset(
        CALIBRATION_PKL,
        PIE_ROOT,
        max_frames=MAX_FRAMES,
        split='test'
    )
    
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"✓ Loaded {len(calibration_dataset)} calibration samples")
    
    # Quantize and export
    model_int8, quant_config = quantize_and_export_weights(
        model,
        calibration_loader,
        device=DEVICE,
        output_dir=OUTPUT_DIR
    )
    
    # Validate quantized model
    print("\n" + "=" * 80)
    print("Optional: Validate quantized model accuracy")
    print("=" * 80)
    validate_choice = input("Run validation? (y/n): ").strip().lower()
    
    if validate_choice == 'y':
        validate_quantized_model(
            model,
            model_int8,
            calibration_loader,
            device=DEVICE
        )
    
    print("\n" + "=" * 80)
    print("✓ Weight export complete!")
    print("=" * 80)
    print(f"\nYour FPGA weights are ready in: {os.path.abspath(OUTPUT_DIR)}/")
    print("\n🚀 Next steps:")
    print("   1. Open tinymobilenet_xs_weights.h")
    print("   2. Copy to your Vivado/Verilog project")
    print("   3. Implement INT8 inference engine in RTL")
    print("   4. Use quantization scales from quantization_config.json")
    print("=" * 80)


if __name__ == '__main__':
    main()
