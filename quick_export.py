"""
Quick Export Script - No User Interaction
Exports quantized weights directly from trained model
"""

import torch
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from export_quantized_weights import (
    load_trained_model,
    quantize_and_export_weights,
    PIEIntentionDataset
)
from torch.utils.data import DataLoader


def quick_export():
    """Quick export without validation"""
    
    # Configuration
    PIE_ROOT = r'C:\Users\jajit\OneDrive\Desktop\fpga_hackathon\PIE'
    CHECKPOINT_PATH = './checkpoints/best_model.pth'
    CALIBRATION_PKL = os.path.join(PIE_ROOT, 'data_cache/sequences/test_sequences.pkl')
    OUTPUT_DIR = './fpga_weights'
    
    BATCH_SIZE = 16
    MAX_FRAMES = 4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n🚀 QUICK EXPORT MODE - Exporting FPGA weights...")
    print("=" * 80)
    
    # Check checkpoint
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ ERROR: No trained model found at {CHECKPOINT_PATH}")
        print("Run train.py first to train the model!")
        return False
    
    # Load model
    model = load_trained_model(CHECKPOINT_PATH, device=DEVICE)
    
    # Load calibration data
    print("\nLoading calibration data...")
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
    
    # Quantize and export
    model_int8, quant_config = quantize_and_export_weights(
        model,
        calibration_loader,
        device=DEVICE,
        output_dir=OUTPUT_DIR
    )
    
    print("\n✅ SUCCESS! Weights exported to ./fpga_weights/")
    return True


if __name__ == '__main__':
    success = quick_export()
    sys.exit(0 if success else 1)
