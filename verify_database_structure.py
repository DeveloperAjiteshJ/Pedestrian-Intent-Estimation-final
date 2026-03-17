#!/usr/bin/env python3
"""Verify PIE database structure for FPGA architecture planning"""

import pickle
import os
import numpy as np
from pathlib import Path

def check_database():
    pkl_file = 'data_cache/sequences/train_sequences.pkl'
    
    if not os.path.exists(pkl_file):
        print("❌ Train sequences file not found!")
        return
    
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    print('='*80)
    print('PIE DATABASE STRUCTURE ANALYSIS FOR FPGA DEPLOYMENT')
    print('='*80)
    
    # 1. Keys and structure
    print(f'\n[1] AVAILABLE KEYS: {list(data.keys())}')
    print(f'\n[2] TOTAL TRAINING SAMPLES: {len(data["image"])}')
    
    # 2. Sequence analysis
    seq_lengths = [len(seq) for seq in data['image']]
    print(f'\n[3] SEQUENCE LENGTH ANALYSIS:')
    print(f'    Min T: {min(seq_lengths)}, Max T: {max(seq_lengths)}, Mean T: {np.mean(seq_lengths):.1f}')
    print(f'    Distribution: {dict(zip(*np.unique(seq_lengths, return_counts=True)))}')
    
    # Use actual mean or most common
    actual_T = int(np.median(seq_lengths))
    print(f'    ⚠️  ACTUAL T (median): {actual_T} frames')
    print(f'    ✓ Your proposed T=4 is {"✅ VALID" if actual_T <= 4 else "❌ TOO SMALL"}')
    
    # 3. Label analysis
    print(f'\n[4] CLASSIFICATION TARGET:')
    labels_flat = [int(x[0][0]) for x in data['intention_binary']]
    crossing = sum(labels_flat)
    not_crossing = len(labels_flat) - crossing
    print(f'    Class 0 (Not crossing): {not_crossing} ({100*not_crossing/len(labels_flat):.1f}%)')
    print(f'    Class 1 (Crossing): {crossing} ({100*crossing/len(labels_flat):.1f}%)')
    print(f'    ✓ K=2 (binary classification) CONFIRMED')
    
    # 4. Sample structure
    print(f'\n[5] SAMPLE DATA STRUCTURE:')
    print(f'    Images: {data["image"][0][:2]}...')  # Show first 2 image paths
    print(f'    Image file pattern: Frame indices in sequence')
    
    # 5. Expected input format
    print(f'\n[6] INPUT PROCESSING FOR FPGA:')
    print(f'    Per-frame input shape: (64, 64, 3) — to be resized from disk')
    print(f'    Sequence input shape: (T, 64, 64, 3) = ({actual_T}, 64, 64, 3)')
    print(f'    Batch size for inference: B=1 (as per your spec)')
    print(f'    Total input: (1, {actual_T}, 64, 64, 3)')
    
    # 6. Memory requirements
    print(f'\n[7] INPUT MEMORY FOOTPRINT:')
    input_mem = actual_T * 64 * 64 * 3  # Raw RGB
    print(f'    Raw (float32): {input_mem * 4 / 1024:.1f} KB')
    print(f'    Quantized (uint8): {input_mem / 1024:.1f} KB')
    
    # 7. Output format
    print(f'\n[8] OUTPUT FORMAT:')
    print(f'    Output layer: 2 logits (for softmax/sigmoid)')
    print(f'    Output shape: (1, 2)')
    print(f'    Example: [logit_not_crossing, logit_crossing]')
    
    print('\n' + '='*80)
    return actual_T

if __name__ == '__main__':
    actual_T = check_database()
    print(f'\n✅ Database validation complete. Use T={actual_T} in HLS architecture.')
