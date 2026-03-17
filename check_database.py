#!/usr/bin/env python3
import pickle
import os
from pathlib import Path

pkl_file = 'data_cache/sequences/train_sequences.pkl'

if os.path.exists(pkl_file):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    print('='*70)
    print('TRAIN SEQUENCES DATA STRUCTURE')
    print('='*70)
    print(f'\nAvailable keys: {list(data.keys())}')
    print(f'\nTotal samples: {len(data["image"])}')
    
    # Check first sample
    print(f'\n[FIRST SAMPLE DETAILS]')
    print(f'  Image sequence length (T): {len(data["image"][0])}')
    print(f'  Sample image path: {data["image"][0][0]}')
    print(f'  Number of bboxes in sequence: {len(data["bbox"][0])}')
    print(f'  First bbox: {data["bbox"][0][0]}')
    print(f'  Intention label: {data["intention_binary"][0]}')
    
    # Statistics
    print(f'\n[DATASET STATISTICS]')
    intention_flat = [x[0][0] for x in data['intention_binary']]
    crossing = sum(intention_flat)
    not_crossing = len(intention_flat) - crossing
    print(f'  Crossing samples: {crossing}')
    print(f'  Not crossing samples: {not_crossing}')
    print(f'  Class balance: {crossing}/{len(intention_flat)} = {100*crossing/len(intention_flat):.1f}%')
    
    # Output dimensions
    print(f'\n[OUTPUT LAYER REQUIREMENTS]')
    print(f'  Output classes K: 2 (binary: crossing/not-crossing)')
    print(f'  Per-frame input shape: (1, 64, 64, 3) after resize')
    print(f'  Sequence length T: {len(data["image"][0])} frames')
    
else:
    print('ERROR: Train sequences file not found!')
    print('Available files:')
    cache_path = Path('data_cache/sequences')
    if cache_path.exists():
        for f in cache_path.glob('*'):
            print(f'  - {f.name}')
    else:
        print('  data_cache/sequences folder does not exist!')
