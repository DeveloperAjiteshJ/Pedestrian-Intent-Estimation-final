"""
Save test data indices mapping for FPGA deployment
So FPGA uses the EXACT same test set
"""

import os
import pickle
import numpy as np
import json

PIE_ROOT = r'C:\Users\jajit\OneDrive\Desktop\fpga_hackathon\PIE'
SEQUENCES_DIR = os.path.join(PIE_ROOT, 'data_cache', 'sequences')

print("=" * 80)
print("SAVE TEST SET MAPPING FOR FPGA")
print("=" * 80)

# Load test data
test_file = os.path.join(SEQUENCES_DIR, 'test_sequences.pkl')

with open(test_file, 'rb') as f:
    test_data = pickle.load(f)

num_test = len(test_data['image'])
print(f"\nTest sequences: {num_test}")

# Extract test set info for FPGA
test_info = {
    'num_test_samples': num_test,
    'test_samples': []
}

for idx, image_paths in enumerate(test_data['image']):
    # Get info about this sequence
    first_frame_path = image_paths[0]
    
    # Parse path: C:\...\images\set01\video_0001\01016.png
    parts = first_frame_path.replace('\\', '/').split('/')
    
    set_id = parts[-3]
    video_id = parts[-2]
    first_frame = parts[-1].replace('.png', '')
    
    test_info['test_samples'].append({
        'index': idx,
        'set_id': set_id,
        'video_id': video_id,
        'num_frames': len(image_paths),
        'first_frame_id': first_frame,
        'label': int(test_data['intention_binary'][idx][0][0]),
    })
    
    print(f"  [{idx}] {set_id}/{video_id} - {len(image_paths)} frames - Label: {test_info['test_samples'][-1]['label']}")

# Save mapping
info_file = os.path.join(SEQUENCES_DIR, 'test_set_mapping.json')

with open(info_file, 'w') as f:
    json.dump(test_info, f, indent=2)

print(f"\n[OK] Saved test mapping: {info_file}")
print("\nThis file contains the EXACT test set for FPGA deployment.")
print("Use this to ensure FPGA tests on the same 23 sequences.")
print("=" * 80)
