"""
Fast way to get test data: Split existing set01 into 80% train, 20% test
Takes ~2 minutes
"""

import os
import pickle
import numpy as np

PIE_ROOT = r'C:\Users\jajit\OneDrive\Desktop\fpga_hackathon\PIE'
SEQUENCES_DIR = os.path.join(PIE_ROOT, 'data_cache', 'sequences')

print("=" * 80)
print("FAST SPLIT: Convert 111 samples into 80% train, 20% test")
print("=" * 80)

# Load current combined data
train_file = os.path.join(SEQUENCES_DIR, 'train_sequences.pkl')

with open(train_file, 'rb') as f:
    combined_data = pickle.load(f)

num_sequences = len(combined_data['image'])
print(f"\nTotal sequences: {num_sequences}")

# Create 80/20 split
split_idx = int(0.8 * num_sequences)
indices = np.random.permutation(num_sequences)

train_indices = indices[:split_idx]
test_indices = indices[split_idx:]

print(f"Train sequences: {len(train_indices)}")
print(f"Test sequences: {len(test_indices)}")

# Split all arrays
def split_data(data, train_idx, test_idx):
    """Split data dict by indices"""
    train_data = {
        'image': [data['image'][i] for i in train_idx],
        'bbox': [data['bbox'][i] for i in train_idx],
        'occlusion': [data['occlusion'][i] for i in train_idx],
        'intention_prob': [data['intention_prob'][i] for i in train_idx],
        'intention_binary': [data['intention_binary'][i] for i in train_idx],
        'ped_id': [data['ped_id'][i] for i in train_idx],
    }
    
    test_data = {
        'image': [data['image'][i] for i in test_idx],
        'bbox': [data['bbox'][i] for i in test_idx],
        'occlusion': [data['occlusion'][i] for i in test_idx],
        'intention_prob': [data['intention_prob'][i] for i in test_idx],
        'intention_binary': [data['intention_binary'][i] for i in test_idx],
        'ped_id': [data['ped_id'][i] for i in test_idx],
    }
    
    return train_data, test_data

train_data, test_data = split_data(combined_data, train_indices, test_indices)

# Save
print("\nSaving split files...")

train_file_new = os.path.join(SEQUENCES_DIR, 'train_sequences.pkl')
test_file_new = os.path.join(SEQUENCES_DIR, 'test_sequences.pkl')

with open(train_file_new, 'wb') as f:
    pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
print(f"[OK] Saved train: {len(train_data['image'])} sequences")

with open(test_file_new, 'wb') as f:
    pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)
print(f"[OK] Saved test: {len(test_data['image'])} sequences")

print("\n" + "=" * 80)
print("[DONE] Now you have test data! Run training again:")
print("  python train.py")
print("=" * 80)
