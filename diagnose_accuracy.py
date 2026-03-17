"""
Diagnose class imbalance and accuracy calculation issues
"""

import pickle

PIE_ROOT = r'C:\Users\jajit\OneDrive\Desktop\fpga_hackathon\PIE'
train_pkl = f'{PIE_ROOT}/data_cache/sequences/train_sequences.pkl'
test_pkl = f'{PIE_ROOT}/data_cache/sequences/test_sequences.pkl'

print("=" * 80)
print("DATA DIAGNOSTICS")
print("=" * 80)

with open(train_pkl, 'rb') as f:
    train_data = pickle.load(f)
    
with open(test_pkl, 'rb') as f:
    test_data = pickle.load(f)

# Extract labels
train_labels = [int(x[0][0]) for x in train_data['intention_binary']]
test_labels = [int(x[0][0]) for x in test_data['intention_binary']]

print("\n[TRAINING SET - set01]")
print(f"  Total samples: {len(train_labels)}")
print(f"  Class 0 (not-crossing): {len([x for x in train_labels if x == 0])} ({100*len([x for x in train_labels if x == 0])/len(train_labels):.1f}%)")
print(f"  Class 1 (crossing): {len([x for x in train_labels if x == 1])} ({100*len([x for x in train_labels if x == 1])/len(train_labels):.1f}%)")
print(f"  Majority class accuracy (if model predicts majority): {100*max(len([x for x in train_labels if x == 0]), len([x for x in train_labels if x == 1]))/len(train_labels):.2f}%")

print("\n[TEST SET - set05]")
print(f"  Total samples: {len(test_labels)}")
print(f"  Class 0 (not-crossing): {len([x for x in test_labels if x == 0])} ({100*len([x for x in test_labels if x == 0])/len(test_labels):.1f}%)")
print(f"  Class 1 (crossing): {len([x for x in test_labels if x == 1])} ({100*len([x for x in test_labels if x == 1])/len(test_labels):.1f}%)")
print(f"  Majority class accuracy (if model predicts majority): {100*max(len([x for x in test_labels if x == 0]), len([x for x in test_labels if x == 1]))/len(test_labels):.2f}%")

print("\n[DIAGNOSIS]")
if abs(100*len([x for x in train_labels if x == 0])/len(train_labels) - 70.27) < 1:
    print("  [!] ISSUE FOUND: Model is predicting class 0 for ALL training samples!")
    print("  [!] Train accuracy is just majority class baseline")
elif abs(100*len([x for x in train_labels if x == 1])/len(train_labels) - 70.27) < 1:
    print("  [!] ISSUE FOUND: Model is predicting class 1 for ALL training samples!")
    print("  [!] Train accuracy is just majority class baseline")
    
if abs(100*len([x for x in test_labels if x == 1])/len(test_labels) - 81.25) < 1 or abs(100*len([x for x in test_labels if x == 0])/len(test_labels) - 81.25) < 1:
    print("  [!] ISSUE FOUND: Model is predicting one class for ALL test samples!")
    print("  [!] Test accuracy is just majority class baseline on test set")

print("\n[RECOMMENDATIONS]")
print("  1. Check if model weights are frozen")
print("  2. Add class weights to loss function to handle imbalance")
print("  3. Lower learning rate - current LR might be too high")
print("  4. Add debugging prints to see actual predictions")

print("=" * 80)
