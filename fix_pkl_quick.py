"""
QUICK FIX: Generate PKL files using ONLY existing images on disk
Maps annotation frame IDs to actual image files

This script:
1. Loads PIE annotations to get frame mappings
2. Maps annotation frame IDs to actual filenames on disk  
3. Generates train/test pkl files with correct paths
"""

import os
import pickle
import sys
from pathlib import Path

# Add utilities to path
sys.path.insert(0, 'utilities')
from pie_data import PIE

PIE_ROOT = r'C:\Users\jajit\OneDrive\Desktop\fpga_hackathon\PIE'
IMAGES_DIR = os.path.join(PIE_ROOT, 'images')
OUTPUT_DIR = os.path.join(PIE_ROOT, 'data_cache', 'sequences')

print("=" * 80)
print("FIX: Generate PKL from existing images with proper frame ID mapping")
print("=" * 80)

# Step 1: Load PIE to get frame information
print("\n[1/3] Loading PIE annotations and frame data...")
pie = PIE(data_path=PIE_ROOT)
annotations = pie.generate_database()

# Build a map of which images exist: {set_id: {video_id: set(frame_ids)}}
print("\n[2/3] Scanning actual images on disk...")
actual_frames_on_disk = {}  # {set_id: {video_id: set of frame IDs that exist}}

for set_id in os.listdir(IMAGES_DIR):
    set_path = os.path.join(IMAGES_DIR, set_id)
    if not os.path.isdir(set_path):
        continue
    
    actual_frames_on_disk[set_id] = {}
    
    for video_id in os.listdir(set_path):
        video_path = os.path.join(set_path, video_id)
        if not os.path.isdir(video_path):
            continue
        
        # Extract frame IDs from filenames (e.g., "01016.png" -> 1016)
        frame_ids = set()
        for fname in os.listdir(video_path):
            if fname.endswith('.png'):
                try:
                    frame_id = int(fname.replace('.png', ''))
                    frame_ids.add(frame_id)
                except ValueError:
                    pass
        
        actual_frames_on_disk[set_id][video_id] = frame_ids
        if frame_ids:
            print(f"  {set_id}/{video_id}: {len(frame_ids)} frames available")

# Step 3: Generate sequences using actual images
print("\n[3/3] Generating sequences from actual images...")

def generate_sequences_from_actual_images(image_set, set_ids):
    """Generate sequences using only images that exist on disk"""
    
    intention_prob, intention_binary = [], []
    image_seq, pids_seq = [], []
    box_seq, occ_seq = [], []
    
    samples_created = 0
    samples_skipped = 0
    
    for sid in set_ids:
        if sid not in actual_frames_on_disk:
            print(f"  [!] Skipping {sid} - no images found")
            continue
        
        if sid not in annotations:
            print(f"  [!] Skipping {sid} - no annotations found")
            continue
        
        for vid in sorted(annotations[sid]):
            if vid not in actual_frames_on_disk[sid]:
                print(f"  [!] Skipping {sid}/{vid} - no images found")
                continue
            
            available_frames = actual_frames_on_disk[sid][vid]
            pid_annots = annotations[sid][vid]['ped_annotations']
            
            for pid in sorted(pid_annots):
                # Get annotation frame IDs
                exp_start_frame = pid_annots[pid]['attributes']['exp_start_point']
                critical_frame = pid_annots[pid]['attributes']['critical_point']
                frames_annot = pid_annots[pid]['frames']
                
                try:
                    start_idx = frames_annot.index(exp_start_frame)
                    end_idx = frames_annot.index(critical_frame)
                except ValueError:
                    samples_skipped += 1
                    continue
                
                # Get the frame IDs from annotations for this range
                frame_ids = frames_annot[start_idx:end_idx + 1]
                boxes = pid_annots[pid]['bbox'][start_idx:end_idx + 1]
                occlusions = pid_annots[pid]['occlusion'][start_idx:end_idx + 1]
                
                # Check if ALL required frames exist on disk
                frames_exist = all(fid in available_frames for fid in frame_ids)
                
                if not frames_exist:
                    samples_skipped += 1
                    continue
                
                # Build image paths using actual frame IDs
                images = []
                valid_boxes = []
                valid_occs = []
                
                for i, frame_id in enumerate(frame_ids):
                    # Use the proper format: zero-padded 5-digit frame ID
                    img_filename = '{:05d}.png'.format(frame_id)
                    img_path = os.path.join(IMAGES_DIR, sid, vid, img_filename)
                    
                    # Verify file exists
                    if os.path.isfile(img_path):
                        images.append(img_path)
                        valid_boxes.append(boxes[i])
                        valid_occs.append(occlusions[i])
                    else:
                        # Frame doesn't exist, skip this sequence
                        break
                
                # Check if we got all frames and minimum sequence length
                if len(images) < len(frame_ids) or len(images) < 15:  # min_track_size = 15
                    samples_skipped += 1
                    continue
                
                # Create sequence
                int_prob = [[pid_annots[pid]['attributes']['intention_prob']]] * len(images)
                int_bin = [[int(pid_annots[pid]['attributes']['intention_prob'] > 0.5)]] * len(images)
                
                image_seq.append(images)
                box_seq.append(valid_boxes)
                occ_seq.append(valid_occs)
                intention_prob.append(int_prob)
                intention_binary.append(int_bin)
                
                ped_ids = [[pid]] * len(images)
                pids_seq.append(ped_ids)
                
                samples_created += 1
    
    print(f"  [OK] Created: {samples_created} sequences")
    print(f"  [!] Skipped: {samples_skipped} sequences (missing frames or too short)")
    
    return {
        'image': image_seq,
        'bbox': box_seq,
        'occlusion': occ_seq,
        'intention_prob': intention_prob,
        'intention_binary': intention_binary,
        'ped_id': pids_seq
    }

# Generate train and test sets
train_sets = ['set01', 'set02', 'set04']  # PIE default train split
test_sets = ['set03']  # PIE default test split

print("\nGenerating TRAIN sequences...")
train_data = generate_sequences_from_actual_images('train', train_sets)

print("\nGenerating TEST sequences...")
test_data = generate_sequences_from_actual_images('test', test_sets)

# Step 4: Save pkl files
print("\n[4/4] Saving pkl files...")

os.makedirs(OUTPUT_DIR, exist_ok=True)

train_file = os.path.join(OUTPUT_DIR, 'train_sequences.pkl')
test_file = os.path.join(OUTPUT_DIR, 'test_sequences.pkl')

with open(train_file, 'wb') as f:
    pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
print(f"[OK] Saved: {train_file}")
print(f"   Samples: {len(train_data['image'])}")

with open(test_file, 'wb') as f:
    pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)
print(f"[OK] Saved: {test_file}")
print(f"   Samples: {len(test_data['image'])}")

# Final summary
print("\n" + "=" * 80)
print("[SUCCESS] PKL GENERATION COMPLETE!")
print("=" * 80)
print(f"Train samples: {len(train_data['image'])}")
print(f"Test samples: {len(test_data['image'])}")
print("\nNow run:")
print("  python test_image_loading.py  # Verify it works")
print("  python train.py               # Start training!")
print("=" * 80)
