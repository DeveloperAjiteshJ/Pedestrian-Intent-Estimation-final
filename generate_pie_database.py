"""
PIE Dataset Generation Script - GPU Optimized
Extracts images from videos and generates the database for pedestrian intention estimation.
Optimized for NVIDIA RTX 3050 (6GB VRAM) with CUDA-accelerated video decoding.

VERIFIED REPOSITORY STRUCTURE:
PIE/
├── set01/, set02/, set05/          ← Videos are HERE (already have videos!)
├── annotations/set01/, set02/, etc ← XML annotations (ALREADY EXTRACTED!)
├── annotations_attributes/set01/   ← Pedestrian attributes (ALREADY EXTRACTED!)
├── annotations_vehicle/set01/      ← Vehicle/OBD data (ALREADY EXTRACTED!)
└── utilities/                      ← Dataset utilities
"""

import os
import sys
import pickle
import time
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

# Add utilities to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utilities.pie_data import PIE

# Check CUDA availability
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"OpenCV version: {cv2.__version__}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"GPU Memory: {gpu_props.total_memory / 1024**3:.2f} GB")
    device = torch.device('cuda:0')
    
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
else:
    print("WARNING: CUDA not available, using CPU")
    device = torch.device('cpu')

# Configuration
PIE_PATH = r'C:\Users\jajit\OneDrive\Desktop\fpga_hackathon\PIE'
EXTRACT_FRAME_TYPE = 'annotated'
TARGET_SET = 'set01'
USE_GPU_DECODE = True
NUM_WORKERS = min(4, cpu_count())


def check_gpu_capabilities():
    """Check GPU capabilities"""
    print("\n" + "="*70)
    print("GPU CAPABILITIES CHECK")
    print("="*70)
    
    gpu_features = {
        'CUDA': torch.cuda.is_available(),
        'cuDNN': torch.backends.cudnn.is_available() if torch.cuda.is_available() else False,
        'GPU Memory': f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB" if torch.cuda.is_available() else "N/A",
    }
    
    for feature, status in gpu_features.items():
        status_icon = "✅" if (isinstance(status, bool) and status) or (isinstance(status, str)) else "❌"
        print(f"{status_icon} {feature}: {status}")
    
    return gpu_features


def check_video_files():
    """Check if video files exist in PIE/set01/"""
    clips_path = os.path.join(PIE_PATH, TARGET_SET)
    print("\n" + "="*70)
    print("CHECKING VIDEO FILES")
    print("="*70)
    print(f"Looking for videos in: {clips_path}")
    
    if not os.path.exists(clips_path):
        print(f"❌ ERROR: Video folder not found: {clips_path}")
        return False, []
    
    video_files = [f for f in os.listdir(clips_path) if f.endswith('.mp4')]
    
    if not video_files:
        print(f"❌ ERROR: No .mp4 files found in {clips_path}")
        return False, []
    
    print(f"✅ Found {len(video_files)} video file(s) in {TARGET_SET}:")
    for vf in sorted(video_files):
        video_path = os.path.join(clips_path, vf)
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"   - {vf} ({size_mb:.2f} MB)")
    
    return True, video_files


def check_annotations():
    """Check if annotation files exist (should be pre-extracted)"""
    annot_path = os.path.join(PIE_PATH, 'annotations', TARGET_SET)
    print("\n" + "="*70)
    print("CHECKING ANNOTATION FILES")
    print("="*70)
    print(f"Looking for annotations in: {annot_path}")
    
    if not os.path.exists(annot_path):
        print(f"❌ ERROR: Annotation folder not found: {annot_path}")
        return False
    
    xml_files = [f for f in os.listdir(annot_path) if f.endswith('_annt.xml')]
    
    if not xml_files:
        print(f"❌ ERROR: No annotation files found in {annot_path}")
        return False
    
    print(f"✅ Found {len(xml_files)} annotation file(s):")
    for xf in sorted(xml_files):
        print(f"   - {xf}")
    
    # Check attribute annotations
    annot_attr_path = os.path.join(PIE_PATH, 'annotations_attributes', TARGET_SET)
    if os.path.exists(annot_attr_path):
        print(f"✅ Attribute annotations found at: {annot_attr_path}")
    else:
        print(f"⚠️  WARNING: Attribute annotations not found at {annot_attr_path}")
    
    # Check vehicle annotations
    annot_vehicle_path = os.path.join(PIE_PATH, 'annotations_vehicle', TARGET_SET)
    if os.path.exists(annot_vehicle_path):
        print(f"✅ Vehicle annotations found at: {annot_vehicle_path}")
    else:
        print(f"⚠️  WARNING: Vehicle annotations not found at {annot_vehicle_path}")
    
    return True


def extract_video_frames_gpu(video_path, output_dir, frame_list, use_gpu=True):
    """Extract specific frames from video with GPU acceleration"""
    os.makedirs(output_dir, exist_ok=True)
    
    vidcap = cv2.VideoCapture(video_path)
    
    if use_gpu:
        vidcap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
    
    if not vidcap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return 0
    
    frame_num = 0
    extracted_count = 0
    frame_set = set(frame_list)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    while vidcap.isOpened():
        ret, frame = vidcap.read()
        
        if not ret:
            break
        
        if frame_num in frame_set:
            output_path = os.path.join(output_dir, f"{frame_num:05d}.png")
            
            if not os.path.exists(output_path):
                cv2.imwrite(output_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            
            extracted_count += 1
            
            if extracted_count % 100 == 0:
                progress = (extracted_count / len(frame_list)) * 100
                print(f"    Progress: {progress:.1f}% ({extracted_count}/{len(frame_list)} frames)", end='\r')
        
        frame_num += 1
        
        if extracted_count >= len(frame_list):
            break
    
    vidcap.release()
    print(f"    ✅ Extracted {extracted_count} frames from {os.path.basename(video_path)}")
    
    return extracted_count


def extract_images_parallel():
    """Extract images from video clips"""
    print("\n" + "="*70)
    print("STEP 1: EXTRACTING IMAGES FROM VIDEOS")
    print("="*70)
    
    images_path = os.path.join(PIE_PATH, 'images', TARGET_SET)
    
    if os.path.exists(images_path):
        existing_videos = [d for d in os.listdir(images_path) if os.path.isdir(os.path.join(images_path, d))]
        if existing_videos:
            print(f"⚠️  Images already exist for {len(existing_videos)} video(s)")
            user_input = input("Do you want to re-extract? (y/n): ").lower()
            if user_input != 'y':
                print("Skipping image extraction...")
                return True
    
    print(f"Extracting {EXTRACT_FRAME_TYPE} frames from {TARGET_SET}...")
    print(f"Using {NUM_WORKERS} parallel workers")
    
    try:
        pie = PIE(data_path=PIE_PATH)
        
        if EXTRACT_FRAME_TYPE == 'annotated':
            extract_frames = pie.get_annotated_frame_numbers(TARGET_SET)
        else:
            extract_frames = pie.get_frame_numbers(TARGET_SET)
        
        print(f"\n📹 Videos to process: {len(extract_frames)}")
        total_frames = sum(frames[0] for frames in extract_frames.values())
        print(f"📊 Total frames to extract: {total_frames}")
        
        # Videos are in PIE/set01/
        clips_path = os.path.join(PIE_PATH, TARGET_SET)
        
        # Prepare tasks
        tasks = []
        for vid, frames in extract_frames.items():
            video_path = os.path.join(clips_path, f"{vid}.mp4")
            output_dir = os.path.join(images_path, vid)
            frame_list = frames[1:]  # Skip count, get frame numbers
            tasks.append((video_path, output_dir, frame_list))
        
        start_time = time.time()
        
        # Process videos in parallel
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {
                executor.submit(extract_video_frames_gpu, vpath, odir, flist, USE_GPU_DECODE): vid 
                for (vpath, odir, flist), vid in zip(tasks, extract_frames.keys())
            }
            
            for future in as_completed(futures):
                vid = futures[future]
                try:
                    frames_extracted = future.result()
                    print(f"  ✅ {vid}: {frames_extracted} frames")
                except Exception as e:
                    print(f"  ❌ {vid} failed: {str(e)}")
        
        elapsed = time.time() - start_time
        fps = total_frames / elapsed if elapsed > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"✅ Image extraction completed!")
        print(f"⏱️  Time: {elapsed/60:.2f} minutes")
        print(f"🚀 Speed: {fps:.1f} frames/second")
        
        # Check GPU usage
        if torch.cuda.is_available():
            print(f"🎮 GPU Memory Used: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR during image extraction: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def generate_database():
    """Generate annotation database"""
    print("\n" + "="*70)
    print("STEP 2: GENERATING ANNOTATION DATABASE")
    print("="*70)
    
    try:
        pie = PIE(data_path=PIE_PATH)
        
        cache_file = os.path.join(pie.cache_path, 'pie_database.pkl')
        if os.path.exists(cache_file):
            print(f"⚠️  Database cache already exists: {cache_file}")
            user_input = input("Do you want to regenerate? (y/n): ").lower()
            if user_input != 'y':
                print("Loading existing database...")
                with open(cache_file, 'rb') as f:
                    database = pickle.load(f)
                print(f"✅ Database loaded from cache")
                return database
        
        print("Generating database from annotations...")
        start_time = time.time()
        
        database = pie.generate_database()
        
        elapsed = time.time() - start_time
        print(f"\n✅ Database generated in {elapsed:.2f} seconds")
        
        # Print statistics
        print_database_stats(database)
        
        return database
        
    except Exception as e:
        print(f"\n❌ ERROR during database generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def print_database_stats(database):
    """Print statistics about generated database"""
    print("\n" + "="*70)
    print("DATABASE STATISTICS")
    print("="*70)
    
    total_pedestrians = 0
    total_frames = 0
    total_videos = 0
    intention_data = {'crossing': 0, 'not_crossing': 0, 'irrelevant': 0}
    
    for set_id, videos in database.items():
        print(f"\n📁 {set_id}:")
        total_videos += len(videos)
        
        for vid_id, vid_data in videos.items():
            num_peds = len(vid_data['ped_annotations'])
            total_pedestrians += num_peds
            total_frames += vid_data['num_frames']
            
            print(f"  📹 {vid_id}: {num_peds} pedestrians, {vid_data['num_frames']} frames")
            
            # Count intention labels
            for ped_id, ped_data in vid_data['ped_annotations'].items():
                crossing_label = ped_data['attributes']['crossing']
                intent_prob = ped_data['attributes']['intention_prob']
                
                if crossing_label == 1:
                    intention_data['crossing'] += 1
                elif crossing_label == 0:
                    intention_data['not_crossing'] += 1
                else:
                    intention_data['irrelevant'] += 1
                
                print(f"    Ped {ped_id}: intent_prob={intent_prob:.2f}, crossing={crossing_label}")
    
    print(f"\n" + "-"*70)
    print(f"Total videos: {total_videos}")
    print(f"Total pedestrians: {total_pedestrians}")
    print(f"Total frames: {total_frames}")
    print(f"\nIntention distribution:")
    print(f"  🚶 Crossing: {intention_data['crossing']}")
    print(f"  🛑 Not crossing: {intention_data['not_crossing']}")
    print(f"  ❓ Irrelevant: {intention_data['irrelevant']}")
    
    if total_pedestrians > 0:
        cross_pct = (intention_data['crossing'] / total_pedestrians) * 100
        print(f"\n  Crossing percentage: {cross_pct:.1f}%")


def generate_sequences():
    """Generate training sequences"""
    print("\n" + "="*70)
    print("STEP 3: GENERATING TRAINING SEQUENCES")
    print("="*70)
    
    try:
        pie = PIE(data_path=PIE_PATH)
        
        print("⚠️  Note: With only set01, using 80% train / 20% test split")
        
        seq_params = {
            'seq_type': 'intention',
            'fstride': 1,
            'min_track_size': 15,
            'height_rng': [0, float('inf')],
            'squarify_ratio': 0,
            'data_split_type': 'random',
            'random_params': {
                'ratios': [0.8, 0.2],
                'val_data': False,
                'regen_data': True
            }
        }
        
        print("\nGenerating training sequences...")
        start_time = time.time()
        train_data = pie.generate_data_trajectory_sequence(
            image_set='train',
            **seq_params
        )
        
        print("\nGenerating test sequences...")
        test_data = pie.generate_data_trajectory_sequence(
            image_set='test',
            **seq_params
        )
        
        elapsed = time.time() - start_time
        print(f"\n✅ Sequences generated in {elapsed:.2f} seconds")
        
        # Print sequence statistics
        print(f"\n📊 Sequence Statistics (Before Balancing):")
        train_pos = sum(1 for x in train_data['intention_binary'] if x[0][0] == 1)
        train_neg = len(train_data['intention_binary']) - train_pos
        test_pos = sum(1 for x in test_data['intention_binary'] if x[0][0] == 1)
        test_neg = len(test_data['intention_binary']) - test_pos
        
        print(f"  Train: {len(train_data['intention_binary'])} samples (Pos: {train_pos}, Neg: {train_neg})")
        print(f"  Test: {len(test_data['intention_binary'])} samples (Pos: {test_pos}, Neg: {test_neg})")
        
        # Balance samples
        print("\n⚖️  Balancing positive/negative samples...")
        train_data_balanced = pie.balance_samples_count(train_data, label_type='intention_binary')
        test_data_balanced = pie.balance_samples_count(test_data, label_type='intention_binary')
        
        # Save sequences
        output_dir = os.path.join(PIE_PATH, 'data_cache', 'sequences')
        os.makedirs(output_dir, exist_ok=True)
        
        train_file = os.path.join(output_dir, 'train_sequences.pkl')
        test_file = os.path.join(output_dir, 'test_sequences.pkl')
        
        with open(train_file, 'wb') as f:
            pickle.dump(train_data_balanced, f, pickle.HIGHEST_PROTOCOL)
        
        with open(test_file, 'wb') as f:
            pickle.dump(test_data_balanced, f, pickle.HIGHEST_PROTOCOL)
        
        print(f"\n✅ Sequences saved:")
        print(f"  📄 Train: {train_file}")
        print(f"  📄 Test: {test_file}")
        
        return train_data_balanced, test_data_balanced
        
    except Exception as e:
        print(f"\n❌ ERROR during sequence generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("🚀 PIE DATASET GENERATION PIPELINE - GPU ACCELERATED")
    print("="*70)
    print(f"Target: {TARGET_SET}")
    print(f"Frame extraction type: {EXTRACT_FRAME_TYPE}")
    print(f"Parallel workers: {NUM_WORKERS}")
    print("="*70)
    
    gpu_caps = check_gpu_capabilities()
    
    success, video_files = check_video_files()
    if not success:
        print("\n❌ FAILED: Video files not found")
        return False
    
    if not check_annotations():
        print("\n❌ FAILED: Annotation files not found")
        return False
    
    if not extract_images_parallel():
        print("\n❌ FAILED: Image extraction failed")
        return False
    
    database = generate_database()
    if database is None:
        print("\n❌ FAILED: Database generation failed")
        return False
    
    train_data, test_data = generate_sequences()
    if train_data is None:
        print("\n❌ FAILED: Sequence generation failed")
        return False
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\n📦 Generated files:")
    print(f"  1. Database: {os.path.join(PIE_PATH, 'data_cache', 'pie_database.pkl')}")
    print(f"  2. Images: {os.path.join(PIE_PATH, 'images', TARGET_SET)}")
    print(f"  3. Train sequences: {os.path.join(PIE_PATH, 'data_cache', 'sequences', 'train_sequences.pkl')}")
    print(f"  4. Test sequences: {os.path.join(PIE_PATH, 'data_cache', 'sequences', 'test_sequences.pkl')}")
    
    print("\n🎯 Next steps:")
    print("  1. Review the generated statistics above")
    print("  2. Create model architecture (model.py)")
    print("  3. Create training script (train.py)")
    print("  4. Start training on your RTX 3050!")
    
    # Final GPU stats
    if torch.cuda.is_available():
        print(f"\n🎮 Final GPU Stats:")
        print(f"  Memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"  Memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        print(f"  Max memory allocated: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
    
    return True


if __name__ == '__main__':
    try:
        success = main()
        
        print("\n" + "="*70)
        if success:
            print("🎉 All done! Ready for model training.")
        else:
            print("⚠️  Pipeline completed with warnings. Check output above.")
        print("="*70)
        
        input("\nPress Enter to exit...")
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        input("\nPress Enter to exit...")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n\n❌ UNEXPECTED ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)