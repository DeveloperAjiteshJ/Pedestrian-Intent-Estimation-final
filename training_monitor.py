"""
TRAINING MONITOR
Checks training progress by analyzing checkpoint files and directories
Run this to see current training status WITHOUT needing terminal access
"""

import os
import torch
from pathlib import Path
import time
import json

PIE_ROOT = r'C:\Users\jajit\OneDrive\Desktop\fpga_hackathon\PIE'

print("=" * 80)
print("🚀 TRAINING PROGRESS MONITOR")
print("=" * 80)
print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Check if training has started
checkpoint_dir = os.path.join(PIE_ROOT, 'checkpoints')
fpga_weights_dir = os.path.join(PIE_ROOT, 'fpga_weights')

print(f"\n[1] CHECKING CHECKPOINT DIRECTORY")
print(f"    Path: {checkpoint_dir}")

if os.path.exists(checkpoint_dir):
    print(f"    ✅ EXISTS")
    
    # List files
    files = os.listdir(checkpoint_dir)
    print(f"    Files created: {files}")
    
    if 'best_model.pth' in files:
        # Get file info
        model_path = os.path.join(checkpoint_dir, 'best_model.pth')
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # Convert to MB
        mod_time = os.path.getmtime(model_path)
        
        print(f"\n    📦 best_model.pth")
        print(f"       Size: {file_size:.1f} MB")
        print(f"       Last modified: {time.strftime('%H:%M:%S', time.localtime(mod_time))}")
        
        # Try to load and check accuracy
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'best_acc' in checkpoint:
                best_acc = checkpoint['best_acc']
                epoch = checkpoint.get('epoch', '?')
                print(f"       Best accuracy so far: {best_acc:.2f}% (Epoch {epoch})")
                
                # Assess progress
                print(f"\n    📊 PROGRESS ASSESSMENT:")
                if best_acc >= 90:
                    print(f"       ✅ EXCELLENT! Target (90%+) REACHED!")
                elif best_acc >= 85:
                    print(f"       ✅ VERY GOOD! Nearly at target...")
                elif best_acc >= 80:
                    print(f"       ⚠️  GOOD! On track for 90%...")
                elif best_acc >= 70:
                    print(f"       ⚠️  ACCEPTABLE! Still improving...")
                elif best_acc >= 50:
                    print(f"       ⏳ EARLY STAGE! Still learning...")
                else:
                    print(f"       ❌ ISSUE! Accuracy too low")
        except Exception as e:
            print(f"       ⚠️  Could not read checkpoint: {e}")
    else:
        print(f"    ⏳ best_model.pth NOT YET CREATED")
        print(f"       Training might still be initializing...")
else:
    print(f"    ❌ NOT CREATED YET")
    print(f"    Training might not have started or is still loading data...")

# Check FPGA weights
print(f"\n[2] CHECKING FPGA WEIGHTS DIRECTORY")
print(f"    Path: {fpga_weights_dir}")

if os.path.exists(fpga_weights_dir):
    print(f"    ✅ EXISTS")
    files = os.listdir(fpga_weights_dir)
    print(f"    Files created: {files}")
    
    if 'tinymobilenet_xs_weights.h' in files:
        weights_path = os.path.join(fpga_weights_dir, 'tinymobilenet_xs_weights.h')
        file_size = os.path.getsize(weights_path) / 1024  # KB
        print(f"\n    ✅ tinymobilenet_xs_weights.h")
        print(f"       Size: {file_size:.1f} KB")
        print(f"       STATUS: READY FOR FPGA! ✓")
    
    if 'quantization_config.json' in files:
        config_path = os.path.join(fpga_weights_dir, 'quantization_config.json')
        print(f"\n    ✅ quantization_config.json")
        print(f"       STATUS: SCALE FACTORS EXPORTED ✓")
else:
    print(f"    ❌ NOT YET CREATED")
    print(f"    Will be created when training finishes...")

# Estimate progress
print(f"\n[3] ESTIMATED PROGRESS")
print(f"    Total epochs configured: 50")

if os.path.exists(checkpoint_dir) and 'best_model.pth' in os.listdir(checkpoint_dir):
    try:
        checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pth'), map_location='cpu')
        epoch_done = checkpoint.get('epoch', 0)
        epochs_remaining = 50 - epoch_done - 1
        
        print(f"    Epochs completed: {epoch_done + 1}/50")
        print(f"    Epochs remaining: {epochs_remaining}")
        print(f"    Progress: {100 * (epoch_done + 1) / 50:.1f}%")
        
        # Time estimate (each epoch ~2-3 min)
        time_per_epoch = 2.5  # minutes
        time_remaining_min = epochs_remaining * time_per_epoch
        hours = int(time_remaining_min // 60)
        mins = int(time_remaining_min % 60)
        
        print(f"    ⏱️  Estimated time remaining: {hours}h {mins}m")
        
    except Exception as e:
        print(f"    Could not determine progress: {e}")
else:
    print(f"    Still initializing or in early stages...")

# Final status
print(f"\n[4] OVERALL STATUS")
print("=" * 80)

has_checkpoint = os.path.exists(checkpoint_dir) and 'best_model.pth' in os.listdir(checkpoint_dir)
has_weights = os.path.exists(fpga_weights_dir) and 'tinymobilenet_xs_weights.h' in os.listdir(fpga_weights_dir)

if has_weights:
    print("✅ TRAINING COMPLETE!")
    print("   • Best model saved")
    print("   • INT8 weights exported")
    print("   • Ready for FPGA deployment!")
elif has_checkpoint:
    try:
        checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pth'), map_location='cpu')
        acc = checkpoint.get('best_acc', 0)
        epoch = checkpoint.get('epoch', 0)
        
        if epoch >= 49:
            print("✅ TRAINING ALMOST COMPLETE!")
            print(f"   • Best accuracy achieved: {acc:.2f}%")
            print(f"   • Finalizing quantization...")
        elif acc >= 90:
            print("🔄 TRAINING IN PROGRESS")
            print(f"   • Current best accuracy: {acc:.2f}% ✓ (Target achieved!)")
            print(f"   • Continuing to refine...")
        elif acc >= 85:
            print("🔄 TRAINING IN PROGRESS")
            print(f"   • Current best accuracy: {acc:.2f}%")
            print(f"   • On track for 90%+ target...")
        else:
            print("🔄 TRAINING IN PROGRESS")
            print(f"   • Current best accuracy: {acc:.2f}%")
            print(f"   • Still improving...")
    except:
        print("🔄 TRAINING IN PROGRESS")
        print("   Checkpoint exists but couldn't read details")
else:
    print("⏳ TRAINING STARTING...")
    print("   • Loading data")
    print("   • Initializing model")
    print("   • Check back in a few minutes")

print("\n" + "=" * 80)
print("💡 NEXT STEPS:")
print("=" * 80)

if has_weights:
    print("1. ✅ Copy tinymobilenet_xs_weights.h to Vivado project")
    print("2. ✅ Integrate with RTL conv_kernel_systolic.v")
    print("3. ✅ Test on Nexys DDR4 hardware")
elif has_checkpoint:
    print("1. ⏳ Wait for training to complete")
    print("2. ⏳ INT8 weights will be exported automatically")
    print("3. ✅ Then copy to Vivado project")
else:
    print("1. ⏳ Training is starting/initializing")
    print("2. ⏳ Wait for first checkpoint")
    print("3. ✅ Check this monitor in 30 minutes")

print("\n" + "=" * 80)
print("📝 HOW TO USE THIS MONITOR:")
print("=" * 80)
print("• Run this script anytime to check progress")
print("• NO terminal needed - just run the file")
print("• Shows real-time accuracy as training progresses")
print("• Estimates time remaining")
print("\n✅ Safe to run while training is happening!")
print("=" * 80)

# Keep window open for 30 seconds, or until user presses ENTER
import time
print("\n\n⏳ Window will close in 30 seconds... (or press ENTER to close now)")
try:
    input()  # Will close immediately if user presses ENTER
except:
    time.sleep(30)  # Otherwise wait 30 seconds
