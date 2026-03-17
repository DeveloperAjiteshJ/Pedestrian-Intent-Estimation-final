"""
DIRECT DIAGNOSIS: Scan actual images directory structure
Find out exactly where images are and why they're not being found
"""

import os
from pathlib import Path

PIE_ROOT = r'C:\Users\jajit\OneDrive\Desktop\fpga_hackathon\PIE'
IMAGES_DIR = os.path.join(PIE_ROOT, 'images')

print("=" * 80)
print("🔍 SCANNING ACTUAL DIRECTORY STRUCTURE")
print("=" * 80)

print(f"\nImages directory: {IMAGES_DIR}")
print(f"Exists: {os.path.exists(IMAGES_DIR)}")

if os.path.exists(IMAGES_DIR):
    # List all sets
    sets = [d for d in os.listdir(IMAGES_DIR) if os.path.isdir(os.path.join(IMAGES_DIR, d))]
    print(f"\nSets found: {sets}")
    
    # For each set, list videos
    for set_name in sets[:2]:  # Just first 2 sets
        set_path = os.path.join(IMAGES_DIR, set_name)
        videos = [d for d in os.listdir(set_path) if os.path.isdir(os.path.join(set_path, d))]
        print(f"\n{set_name} videos ({len(videos)} total):")
        
        # Show first few
        for video_name in videos[:5]:
            video_path = os.path.join(set_path, video_name)
            frames = [f for f in os.listdir(video_path) if f.endswith('.png')]
            print(f"  {video_name}: {len(frames)} frames")
            
            # Show first few frame names
            if frames:
                print(f"    Samples: {frames[:3]}")

# Now test if we can find specific images
print("\n\n" + "=" * 80)
print("🔎 TESTING IMAGE SEARCH")
print("=" * 80)

test_filename = "04511.png"
print(f"\nSearching for: {test_filename}")

found = False
for root, dirs, files in os.walk(IMAGES_DIR):
    if test_filename in files:
        full_path = os.path.join(root, test_filename)
        print(f"✅ FOUND: {full_path}")
        found = True
        break

if not found:
    print(f"❌ NOT FOUND: {test_filename}")

# Test another
test_filename2 = "04509.png"
print(f"\nSearching for: {test_filename2}")

found2 = False
for root, dirs, files in os.walk(IMAGES_DIR):
    if test_filename2 in files:
        full_path = os.path.join(root, test_filename2)
        print(f"✅ FOUND: {full_path}")
        found2 = True
        break

if not found2:
    print(f"❌ NOT FOUND: {test_filename2}")

# Count total images
print("\n\n" + "=" * 80)
print("📊 TOTAL IMAGE COUNT")
print("=" * 80)

total_images = 0
for root, dirs, files in os.walk(IMAGES_DIR):
    for f in files:
        if f.endswith('.png'):
            total_images += 1

print(f"\nTotal PNG files in {IMAGES_DIR}: {total_images}")

if total_images == 0:
    print("⚠️  WARNING: No PNG files found!")
    print("Images might be missing or in wrong format")
else:
    print(f"✅ Images found: {total_images}")

print("\n" + "=" * 80)
