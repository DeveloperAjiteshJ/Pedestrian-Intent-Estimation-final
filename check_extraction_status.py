"""Check extraction status for set02 videos"""
from pathlib import Path

print("="*60)
print("SET02 FRAME EXTRACTION STATUS")
print("="*60)

videos = [
    ('video_0001', 18000),
    ('video_0002', 18000),
    ('video_0003', 6990)
]

total_expected = 0
total_extracted = 0

for video_name, total_frames in videos:
    video_path = Path(f'./images/set02/{video_name}')

    if video_path.exists():
        extracted_png = len(list(video_path.glob('*.png')))
        extracted_jpg = len(list(video_path.glob('*.jpg')))
        extracted = extracted_png + extracted_jpg
    else:
        extracted_png = 0
        extracted_jpg = 0
        extracted = 0

    pct = (100 * extracted / total_frames) if total_frames > 0 else 0
    status = "✅" if pct >= 95 else "⚠️" if extracted > 0 else "❌"

    print(f"{status} {video_name}: {extracted:5d} / {total_frames:5d} ({pct:5.1f}%)")
    if extracted_png > 0 and extracted_jpg > 0:
        print(f"   Note: {extracted_png} PNG + {extracted_jpg} JPEG")

    total_expected += total_frames
    total_extracted += extracted

print("="*60)
print(f"TOTAL: {total_extracted:5d} / {total_expected:5d} ({100*total_extracted/total_expected:5.1f}%)")
print("="*60)
