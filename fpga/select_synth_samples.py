import random
from pathlib import Path

BYTES_PER_SAMPLE = 49152
MAX_SYNTH_SAMPLES = 6  # Safe number that fits in BRAM
SEED = 42

# Manually selected indices: 4 correct + 2 wrong to ensure at least one mismatch on board
SELECTED_INDICES = [2, 7, 8, 15, 35, 58]

src_samples = Path('../fpga_test_vectors/all_samples.mem')
src_labels = Path('../fpga_test_vectors/sample_labels.mem')
out_samples = Path('../fpga_test_vectors/all_samples_synth.mem')
out_labels = Path('../fpga_test_vectors/sample_labels_synth.mem')

sample_lines = src_samples.read_text().strip().splitlines()
labels = src_labels.read_text().strip().splitlines()
num_total = len(labels)

bytes_total = len(sample_lines)
if bytes_total != num_total * BYTES_PER_SAMPLE:
    raise RuntimeError(f'Mismatch: bytes={bytes_total}, labels={num_total}, expected={num_total*BYTES_PER_SAMPLE}')

# Use manually selected indices
indices = sorted(SELECTED_INDICES[:min(MAX_SYNTH_SAMPLES, len(SELECTED_INDICES))])

with out_samples.open('w') as fs, out_labels.open('w') as fl:
    for i in indices:
        s = i * BYTES_PER_SAMPLE
        e = s + BYTES_PER_SAMPLE
        fs.write('\n'.join(sample_lines[s:e]))
        fs.write('\n')
        fl.write(labels[i].strip() + '\n')

print(f'total_samples={num_total}')
print(f'selected_samples={len(indices)}')
print(f'selected_indices={indices}')
print(f'selected_labels={[labels[i].strip() for i in indices]}')
print(f'total_bytes={len(indices) * BYTES_PER_SAMPLE}')
print(f'total_kb={len(indices) * BYTES_PER_SAMPLE / 1024:.2f} KB')
print(f'output_samples={out_samples}')
print(f'output_labels={out_labels}')
