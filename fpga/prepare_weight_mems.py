import re
from pathlib import Path

src = Path('fpga_weights/tinymobilenet_xs_weights.h')
out_dir = Path('fpga/weights_mem')
out_dir.mkdir(parents=True, exist_ok=True)

text = src.read_text(encoding='utf-8', errors='ignore')
pattern = re.compile(r'int8_t\s+weights_(\d+)\[(\d+)\]\s*=\s*\{([^}]*)\};', re.S)

count = 0
for m in pattern.finditer(text):
    idx = int(m.group(1))
    body = m.group(3)
    vals = [v.strip() for v in body.replace('\n', ' ').split(',') if v.strip()]
    nums = [int(v) for v in vals]
    mem_path = out_dir / f'weights_{idx:02d}.mem'
    with mem_path.open('w') as f:
        for n in nums:
            f.write(f"{(n & 0xFF):02X}\n")
    count += 1

print(f'exported_layers={count}')
print(f'output_dir={out_dir}')
