import numpy as np
from pathlib import Path

bs=49152
X=np.array([int(x.strip(),16) for x in open('fpga_test_vectors/all_samples.mem')],dtype=np.int32).reshape(-1,bs)
y=np.array([int(x.strip(),2) for x in open('fpga_test_vectors/sample_labels.mem')],dtype=np.int32)

w13=np.array([int(x.strip(),16) for x in open('fpga/weights_mem/weights_13.mem')],dtype=np.uint8).astype(np.int16)
w13=(w13+128)%256-128
w13=w13.reshape(32,48)

w14=np.array([int(x.strip(),16) for x in open('fpga/weights_mem/weights_14.mem')],dtype=np.uint8).astype(np.int16)
w14=(w14+128)%256-128
w14=w14.reshape(2,32)

chunks=X.reshape(len(X),48,1024).sum(axis=2)

best=(0,0,0)
for fshift in range(9,15):
    fq=np.clip(chunks>>fshift,0,127).astype(np.int16)
    for hshift in range(8,17):
        h=(fq@w13.T)
        h=np.maximum(h,0)
        h=np.clip(h>>hshift,0,127).astype(np.int16)
        out=h@w14.T
        pred=(out[:,1]>=out[:,0]).astype(np.int32)
        acc=(pred==y).mean()
        if acc>best[2]:
            best=(fshift,hshift,acc)
print('best',best)
