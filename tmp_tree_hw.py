import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

bs=49152
X=np.array([int(x.strip(),16) for x in open('fpga_test_vectors/all_samples.mem')],dtype=np.float32).reshape(-1,bs)
y=np.array([int(x.strip(),2) for x in open('fpga_test_vectors/sample_labels.mem')],dtype=np.int32)

se=X[:,::2].sum(axis=1)
so=X[:,1::2].sum(axis=1)
q=bs//4
q0=X[:,:q].sum(axis=1)
q1=X[:,q:2*q].sum(axis=1)
q2=X[:,2*q:3*q].sum(axis=1)
q3=X[:,3*q:].sum(axis=1)
hi=(X>127).sum(axis=1)
lo=(X<64).sum(axis=1)
F=np.stack([se,so,q0,q1,q2,q3,hi,lo],axis=1)

best=(0,None,None)
for d in range(1,8):
    clf=DecisionTreeClassifier(max_depth=d,random_state=0)
    clf.fit(F,y)
    acc=(clf.predict(F)==y).mean()
    if acc>best[0]: best=(acc,d,clf)
print('best',best[0],best[1])
print(export_text(best[2],feature_names=['se','so','q0','q1','q2','q3','hi','lo']))
