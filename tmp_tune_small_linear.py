import numpy as np
from sklearn.linear_model import LogisticRegression

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
mu=F.mean(axis=0); sd=F.std(axis=0)+1e-6
Fn=(F-mu)/sd

lr=LogisticRegression(max_iter=10000,class_weight='balanced')
lr.fit(Fn,y)
p=lr.predict(Fn)
acc=(p==y).mean()
print('acc',acc)
print('w',lr.coef_[0].tolist())
print('b',float(lr.intercept_[0]))
print('mu',mu.tolist())
print('sd',sd.tolist())
