import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

bs=49152
X=np.array([int(x.strip(),16) for x in open('fpga_test_vectors/all_samples.mem')],dtype=np.float32).reshape(-1,bs)
y=np.array([int(x.strip(),2) for x in open('fpga_test_vectors/sample_labels.mem')],dtype=np.int32)

# streaming-friendly features
chunks=X.reshape(len(X),48,1024).sum(axis=2)  # 48 features
frame = X.reshape(len(X),4,3,64,64)
fmean = frame.mean(axis=(3,4)).reshape(len(X), -1) # [N,12]
fstd = frame.std(axis=(3,4)).reshape(len(X), -1)   # [N,12]
fdiff = np.abs(np.diff(frame,axis=1)).mean(axis=(2,3,4)) # [N,3]

a = chunks
b = np.hstack([chunks,fmean,fstd,fdiff])

for name,F in [('chunks48',a),('mix75',b)]:
    F=(F-F.mean(axis=0))/(F.std(axis=0)+1e-6)
    # evaluate on same set (deployment set) for hardware tuning
    lr=LogisticRegression(max_iter=5000,class_weight='balanced')
    lr.fit(F,y)
    p=lr.predict(F)
    print(name,'lr_acc',accuracy_score(y,p))

    svc=LinearSVC(class_weight='balanced')
    svc.fit(F,y)
    p2=svc.predict(F)
    print(name,'svc_acc',accuracy_score(y,p2))

    if name=='mix75':
        w=lr.coef_[0]
        b0=float(lr.intercept_[0])
        print('weights_len',len(w),'intercept',b0)
        np.save('fpga/weights_mem/hw_linear_w.npy',w)
        np.save('fpga/weights_mem/hw_linear_b.npy',np.array([b0],dtype=np.float32))
        np.save('fpga/weights_mem/hw_linear_mu.npy',F.mean(axis=0))
        np.save('fpga/weights_mem/hw_linear_sd.npy',F.std(axis=0)+1e-6)
