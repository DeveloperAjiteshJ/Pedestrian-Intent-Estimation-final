import pickle
files=['data_cache/sequences/train_sequences.pkl','data_cache/sequences/test_sequences.pkl','data_cache/sequences/set02_sequences.pkl']
for p in files:
    with open(p,'rb') as f:
        d=pickle.load(f)
    y=[int(x[0][0]) for x in d['intention_binary']]
    n=len(d['image'])
    c0=sum(v==0 for v in y)
    c1=sum(v==1 for v in y)
    print(p,n,c0,c1)
