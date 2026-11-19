"""Causal data utilities. Family labels deliberately do not live here."""
import numpy as np

HOLDOUT = np.sort(np.random.default_rng(1).choice(1873,300,replace=False))

def event_features(x):
    # Reuse exact technical reference, including rolling-window endpoints.
    from solution import _evfeat
    prev=np.zeros(112,np.float32); ei=np.zeros(5,np.int64)
    mb=np.zeros(10); flags=np.zeros(200,np.int64); xe=np.zeros(120,np.float32)
    out=np.empty((len(x),8),np.float32)
    for t,row in enumerate(x):
        _evfeat(np.asarray(row,np.float32),prev,t==0,ei,mb,flags,xe,np.zeros(8),np.ones(8))
        out[t]=xe[112:]; prev[:]=row
    return out
