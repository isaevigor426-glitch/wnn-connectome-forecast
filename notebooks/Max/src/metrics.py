"""Offline-only family reweighted Pearson; never imported by inference."""
import numpy as np
UN=np.array([.1254,.2413,.4685,.6569])
MIX=np.array([.2125,.1853,.2301,.3721])

def families(y):
    a=np.abs(np.clip(y[:,:,0],-2,2)).astype(np.float64)
    mn=np.where(a>0,a,9).min(1)
    return np.abs(mn[:,None]-UN).argmin(1)

def sums(p,y,m):
    if p.shape!=y.shape or m.shape!=y.shape[:-1] or not np.isfinite(p).all():
        raise ValueError('Invalid prediction shape or values')
    mask=m.copy(); mask[:99]=False
    y=np.clip(y[mask],-2,2).astype(np.float64)
    p=np.clip(p[mask],-2,2).astype(np.float64); w=np.abs(y)
    return np.stack([w.sum(0),(w*y).sum(0),(w*p).sum(0),(w*y*y).sum(0),(w*p*p).sum(0),(w*y*p).sum(0)],1)

def corr(s):
    if (s[:,0]<=0).any(): raise ValueError('No weighted observations')
    my,mp=s[:,1]/s[:,0],s[:,2]/s[:,0]
    den=(s[:,3]/s[:,0]-my**2)*(s[:,4]/s[:,0]-mp**2)
    return np.where(den>1e-16,(s[:,5]/s[:,0]-my*mp)/np.sqrt(np.maximum(den,1e-16)),0)

def sequence_sums(p,y,m):
    return np.stack([sums(p[i],y[i],m[i]) for i in range(len(y))])

def score_sums(s,f):
    share=np.bincount(f,minlength=4)/len(f)
    if (share==0).any(): raise ValueError('Missing validation family')
    return corr((s*(MIX/share)[f,None,None]).sum(0))

def score(p,y,m): return score_sums(sequence_sums(p,y,m),families(y))
