"""Honest fixed-holdout comparison, reference check and paired sequence bootstrap."""
import argparse,importlib.util,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from metrics import families,sequence_sums,score_sums,MIX
from checkpoint import digest
from data import HOLDOUT

def main(a):
    root=Path(a.assets); y=np.load(root/'valid300_Y.npy',mmap_mode='r'); m=np.load(root/'valid300_M.npy',mmap_mode='r')
    if y.shape!=(300,20000,2) or m.shape!=(300,20000): raise ValueError('Invalid holdout shape')
    f=families(y); predictions={}
    for name in ('b','gru6573','corr3k','gruall','gruev','gruh256'):
        p=np.load(root/f'P{name if name=="b" else "_"+name}_valid300.npy',mmap_mode='r')
        if p.shape!=y.shape or p.dtype!=np.float32 or not np.isfinite(p).all(): raise ValueError(name)
        predictions[name]=p*.3 if name=='b' else p
    base=(predictions['b']+predictions['gru6573']+predictions['corr3k'])/3
    result={'scores':{},'family_counts':np.bincount(f,minlength=4).tolist(),'holdout':HOLDOUT.tolist(),'quality_pass':False}
    for name,p in {**predictions,'honest3':base}.items(): result['scores'][name]=score_sums(sequence_sums(p,y,m),f).tolist()
    # Compare independent implementation with original reference family-wise sums.
    spec=importlib.util.spec_from_file_location('reference_lib',Path(__file__).resolve().parents[1]/'reference/lib.py')
    ref=importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
    total=np.zeros((2,6)); share=np.bincount(f,minlength=4)/300
    for k in range(4): total+=MIX[k]/share[k]*ref.sums_np(base[f==k],y[f==k],m[f==k])['is_scored']
    reference=ref.corr(total); independent=np.mean(result['scores']['honest3'])
    if abs(reference-independent)>1e-10: raise AssertionError('Reference metric disagreement')
    result['reference_difference']=abs(reference-independent)
    if a.lstm:
        p=np.load(a.lstm,mmap_mode='r')
        if p.dtype!=np.float32 or p.shape!=y.shape: raise ValueError('LSTM dtype/shape mismatch')
        provenance=json.loads(Path(a.provenance).read_text())
        if provenance['holdout']!=HOLDOUT.tolist() or provenance['prediction_sha256']!=digest(a.lstm): raise ValueError('Prediction provenance mismatch')
        if provenance.get('target_sha256')!=digest(root/'valid300_Y.npy') or provenance.get('mask_sha256')!=digest(root/'valid300_M.npy'): raise ValueError('Holdout targets/masks differ from bundled assets')
        result['provenance']=provenance
        blend=p if a.candidate else (3*base+p)/4
        result['candidate_exact']=a.candidate
        bs=sequence_sums(base,y,m); ls=sequence_sums(blend,y,m)
        result['scores']['lstm']=score_sums(sequence_sums(p,y,m),f).tolist()
        result['scores']['honest3_plus_lstm']=score_sums(ls,f).tolist()
        gain=float(score_sums(ls,f).mean()-score_sums(bs,f).mean())
        rng=np.random.default_rng(0); gains=[]
        for _ in range(1000):
            ix=np.concatenate([rng.choice(np.flatnonzero(f==k),size=(f==k).sum(),replace=True) for k in range(4)])
            gains.append(score_sums(ls[ix],f[ix]).mean()-score_sums(bs[ix],f[ix]).mean())
        result.update(gain=gain,bootstrap_ci95=np.quantile(gains,[.025,.975]).tolist(),quality_pass=gain>=.0015)
    Path(a.out).write_text(json.dumps(result,indent=2)); print(json.dumps(result['scores'],indent=2)); print('reference difference',result['reference_difference'])

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--assets',required=True); p.add_argument('--lstm'); p.add_argument('--candidate',action='store_true'); p.add_argument('--provenance'); p.add_argument('--out',required=True); main(p.parse_args())
