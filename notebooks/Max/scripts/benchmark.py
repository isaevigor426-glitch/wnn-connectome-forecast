"""One-thread, warmed-up public predict API benchmark, no GPU training."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import argparse,json,platform,sys,tempfile,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from types import SimpleNamespace
from models import Net
from export import export
from solution import PredictionModel
from checkpoint import digest

def measure(path,rows):
    e=PredictionModel(path) # excludes JIT/selftest
    x=np.random.default_rng(0).normal(size=(rows,112)).astype(np.float32)
    timings=[]
    dp=SimpleNamespace(seq_ix=0,state=x[0],need_prediction=True)
    for run in range(3):
        dp.seq_ix=run; start=time.perf_counter()
        for row in x:
            dp.state=row; e.predict(dp)
        timings.append((time.perf_counter()-start)/rows)
    return {'microseconds_per_row':float(np.median(timings)*1e6),'projected_minutes_39_4M':float(max(timings)*39.4e6/60),'repeats_seconds_per_row':timings}

def main(a):
    torch.set_num_threads(1); torch.manual_seed(0)
    affinity=False
    if hasattr(os,'sched_setaffinity'):
        os.sched_setaffinity(0,{min(os.sched_getaffinity(0))}); affinity=True
    result={'platform':platform.platform(),'processor':platform.processor(),'single_thread':True,'core_affinity':affinity,'rows':a.rows}
    with tempfile.TemporaryDirectory() as tmp:
        for kind in ('gru','lstm'):
            path=Path(tmp)/f'{kind}.npz'; export([Net(kind)],path); result[kind]=measure(path,a.rows)
    result['lstm_gru_ratio']=result['lstm']['microseconds_per_row']/result['gru']['microseconds_per_row']
    if a.weights:
        result['ensemble']=measure(a.weights,a.rows); result['weights_sha256']=digest(a.weights)
        result['performance_pass']=result['lstm_gru_ratio']<=1.4 and result['ensemble']['projected_minutes_39_4M']<=54 and affinity
    Path(a.out).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--weights'); p.add_argument('--rows',type=int,default=20000); p.add_argument('--out',required=True); main(p.parse_args())
