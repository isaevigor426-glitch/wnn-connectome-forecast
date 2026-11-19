"""Compare exported engine with saved torch predictions on all real held-out X."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from types import SimpleNamespace
from tqdm.auto import tqdm
from solution import PredictionModel
from data import HOLDOUT
from checkpoint import digest

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('weights','data-dir','prediction','out'): p.add_argument('--'+k,required=True)
    a=p.parse_args(); root=Path(a.data_dir)
    if json.loads((root/'manifest.json').read_text())['holdout']!=HOLDOUT.tolist(): raise ValueError('Wrong holdout')
    x=np.load(root/'valid300_X.npy',mmap_mode='r'); y=np.load(a.prediction,mmap_mode='r')
    if x.shape!=(300,20000,112) or y.shape!=(300,20000,2) or y.dtype!=np.float32: raise ValueError('Wrong shape/dtype')
    e=PredictionModel(a.weights); err=0.; start=time.monotonic()
    for i in tqdm(range(300),desc='Numba export verification'):
        for t in range(20000):
            o=e.predict(SimpleNamespace(seq_ix=i,state=x[i,t],need_prediction=t>=99))
            if t<99:
                if o is not None: raise AssertionError('Warmup')
            else:
                if not np.isfinite(o).all() or not np.isfinite(y[i,t]).all(): raise ValueError('Nonfinite prediction')
                err=max(err,float(np.abs(o-y[i,t]).max()))
    report={'weights_sha256':digest(a.weights),'prediction_sha256':digest(a.prediction),'sequences':300,'max_error':err,'passed':err<1e-3,'elapsed_seconds':time.monotonic()-start}
    Path(a.out).write_text(json.dumps(report,indent=2)); print(report)
