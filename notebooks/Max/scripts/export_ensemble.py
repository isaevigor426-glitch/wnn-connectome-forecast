"""Export equal-weight native checkpoint ensemble; no legacy partner leakage."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from checkpoint import load
from models import Net
from export import export
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--checkpoints',nargs='+',required=True); p.add_argument('--out',required=True); a=p.parse_args()
    models=[]; mu=sd=None
    for root in a.checkpoints:
        ck=load(root,'best'); cfg=ck['config']
        if ck['steps']<=0: raise ValueError('Untrained checkpoint')
        m=Net(cfg['model'],cfg['hidden']); m.load_state_dict(ck['model']); models.append(m)
        if m.kind=='gru_ev':
            if mu is not None and (not np.array_equal(mu,ck['ev_mu']) or not np.array_equal(sd,ck['ev_sd'])): raise ValueError('Event normalization differs across members')
            mu,sd=ck['ev_mu'],ck['ev_sd']
    export(models,a.out,ev_mu=mu,ev_sd=sd)
