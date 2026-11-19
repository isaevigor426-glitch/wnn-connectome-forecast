import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from checkpoint import load
from models import Net
from export import export

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint-dir',required=True); p.add_argument('--out',required=True); p.add_argument('--float32',action='store_true'); a=p.parse_args()
    ck=load(a.checkpoint_dir,'best'); cfg=ck['config']; m=Net(cfg['model'],cfg['hidden']); m.load_state_dict(ck['model'])
    if ck['steps']<=0: raise ValueError('Untrained checkpoint')
    export([m],a.out,half=False if a.float32 else None,ev_mu=ck['ev_mu'],ev_sd=ck['ev_sd'])
