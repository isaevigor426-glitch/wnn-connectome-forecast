"""Manual Colab inference for an exact equal-weight native checkpoint candidate."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from tqdm.auto import tqdm
from train import Dataset
from models import Net
from checkpoint import load,digest
from data import HOLDOUT

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--checkpoints',nargs='+',required=True); p.add_argument('--data-dir',required=True); p.add_argument('--out',required=True); a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('Run this inference manually in Colab')
    data=Dataset(a.data_dir); models=[]
    for root in a.checkpoints:
        ck=load(root,'best'); m=Net(ck['config']['model'],ck['config']['hidden']).cuda().eval(); m.load_state_dict(ck['model']); models.append((m,ck))
    pred=np.lib.format.open_memmap(a.out,mode='w+',dtype=np.float32,shape=(300,20000,2))
    with torch.no_grad():
        for i in tqdm(range(300)):
            pred[i]=0
            for model,ck in models:
                data.event=model.kind=='gru_ev'; data.mu=ck['ev_mu']; data.sd=ck['ev_sd']; x=data.features(data.vx[i]); state=None
                for t in range(0,20000,1000):
                    o,state=model(torch.from_numpy(x[None,t:t+1000]).cuda(),state)
                    pred[i,t:t+1000]+=o[0].cpu().numpy()/len(models)
    pred.flush()
    provenance={'holdout':HOLDOUT.tolist(),'prediction_sha256':digest(a.out),'target_sha256':digest(data.path/'valid300_Y.npy'),'mask_sha256':digest(data.path/'valid300_M.npy'),'checkpoints':[json.loads((Path(root)/'best.json').read_text()) for root in a.checkpoints]}
    Path(str(a.out)+'.json').write_text(json.dumps(provenance,indent=2))
