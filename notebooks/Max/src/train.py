"""Colab T4 float32 TBPTT trainer; resumes at the last committed complete batch."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,math,random,time
from pathlib import Path
import numpy as np
import torch
from tqdm.auto import tqdm
from models import Net,detach
from data import HOLDOUT,event_features
from metrics import score
from checkpoint import save,load,rng_state,restore_rng,digest

class Dataset:
    def __init__(self,path,event=False):
        self.path=Path(path); self.manifest=json.loads((self.path/'manifest.json').read_text())
        if self.manifest['holdout']!=HOLDOUT.tolist() or self.manifest['counts']!=[10607,1573,300]: raise ValueError('Invalid holdout manifest')
        self.x=[np.load(self.path/f'{s}_X.npy',mmap_mode='r') for s in ('train','extra')]
        self.y=[np.load(self.path/f'{s}_Y.npy',mmap_mode='r') for s in ('train','extra')]
        self.vx=np.load(self.path/'valid300_X.npy',mmap_mode='r'); self.vy=np.load(self.path/'valid300_Y.npy',mmap_mode='r'); self.vm=np.load(self.path/'valid300_M.npy',mmap_mode='r')
        for x,y,n in zip(self.x,self.y,[10607,1573]):
            if x.shape!=(n,20000,112) or x.dtype!=np.float16 or y.shape!=(n,20000,2) or y.dtype!=np.float32: raise ValueError('Invalid training memmap')
        if self.vx.shape!=(300,20000,112) or self.vx.dtype!=np.float16 or self.vy.shape!=(300,20000,2) or self.vm.shape!=(300,20000): raise ValueError('Invalid validation memmap')
        self.event=event; self.mu=np.zeros(8); self.sd=np.ones(8)
        if event:
            sample=np.concatenate([event_features(self.x[0][i]) for i in range(0,2000,100)]).astype(np.float64)
            self.mu=sample.mean(0); self.sd=np.maximum(sample.std(0),1e-6)
    def __len__(self): return sum(map(len,self.x))
    def features(self,x):
        x=np.asarray(x,np.float32)
        return np.concatenate([x,(event_features(x)-self.mu)/self.sd],axis=1).astype(np.float32) if self.event else x
    def batch(self,indices):
        xs=[]; ys=[]
        for i in indices:
            k=0 if i<len(self.x[0]) else 1; j=i if k==0 else i-len(self.x[0])
            xs.append(self.features(self.x[k][j])); ys.append(self.y[k][j])
        return np.stack(xs),np.stack(ys)

@torch.no_grad()
def predict(model,data,path,device,chunk=1000):
    p=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=data.vy.shape)
    model.eval()
    for i in tqdm(range(len(data.vx)),desc='valid300'):
        x=data.features(data.vx[i]); state=None
        for t in range(0,len(x),chunk):
            o,state=model(torch.from_numpy(x[None,t:t+chunk]).to(device),state)
            p[i,t:t+chunk]=o[0].cpu().numpy()
    p.flush(); model.train()
    return p

def train_batch(model,opt,sched,x,y,chunk=1000,warm=99):
    state=None; losses=[]
    for t in range(0,x.shape[1],chunk):
        o,state=model(x[:,t:t+chunk],state); state=detach(state)
        start=max(0,warm-t)
        if start>=o.shape[1]: continue
        target=y[:,t:t+chunk].clamp(-2,2)
        loss=(target[:,start:].abs()*(o[:,start:]-target[:,start:])**2).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step(); sched.step(); losses.append(loss.item())
    return float(np.mean(losses))

def run(a):
    if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name(): raise RuntimeError('Full training requires Colab T4')
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    data=Dataset(a.data_dir,a.model=='gru_ev'); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    config={k:v for k,v in vars(a).items() if k not in ('resume','checkpoint_interval','out','data_dir')}
    config['manifest_sha256']=digest(data.path/'manifest.json')
    model=Net(a.model,a.hidden).cuda(); opt=torch.optim.Adam(model.parameters(),lr=.002)
    nb=math.ceil(len(data)/32); total=a.epochs*nb*20
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.002,total_steps=total,pct_start=.05)
    ep=pos=steps=0; epoch_loss_sum=0.; epoch_batches=0; best=-float('inf'); history=[]; order=None
    if a.resume:
        ck=load(out)
        if ck['config']!=config: raise ValueError('Resume configuration differs')
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); sched.load_state_dict(ck['scheduler'])
        ep,pos,steps,best,order,history=[ck[k] for k in ('epoch','next_batch','steps','best','order','history')]
        epoch_loss_sum=ck['epoch_loss_sum']; epoch_batches=ck['epoch_batches']
        restore_rng(ck['rng'])
        if sched.last_epoch!=steps: raise ValueError('Scheduler/optimizer position mismatch')
    def checkpoint(label='last'):
        save(out,dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),epoch=ep,next_batch=pos,steps=steps,best=best,order=order,history=history,epoch_loss_sum=epoch_loss_sum,epoch_batches=epoch_batches,config=config,rng=rng_state(),ev_mu=data.mu,ev_sd=data.sd),label)
    began=time.monotonic(); initial_steps=steps
    while ep<a.epochs:
        if order is None: order=np.random.default_rng(a.seed+ep).permutation(len(data))
        loss=None
        bar=tqdm(range(pos,nb),initial=pos,total=nb,desc=f'Epoch {ep+1}/{a.epochs}')
        for b in bar:
            x,y=data.batch(order[b*32:(b+1)*32])
            loss=train_batch(model,opt,sched,torch.from_numpy(x).cuda(),torch.from_numpy(y).cuda())
            steps+=20; pos=b+1; epoch_loss_sum+=loss; epoch_batches+=1
            elapsed=time.monotonic()-began; eta=elapsed/max(steps-initial_steps,1)*(total-steps)
            bar.set_postfix(loss=f'{loss:.5f}',eta_hours=f'{eta/3600:.2f} ~')
            if pos%a.checkpoint_interval==0: checkpoint()
        # Commit completed training before validation; resume repeats validation if interrupted.
        checkpoint()
        pred=predict(model,data,out/'P_epoch.npy','cuda'); sc=score(pred,data.vy,data.vm)
        history.append({'epoch':ep+1,'score':float(sc.mean()),'targets':sc.tolist(),'train_loss':epoch_loss_sum/max(1,epoch_batches),'last_batch_loss':loss,'elapsed_seconds':time.monotonic()-began})
        if sc.mean()>best: best=float(sc.mean()); checkpoint('best')
        ep+=1; pos=0; order=None; epoch_loss_sum=0.; epoch_batches=0; checkpoint()
        (out/'history.json').write_text(json.dumps(history,indent=2)); print(history[-1])
    ck=load(out,'best'); model.load_state_dict(ck['model'])
    pred=predict(model,data,out/'P_lstm_valid300.npy' if a.model=='lstm' else out/f'P_{a.model}_valid300.npy','cuda')
    pred_path=Path(pred.filename)
    (out/'prediction_provenance.json').write_text(json.dumps({'holdout':HOLDOUT.tolist(),'prediction_sha256':digest(pred_path),'target_sha256':digest(data.path/'valid300_Y.npy'),'mask_sha256':digest(data.path/'valid300_M.npy'),'checkpoint':json.loads((out/'best.json').read_text()),'manifest_sha256':config['manifest_sha256']},indent=2))
    print('Best checkpoint score:',score(pred,data.vy,data.vm))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--model',choices=['gru','gru_ev','gru_wide','lstm'],default='lstm'); p.add_argument('--hidden',type=int,default=128); p.add_argument('--seed',type=int,default=0); p.add_argument('--epochs',type=int,default=5); p.add_argument('--data_dir',required=True); p.add_argument('--out',required=True); p.add_argument('--resume',action='store_true'); p.add_argument('--checkpoint_interval',type=int,default=1)
    run(p.parse_args())
