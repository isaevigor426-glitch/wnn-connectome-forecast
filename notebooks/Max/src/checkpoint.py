"""Versioned, checksummed commits: an interrupted copy never replaces last.json."""
import hashlib,json,os,random,shutil,tempfile,uuid
from pathlib import Path
import numpy as np
import torch

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def save(root,state,label='last'):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    name=f'{label}-{uuid.uuid4().hex}.pt'
    with tempfile.TemporaryDirectory() as tmp:
        local=Path(tmp)/name
        torch.save(state,local)
        sha=digest(local)
        dest=root/name; shutil.copyfile(local,dest)
        if digest(dest)!=sha: raise IOError('Checkpoint copy checksum mismatch')
        pointer=root/f'{label}.json.partial'
        with pointer.open('w') as f:
            json.dump({'file':name,'sha256':sha},f); f.flush(); os.fsync(f.fileno())
        pointer.replace(root/f'{label}.json')
    # Keep two versions so a previous commit survives Drive propagation failure.
    versions=sorted(root.glob(f'{label}-*.pt'),key=lambda p:p.stat().st_mtime,reverse=True)
    for old in versions[2:]: old.unlink()

def load(root,label='last'):
    root=Path(root); info=json.loads((root/f'{label}.json').read_text()); path=root/info['file']
    if digest(path)!=info['sha256']: raise ValueError('Checkpoint checksum failed')
    return torch.load(path,map_location='cpu',weights_only=False) # only own trusted checkpoints

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def restore_rng(s):
    random.setstate(s['python']); np.random.set_state(s['numpy']); torch.set_rng_state(s['torch'])
    if s['cuda'] is not None: torch.cuda.set_rng_state_all(s['cuda'])
