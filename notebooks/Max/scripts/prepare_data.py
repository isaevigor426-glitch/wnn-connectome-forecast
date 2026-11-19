"""Prepare float16 X memmaps without materializing the dataset in RAM."""
import argparse,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from data import HOLDOUT

SERVICE={'seq_ix','step_in_seq','need_prediction','is_scored','t0','t1'}

def prepare(source,out):
    source,out=Path(source),Path(out); out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists(): raise ValueError('Prepared manifest already exists; use existing data or a new directory')
    paths=json.loads((source/'raw_paths.json').read_text()) if (source/'raw_paths.json').exists() else {name:str(source/f'{name}.parquet') for name in ('train','valid')}
    files={name:pq.ParquetFile(paths[name]) for name in ('train','valid')}
    feats=None
    for name,pf in files.items():
        expected=10607 if name=='train' else 1873
        if pf.num_row_groups!=expected: raise ValueError(f'{name}: expected {expected} row groups')
        if any(pf.metadata.row_group(i).num_rows!=20000 for i in range(expected)): raise ValueError('Invalid sequence length')
        ff=[c for c in pf.schema_arrow.names if c not in SERVICE]
        if len(ff)!=112 or (feats is not None and ff!=feats): raise ValueError('Feature schema/order mismatch')
        feats=ff
    required=12480*20000*(112*2+2*4+1)
    if shutil.disk_usage(out).free < required*1.05: raise ValueError(f'Need {required*1.05/1e9:.1f} GB free for memmaps')
    groups=[('train',list(range(10607))+[],files['train']),('extra',np.setdiff1d(np.arange(1873),HOLDOUT).tolist(),files['valid']),('valid300',HOLDOUT.tolist(),files['valid'])]
    for name,indices,pf in groups:
        arrays={k:np.lib.format.open_memmap(out/f'{name}_{k}.npy',mode='w+',dtype=dt,shape=(len(indices),20000,*tail)) for k,dt,tail in [('X',np.float16,(112,)),('Y',np.float32,(2,))]}
        if name=='valid300': arrays['M']=np.lib.format.open_memmap(out/f'{name}_M.npy',mode='w+',dtype=bool,shape=(300,20000))
        for j,i in enumerate(tqdm(indices,desc=name)):
            columns=feats+['t0','t1','seq_ix','step_in_seq']+(['is_scored'] if name=='valid300' else [])
            tb=pf.read_row_group(int(i),columns=columns)
            if len(np.unique(tb['seq_ix'].to_numpy()))!=1 or not np.array_equal(tb['step_in_seq'].to_numpy(),np.arange(20000)): raise ValueError('Sequence order invalid')
            x=np.column_stack([tb[c].to_numpy() for c in feats]); y=np.column_stack([tb[c].to_numpy() for c in ('t0','t1')])
            if not np.isfinite(x).all() or not np.isfinite(y).all() or np.abs(x).max()>65504: raise ValueError('Nonfinite or float16 overflow')
            arrays['X'][j]=x; arrays['Y'][j]=y
            if 'M' in arrays: arrays['M'][j]=tb['is_scored'].to_numpy()
        for a in arrays.values(): a.flush()
    manifest={'features':feats,'holdout':HOLDOUT.tolist(),'counts':[10607,1573,300],'length':20000,'dtype':'float16','source_sizes':{k:Path(paths[k]).stat().st_size for k in files}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--source',required=True); p.add_argument('--out',required=True); a=p.parse_args(); prepare(a.source,a.out)
