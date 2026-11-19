"""Resumable official archive acquisition; never pipe network bytes into tar.
Use --search to reuse existing valid data pairs. Existing valid train is preserved
when valid is damaged. Full compressed CRC and tar scan precede any extraction.
"""
import argparse,gzip,json,os,shutil,subprocess,tarfile,time
from pathlib import Path
import pyarrow.parquet as pq

URL='https://files.wundernn.io/wnn_connectome_starterpack.tar.gz'
SERVICE={'seq_ix','step_in_seq','need_prediction','is_scored','t0','t1'}

def inspect(path,name):
    pf=pq.ParquetFile(path)
    expected={'train':10607,'valid':1873}[name]
    if pf.num_row_groups!=expected: raise ValueError(f'{path}: wrong row group count {pf.num_row_groups}')
    names=pf.schema_arrow.names
    required={'seq_ix','step_in_seq','t0','t1'} | ({'is_scored'} if name=='valid' else set())
    if not required.issubset(names): raise ValueError(f'{path}: missing columns {required-set(names)}')
    features=[n for n in names if n not in SERVICE]
    if len(features)!=112: raise ValueError(f'{path}: expected 112 feature columns')
    if any(pf.metadata.row_group(i).num_rows!=20000 for i in range(expected)): raise ValueError(f'{path}: invalid sequence length')
    # Decode boundary groups; full page decoding occurs during prepare_data.
    for i in (0,expected-1): pf.read_row_group(i,columns=['t0','t1',features[0]])
    return features

def valid(path,name):
    try: return inspect(path,name)
    except Exception as e:
        print(f'Not reusable: {path}: {type(e).__name__}: {e}'); return None

def check_archive(path):
    # Drain gzip to EOF to check CRC/length, including trailing bytes.
    with gzip.open(path,'rb') as f:
        while f.read(8*1024*1024): pass
    members={}
    with tarfile.open(path,'r:gz') as tf:
        for m in tf:
            parts=Path(m.name).parts
            if m.name.startswith('/') or '..' in parts or m.issym() or m.islnk(): raise ValueError('Unsafe archive member')
            if Path(m.name).name in ('train.parquet','valid.parquet') and m.isfile():
                name=Path(m.name).stem
                if name in members: raise ValueError('Ambiguous dataset archive')
                members[name]=(m.name,m.size)
    if set(members)!={'train','valid'}: raise ValueError('Archive lacks train/valid')
    return members

def download(url,archive,attempts):
    partial=archive.with_suffix(archive.suffix+'.partial')
    header=subprocess.run(['curl','--http1.1','--fail','--location','--head','--silent','--show-error','--max-time','60',url],capture_output=True,text=True,check=True).stdout
    lengths=[int(line.split(':',1)[1]) for line in header.splitlines() if line.lower().startswith('content-length:')]
    total=lengths[-1] if lengths else None
    existing=partial.stat().st_size if partial.exists() else 0
    if total and shutil.disk_usage(archive.parent).free < max(0,total-existing)*1.05: raise RuntimeError('Insufficient space for archive download')
    if total and existing==total:
        members=check_archive(partial); partial.replace(archive); return members
    for attempt in range(attempts):
        # curl rejects a server that ignores Range instead of concatenating a full response.
        cmd=['curl','--http1.1','--fail','--location','--retry','5','--retry-delay','5','--retry-all-errors','--connect-timeout','30','--speed-limit','1024','--speed-time','120','--continue-at','-','--output',str(partial),url]
        result=subprocess.run(cmd)
        if result.returncode==0:
            try:
                members=check_archive(partial)
                partial.replace(archive)
                return members
            except (EOFError,OSError,tarfile.TarError,ValueError):
                # A successful HTTP body with invalid CRC cannot be safely resumed.
                # Preserve it for diagnosis, never extract it.
                partial.replace(partial.with_suffix('.corrupt'))
                raise RuntimeError('Downloaded archive failed integrity check; preserved as .corrupt. Rerun for a fresh download.')
        if result.returncode==33:
            raise RuntimeError('Server rejected resume Range. Keep partial or remove it explicitly to restart.')
        if attempt+1<attempts: time.sleep(min(5*(attempt+1),30))
    raise RuntimeError('Download interrupted; .partial retained. Rerun to resume.')

def acquire(a):
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    candidates=[out]+[Path(p) for p in a.search]
    found={}; feature_order=None
    for name in ('train','valid'):
        for root in candidates:
            for d in (root,root/'datasets',root/'wnn_connectome_starterpack/datasets'):
                p=d/f'{name}.parquet'
                if not p.is_file(): continue
                feats=valid(p,name)
                if feats is not None:
                    if feature_order is not None and feats!=feature_order: raise ValueError('Train/valid feature order mismatch')
                    feature_order=feats; found[name]=p.resolve(); break
            if name in found: break
    if len(found)<2:
        archive=Path(a.archive); archive.parent.mkdir(parents=True,exist_ok=True)
        members=check_archive(archive) if archive.exists() else download(a.url,archive,a.attempts)
        required=sum(size for name,(_,size) in members.items() if name not in found)
        if shutil.disk_usage(out).free < required*1.05: raise RuntimeError(f'Need {required*1.05/1e9:.1f} GB for extraction')
        with tarfile.open(archive,'r:gz') as tf:
            for name,(member,_) in members.items():
                if name in found: continue
                dest=out/f'{name}.parquet'; tmp=dest.with_suffix('.parquet.partial')
                with tf.extractfile(member) as source,tmp.open('wb') as target: shutil.copyfileobj(source,target,8*1024*1024)
                feats=inspect(tmp,name)
                if feature_order is not None and feats!=feature_order: raise ValueError('Feature order mismatch')
                feature_order=feats; tmp.replace(dest); found[name]=dest.resolve()
    # Return actual locations. No copying tens of GB when sources live on Drive.
    manifest={name:str(path) for name,path in found.items()}
    (out/'raw_paths.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2)); return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--url',default=URL); p.add_argument('--archive',required=True); p.add_argument('--out',required=True); p.add_argument('--search',nargs='*',default=[]); p.add_argument('--attempts',type=int,default=5); acquire(p.parse_args())
