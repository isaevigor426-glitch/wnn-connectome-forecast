"""Resumable HTTP download with validated ranges, length and Parquet footer."""
import argparse,json,urllib.request
from pathlib import Path
from tqdm.auto import tqdm
import pyarrow.parquet as pq

def download(url,dest):
    dest=Path(dest); dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists(): pq.ParquetFile(dest); print('Using existing',dest); return
    part=dest.with_suffix(dest.suffix+'.partial'); metadata=part.with_suffix(part.suffix+'.json')
    old=json.loads(metadata.read_text()) if metadata.exists() else {}
    n=part.stat().st_size if part.exists() else 0
    headers={}
    if n and old.get('url')==url and old.get('validator'):
        headers={'Range':f'bytes={n}-','If-Range':old['validator']}
    with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=120) as r:
        append=r.status==206 and bool(headers)
        if append and not r.headers.get('Content-Range','').startswith(f'bytes {n}-'): raise ValueError('Invalid Content-Range')
        if not append: n=0
        length=r.headers.get('Content-Length'); total=n+int(length) if length else None
        import shutil
        if total and shutil.disk_usage(dest.parent).free < total-n: raise ValueError('Insufficient download disk space')
        metadata.write_text(json.dumps({'url':url,'validator':r.headers.get('ETag') or r.headers.get('Last-Modified')}))
        with part.open('ab' if append else 'wb') as f,tqdm(total=total,initial=n,unit='B',unit_scale=True) as bar:
            while b:=r.read(4*1024*1024): f.write(b); bar.update(len(b))
        if total and part.stat().st_size!=total: raise IOError('Incomplete download; rerun to resume')
    pq.ParquetFile(part); part.replace(dest); metadata.unlink(missing_ok=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('url'); p.add_argument('dest'); a=p.parse_args(); download(a.url,a.dest)
