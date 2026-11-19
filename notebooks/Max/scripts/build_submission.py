"""Package only verified exact weights. No synthetic or default model fallback."""
import argparse,json,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from checkpoint import digest
from solution import PredictionModel

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('weights','validation','benchmark','verification','out'): p.add_argument('--'+k,required=True)
    a=p.parse_args(); wsha=digest(a.weights)
    quality=json.loads(Path(a.validation).read_text()); bench=json.loads(Path(a.benchmark).read_text()); verify=json.loads(Path(a.verification).read_text())
    if not quality.get('quality_pass') or not quality.get('candidate_exact'): raise SystemExit('Blocked: no measured >=0.0015 improvement')
    if not bench.get('performance_pass') or bench.get('weights_sha256')!=wsha: raise SystemExit('Blocked: exact-weight one-core benchmark required')
    if verify.get('weights_sha256')!=wsha or not verify.get('passed') or verify.get('sequences')!=300: raise SystemExit('Blocked: full valid300 export verification required')
    if quality.get('provenance',{}).get('prediction_sha256')!=verify.get('prediction_sha256'): raise SystemExit('Blocked: validation/prediction mismatch')
    PredictionModel(a.weights)
    dest=Path(a.out); tmp=dest.with_suffix('.zip.partial')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        z.write(Path(__file__).resolve().parents[1]/'src/solution.py','solution.py'); z.write(a.weights,'weights.npz')
    tmp.replace(dest); print(dest)
