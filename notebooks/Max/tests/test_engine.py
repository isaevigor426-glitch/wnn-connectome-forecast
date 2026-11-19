import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest
import torch
from types import SimpleNamespace
from models import Net
from export import export
from solution import PredictionModel

torch.set_num_threads(1)

@pytest.mark.parametrize('half',[False,True])
def test_full_parity(tmp_path,half):
    torch.manual_seed(0)
    m=Net().eval(); path=tmp_path/'weights.npz'
    q=export([m],path,half=half)[0]
    engine=PredictionModel(path)
    maxerr=rawerr=0.
    for seq in range(2):
        x=np.random.default_rng(seq).normal(size=(20000,112)).astype(np.float32)
        with torch.no_grad():
            expected=q(torch.from_numpy(x[None]))[0][0].numpy()
            raw=m(torch.from_numpy(x[None]))[0][0].numpy()
        actual=[]
        for t,row in enumerate(x):
            p=engine.predict(SimpleNamespace(seq_ix=seq,state=row,need_prediction=t>=99))
            if t<99: assert p is None
            else: actual.append(p)
        actual=np.array(actual)
        maxerr=max(maxerr,float(np.max(np.abs(actual-expected[99:]))))
        rawerr=max(rawerr,float(np.max(np.abs(actual-raw[99:]))))
    print(f'half={half} quantized_error={maxerr:.9g} original_error={rawerr:.9g}')
    assert maxerr<1e-3 and rawerr<1e-3


def test_mixed_reset(tmp_path):
    torch.manual_seed(3)
    p=tmp_path/'w.npz'; models=export([Net('gru',16),Net('lstm',16)],p)
    e=PredictionModel(p); x=np.random.default_rng(4).normal(size=(120,112)).astype(np.float32)
    with torch.no_grad(): y=sum(m(torch.tensor(x[None]))[0][0].numpy()/2 for m in models)
    for seq in range(2):
        for t,row in enumerate(x):
            o=e.predict(SimpleNamespace(seq_ix=seq,state=row,need_prediction=True))
            np.testing.assert_allclose(o,y[t],atol=1e-5)

def test_legacy_regression(tmp_path):
    import importlib.util,shutil
    root=Path(__file__).resolve().parents[1]
    weights=root.parent/'for_codex/engine_ens5ev/weights.npz'
    if not weights.exists(): pytest.skip('Legacy weights not installed')
    shutil.copyfile(root/'reference/engine_ens5ev/solution.py',tmp_path/'legacy.py')
    (tmp_path/'weights.npz').symlink_to(weights)
    spec=importlib.util.spec_from_file_location('legacy',tmp_path/'legacy.py'); module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    old=module.PredictionModel(); new=PredictionModel(weights)
    for seq in range(2):
        for t,row in enumerate(np.random.default_rng(1).normal(size=(120,112)).astype(np.float32)):
            dp=SimpleNamespace(seq_ix=seq,state=row,need_prediction=t>=99)
            a,b=old.predict(dp),new.predict(dp)
            if a is not None: np.testing.assert_allclose(a,b,atol=1e-6)
            else: assert b is None

def test_invalid_type_and_offsets(tmp_path):
    p=tmp_path/'w.npz'; export([Net('lstm',8),Net('gru',8)],p)
    with np.load(p) as w: data={k:w[k] for k in w.files}
    data['mods'][1,2]=0; np.savez(p,**data)
    with pytest.raises(ValueError,match='overlapping'): PredictionModel(p)
    data['mods'][1,0]=99; np.savez(p,**data)
    with pytest.raises(ValueError,match='Unknown'): PredictionModel(p)
