import importlib.util,io,tarfile
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('acquire',Path(__file__).resolve().parents[1]/'scripts/acquire_data.py')
a=importlib.util.module_from_spec(spec); spec.loader.exec_module(a)

def archive(path,unsafe=False):
    with tarfile.open(path,'w:gz') as f:
        for name in ('train','valid'):
            info=tarfile.TarInfo(('../' if unsafe else 'starter/datasets/')+name+'.parquet'); info.size=4
            f.addfile(info,io.BytesIO(b'test'))

def test_archive_integrity(tmp_path):
    p=tmp_path/'a.tar.gz'; archive(p)
    assert set(a.check_archive(p))=={'train','valid'}
    p.write_bytes(p.read_bytes()[:-8])
    with pytest.raises((EOFError,OSError)): a.check_archive(p)

def test_unsafe_archive(tmp_path):
    p=tmp_path/'a.tar.gz'; archive(p,True)
    with pytest.raises(ValueError,match='Unsafe'): a.check_archive(p)

def test_reuse_no_download(tmp_path,monkeypatch):
    from types import SimpleNamespace
    for name in ('train','valid'): (tmp_path/f'{name}.parquet').write_bytes(b'fake')
    monkeypatch.setattr(a,'inspect',lambda path,name:list(range(112)))
    monkeypatch.setattr(a,'download',lambda *args:pytest.fail('Unexpected download'))
    result=a.acquire(SimpleNamespace(out=tmp_path,search=[],archive=tmp_path/'absent',url='unused',attempts=1))
    assert set(result)=={'train','valid'}
