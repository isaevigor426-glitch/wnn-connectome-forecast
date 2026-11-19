"""Source-only transfer bundle; never a competition submission."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
root=Path(__file__).resolve().parents[1]
output=root/'notebooks/colab_source.zip'
files=list(root.glob('*.md'))+[root/'ASSET_MANIFEST.json']
for directory in ('src','scripts','tests','reference','notebooks'):
    files.extend(p for p in (root/directory).rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json','.txt','.ipynb') and '__pycache__' not in p.parts)
with ZipFile(output,'w',ZIP_DEFLATED) as z:
    for p in sorted(set(files)): z.write(p,Path(root.name)/p.relative_to(root))
print(f'{output}: {output.stat().st_size} bytes; source only')
