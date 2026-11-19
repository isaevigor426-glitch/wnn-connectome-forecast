#!/usr/bin/env python3
"""Extract binary model weights and valid300 arrays locally without printing contents.
Usage: python3 scripts/extract_assets.py for_codex-20261113T133602Z-1-001.zip
"""
from pathlib import Path
from zipfile import ZipFile
import sys

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "for_codex/"
BINARY_DIRS = ("engine_ens5ev/", "best_public_ens3/", "valid300/")
SUFFIXES = (".npz", ".npy")

def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 scripts/extract_assets.py /path/to/original.zip")
    archive = Path(sys.argv[1]).expanduser().resolve()
    if not archive.is_file():
        raise SystemExit(f"Cannot find original archive: {archive}")
    selected = []
    with ZipFile(archive) as z:
        for member in z.infolist():
            name = member.filename
            if not name.startswith(PREFIX):
                continue
            rel = name[len(PREFIX):]
            if not (rel.endswith(SUFFIXES) and rel.startswith(BINARY_DIRS)):
                continue
            path = ROOT / "assets" / rel
            if not path.resolve().is_relative_to((ROOT / "assets").resolve()):
                raise RuntimeError("Unsafe archive path")
            selected.append((member, path))
        for member, dest in selected:
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_name(dest.name + ".partial")
            with z.open(member) as source, temp.open("wb") as out:
                while buf := source.read(2 * 1024 * 1024):
                    out.write(buf)
            temp.replace(dest)
            print(f"{dest.relative_to(ROOT)} | {dest.stat().st_size / (1024**2):.2f} MiB")
    print(f"Extracted {len(selected)} binary assets to {ROOT / 'assets'}")

if __name__ == "__main__":
    main()
