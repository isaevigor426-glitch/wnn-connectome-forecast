"""Put the weights of an ONNX model back INSIDE the .onnx file (one self-contained file).

Recent PyTorch exporters (torch.onnx.export with dynamo=True, the default in new versions)
write the weights to a separate file such as `model.onnx.data`. The .onnx file then only
references that file by name, and a zip without it fails on the platform with
"External data path does not exist".

    pip install onnx onnxruntime
    python embed_onnx_weights.py baseline.onnx                 # rewrites baseline.onnx in place
    python embed_onnx_weights.py baseline.onnx --out fixed.onnx

The data file must sit next to the .onnx file under the name stored inside the model
(printed below; for your model: model.onnx.data - renaming the .onnx does not rename it).
After writing, the script loads the new file alone in an empty folder and checks that it
gives exactly the same outputs as the original model.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx


def external_locations(path: Path) -> set[str]:
    m = onnx.load(str(path), load_external_data=False)
    return {dict((e.key, e.value) for e in t.external_data).get("location", "?")
            for t in m.graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL}


def random_feeds(sess, rng):
    feeds = {}
    for i in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in i.shape]
        feeds[i.name] = rng.standard_normal(shape).astype(np.float32)
    return feeds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx_file")
    ap.add_argument("--out", default=None, help="default: overwrite the input file (a .bak copy is kept)")
    args = ap.parse_args()

    src = Path(args.onnx_file).resolve()
    locs = external_locations(src)
    if not locs:
        print(f"{src.name}: weights are already inside the file - nothing to do")
        return
    print(f"{src.name}: weights are stored in {sorted(locs)}")
    missing = [loc for loc in locs if not (src.parent / loc).exists()]
    if missing:
        sys.exit(f"cannot find {missing} next to {src.name} (in {src.parent}). Copy it there with exactly this "
                 f"name - it is the file written next to the .onnx when the model was exported.")

    model = onnx.load(str(src))                              # reads the external data too
    out = Path(args.out).resolve() if args.out else src
    if out == src:
        shutil.copy(src, src.with_suffix(".onnx.bak"))
    tmp_out = out.with_name(out.name + ".tmp")
    onnx.save_model(model, str(tmp_out), save_as_external_data=False)
    if external_locations(tmp_out):
        sys.exit("still references external data - unexpected; is the model larger than 2 GB?")

    # verify: the new file alone, in an empty folder, gives exactly the original outputs
    import onnxruntime as ort
    with tempfile.TemporaryDirectory() as d:
        alone = Path(d) / "model.onnx"
        shutil.copy(tmp_out, alone)
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        new = ort.InferenceSession(str(alone), so, providers=["CPUExecutionProvider"])
        old = ort.InferenceSession(str(src), so, providers=["CPUExecutionProvider"])
        rng = np.random.default_rng(0)
        worst = 0.0
        for _ in range(20):
            feeds = random_feeds(old, rng)
            for a, b in zip(old.run(None, feeds), new.run(None, feeds)):
                worst = max(worst, float(np.abs(a - b).max()))
    if worst != 0.0:
        sys.exit(f"outputs differ by {worst:.3g} - not writing the result")
    tmp_out.replace(out)
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB), self-contained; outputs identical to the original")
    if out == src:
        print(f"original kept as {src.with_suffix('.onnx.bak').name}; the data file {sorted(locs)} is no longer needed")


if __name__ == "__main__":
    main()
