"""Build ONE submission from a linear blend of several recurrent models (one ONNX graph).

    python build_ensemble.py --valid-dir /content/data/valid --out $W/submissions/ens_v1 \
        --model "ft_swa=$W/runs/ft_baseline_full/epoch_2.pt+...epoch_5.pt@0.5" \
        --model "gru_full=$W/runs/gru_full/best.pt@0.5"

Model spec: NAME=PATH[+PATH...][@WEIGHT]. Supported: our checkpoints of arch "baseline"
(organizers' architecture, e.g. the fine-tuned one; several epochs joined by '+' are
weight-averaged) and "gru" (our GRU with deltas), plus the organizers' baseline.onnx.

How the blend is defined (the same as in ensemble_eval.py, so the validation number carries over):
    z_k = (p_k - mean_k) / std_k        per model and target, stats on the scored validation rows
    B   = sum_k w_k z_k ;  final = 0.6 * B / std(B)
which is linear: final = sum_k a_k p_k + c. The a_k and c are baked into the graph.

What the script does
  1. predicts the validation set with every model and computes a_k, c; prints the blend's WP;
  2. --engine numba (default): writes weights.npz + a solution.py whose GRU steps are compiled by
     numba (the scorer image pins numba==0.67.0) - two models then cost less than one model did
     in numpy/onnxruntime. --weights fp16 (default) stores the weight matrices in float16 (half
     the cache footprint; arithmetic stays float32): the script predicts the validation set again
     with the rounded matrices and prints that score next to the float32 one;
     --engine onnx: one single-step ONNX graph with all models (slower: ~1.5x our accepted GRU);
  3. runs the written solution step by step on real validation rows and checks it reproduces torch
     (with the same rounded matrices), then writes the zip.
Run check_submission.py on the zip afterwards (the notebook does).
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import warnings
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import SEQ_LEN, WARMUP
from ensemble_eval import load_any
from metric import WPAccumulator
from model import CLIP_IN, BaselineGRU, GRUNet
from train import predict_sequences

HERE = Path(__file__).resolve().parent
TARGET_STD = 0.6


def gru_net_step(m: GRUNet, state, prev, h):
    """GRUNet.forward for one step, written without torch.diff (exports cleanly)."""
    f = torch.cat([state, state - prev], -1)
    f = ((f - m.mu) / m.sd).clamp(-CLIP_IN, CLIP_IN)
    z = F.silu(m.inp(f))
    o, h = m.gru(z, h)
    return m.head2(F.silu(m.head1(torch.cat([o, z], -1)))), h


class EnsembleStep(nn.Module):
    """state, prev: (1, 1, 112); h_k: the k-th model's state (layers, 1, hidden) -> blended (1, 1, 2), new states."""

    def __init__(self, models, coef, const):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.register_buffer("coef", torch.as_tensor(np.asarray(coef), dtype=torch.float32))    # (K, 2)
        self.register_buffer("const", torch.as_tensor(np.asarray(const), dtype=torch.float32))  # (2,)

    def forward(self, state, prev, *hs):
        acc, new = self.const.view(1, 1, 2), []
        for k, (m, h) in enumerate(zip(self.models, hs)):
            if isinstance(m, BaselineGRU):
                p, hn = m(state, h)
            else:
                p, hn = gru_net_step(m, state, prev, h)
            acc = acc + p * self.coef[k].view(1, 1, 2)
            new.append(hn)
        return (acc, *new)


def state_shape(m):
    if isinstance(m, BaselineGRU):
        return (2, 1, m.gru0.hidden_size)
    return (m.gru.num_layers, 1, m.gru.hidden_size)


def wp(Y, P, M):
    acc = WPAccumulator(2)
    acc.update(Y, P, M)
    return acc.score()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid-dir", required=True)
    ap.add_argument("--model", action="append", required=True, help="NAME=PATH[+PATH...][@WEIGHT]")
    ap.add_argument("--out", required=True, help="submission folder to create (the zip goes next to it)")
    ap.add_argument("--valid-seqs", type=int, default=None)
    ap.add_argument("--engine", choices=["numba", "onnx"], default="numba")
    ap.add_argument("--weights", choices=["fp16", "fp32"], default="fp16",
                    help="numba engine: storage of the weight matrices (fp16 = half the cache footprint)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = Path(args.valid_dir)
    Xv = np.load(d / "features.npy", mmap_mode="r")
    n = Xv.shape[0] if args.valid_seqs is None else min(args.valid_seqs, Xv.shape[0])
    Y = np.asarray(np.load(d / "targets.npy", mmap_mode="r")[:n])
    M = np.asarray(np.load(d / "is_scored.npy", mmap_mode="r")[:n]) & (np.arange(SEQ_LEN) >= WARMUP)

    names, models, weights, preds = [], [], [], []
    for spec in args.model:
        name, rest = spec.split("=", 1)
        path, w = (rest.rsplit("@", 1) + ["1"])[:2] if "@" in rest else (rest, "1")
        kind, m, how = load_any(path, device)
        if kind != "ours" or not isinstance(m, (BaselineGRU, GRUNet)):
            raise SystemExit(f"{name}: only the baseline architecture and our GRU can go into the ensemble ({how})")
        P = predict_sequences(m, Xv, np.arange(n), device)
        print(f"{name:12s} weight {float(w):.2f}  {how:35s} WP alone (std 0.6): "
              f"{wp(Y, (P - P[M].mean(0)) / P[M].std(0) * TARGET_STD, M):.5f}")
        names.append(name); models.append(m.cpu().eval()); weights.append(float(w)); preds.append(P)

    w = np.array(weights) / sum(weights)
    mean = np.stack([P[M].mean(0) for P in preds])                    # (K, 2)
    std = np.stack([P[M].std(0) for P in preds])
    B = sum(wk * (P[M] - mk) / sk for wk, P, mk, sk in zip(w, preds, mean, std))
    scale = TARGET_STD / B.std(0)                                      # (2,)
    coef = (w[:, None] / std) * scale[None, :]                         # final = sum_k coef_k * p_k + const
    const = -(coef * mean).sum(0)
    final = sum(c * P for c, P in zip(coef, preds)) + const
    score = wp(Y, final, M)
    print(f"\nblend {' + '.join(f'{wk:.2f}*{nm}' for wk, nm in zip(w, names))}: WP on the validation mask {score:.5f}")

    # The submission stores the matrices in float16: its exact reference is the same models with
    # rounded matrices. Their validation score shows what the rounding costs.
    ref_final = final
    fp16 = args.engine == "numba" and args.weights == "fp16"
    if fp16:
        models = [round_matrices_fp16(m) for m in models]
        ref_final = sum(c * predict_sequences(m.to(device), Xv, np.arange(n), device) for c, m in zip(coef, models)) + const
        models = [m.cpu().eval() for m in models]
        score16 = wp(Y, ref_final, M)
        print(f"same blend with float16 weight matrices (what the submission computes): WP {score16:.5f} "
              f"({score16 - score:+.5f})")

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    X0 = np.asarray(Xv[0, :3000], np.float32)
    if args.engine == "onnx":
        files = write_onnx(models, coef, const, out, X0, final)
    else:
        files = write_numba(models, coef, const, out, fp16, X0[:8], ref_final[0, :8])
    worst = check_solution(out, X0, ref_final[0])
    print(f"{args.engine} solution step by step vs torch on 3000 validation rows: max |diff| = {worst:.2e}")
    if worst > 1e-3:
        raise SystemExit("the submission does not reproduce the torch blend - not writing the zip")
    (out / "ensemble.json").write_text(json.dumps(
        {"engine": args.engine, "weights_dtype": "fp16" if fp16 else "fp32", "models": names, "weights": w.tolist(),
         "coef": coef.tolist(), "const": const.tolist(), "valid_wp": score,
         "valid_wp_submission": wp(Y, ref_final, M) if fp16 else score, "valid_seqs": int(n)}, indent=1))
    zp = out.with_suffix(".zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(out / f, f)
    print(f"wrote {zp} ({zp.stat().st_size / 1e6:.2f} MB): {', '.join(files)}")


def round_matrices_fp16(m):
    """A copy of the model whose weight matrices (all 2-D parameters) are rounded to float16 and back."""
    m = copy.deepcopy(m).cpu()
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 2:
                p.copy_(p.half().float())
    return m.eval()


def numba_tensors(m):
    """The tensors of one model in the order solution_template_ensemble_numba.py reads them:
    (array, is_matrix). Matrices go in (in, out) layout, vectors as (1, n)."""
    t = lambda a: (a.detach().cpu().numpy().T, True)                    # noqa: E731
    v = lambda a: (a.detach().cpu().numpy().reshape(1, -1), False)      # noqa: E731
    if isinstance(m, BaselineGRU):
        g0, g1 = m.gru0, m.gru1
        return 0, g0.hidden_size, [t(g0.weight_ih_l0), t(g0.weight_hh_l0), v(g0.bias_ih_l0), v(g0.bias_hh_l0),
                                   t(g1.weight_ih_l0), t(g1.weight_hh_l0), v(g1.bias_ih_l0), v(g1.bias_hh_l0),
                                   t(m.head.weight), v(m.head.bias)]
    g = m.gru
    if g.num_layers != 2:
        raise SystemExit("numba engine: our GRU must have 2 layers")
    return 1, g.hidden_size, [v(m.mu), v(m.sd), t(m.inp.weight), v(m.inp.bias),
                              t(g.weight_ih_l0), t(g.weight_hh_l0), v(g.bias_ih_l0), v(g.bias_hh_l0),
                              t(g.weight_ih_l1), t(g.weight_hh_l1), v(g.bias_ih_l1), v(g.bias_hh_l1),
                              t(m.head1.weight), v(m.head1.bias), t(m.head2.weight), v(m.head2.bias)]


def write_numba(models, coef, const, out: Path, fp16: bool, selftest_x, selftest_y):
    vecs, mats, meta, mods, vo, mo, so = [], [], [], [], 0, 0, 0
    for m in models:
        typ, H, ts = numba_tensors(m)
        mods.append((typ, len(meta), so, H))
        so += 2 * H
        for a, is_mat in ts:
            a = np.ascontiguousarray(a, np.float32)
            buf, off, align = (mats, mo, 32) if is_mat else (vecs, vo, 16)    # 64-byte aligned tensors
            pad = (-off) % align
            buf.append(np.zeros(pad, np.float32))
            meta.append((off + pad, a.shape[0], a.shape[1]))
            buf.append(a.ravel())
            if is_mat:
                mo = off + pad + a.size
            else:
                vo = off + pad + a.size
    M = np.concatenate(mats)
    if fp16:
        H16 = M.astype(np.float16)
        if not np.array_equal(H16.astype(np.float32), M):
            raise SystemExit("internal error: matrices are not float16-rounded")
        M = H16.view(np.uint16)
    np.savez(out / "weights.npz", flat=np.concatenate(vecs), mats=M, meta=np.array(meta, np.int64),
             mods=np.array(mods, np.int64), coef=np.asarray(coef, np.float32), const=np.asarray(const, np.float32),
             selftest_x=np.asarray(selftest_x, np.float32), selftest_y=np.asarray(selftest_y, np.float32))
    shutil.copy(HERE / "solution_template_ensemble_numba.py", out / "solution.py")
    return ["solution.py", "weights.npz"]


def write_onnx(models, coef, const, out: Path, X0, final):
    ens = EnsembleStep(models, coef, const).eval()
    shapes = [state_shape(m) for m in models]
    ex = (torch.zeros(1, 1, 112), torch.zeros(1, 1, 112), *[torch.zeros(s) for s in shapes])
    in_names = ["state", "prev"] + [f"h_{k}" for k in range(len(models))]
    out_names = ["pred"] + [f"h_{k}_out" for k in range(len(models))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(ens, ex, str(out / "ensemble.onnx"), dynamo=False, opset_version=17,
                          do_constant_folding=True, input_names=in_names, output_names=out_names)
    shutil.copy(HERE / "solution_template_ensemble.py", out / "solution.py")
    return ["solution.py", "ensemble.onnx"]


def check_solution(folder: Path, X0, ref):
    """Run the written solution.py row by row (as the platform does) and compare with the torch blend."""
    import dataclasses
    import importlib.util

    @dataclasses.dataclass
    class DP:
        seq_ix: int
        step_in_seq: int
        need_prediction: bool
        state: np.ndarray

    spec = importlib.util.spec_from_file_location(f"ens_solution_{folder.name}", folder / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pm = mod.PredictionModel()
    worst = 0.0
    for t in range(len(X0)):
        p = pm.predict(DP(0, t, t >= WARMUP, X0[t]))
        if p is not None:
            worst = max(worst, float(np.abs(np.asarray(p) - ref[t]).max()))
    return worst


if __name__ == "__main__":
    main()
