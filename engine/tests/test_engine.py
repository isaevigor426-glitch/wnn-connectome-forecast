"""End-to-end on synthetic data (CPU): prepare -> train -> export -> harness."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import synth  # noqa: E402
from model import GRUNet  # noqa: E402
from train import weighted_pearson_loss  # noqa: E402
from metric import weighted_pearson  # noqa: E402


def run(*a):
    r = subprocess.run([sys.executable, *a], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    return r.stdout


def test_loss_equals_metric():
    rng = np.random.default_rng(0)
    y = np.clip(rng.standard_normal((5000, 2)), -2, 2)
    p = 0.4 * y + rng.standard_normal((5000, 2))
    w = np.abs(y)
    l = weighted_pearson_loss(torch.tensor(p), torch.tensor(y), torch.tensor(w)).item()
    ref = np.mean([weighted_pearson(y[:, k], p[:, k], w[:, k]) for k in range(2)])
    assert 1 - l == pytest.approx(ref, abs=1e-6)


def test_chunked_equals_full():
    m = GRUNet(16, 24, 2, 8).eval()
    x = torch.randn(2, 300, 112)
    with torch.no_grad():
        full, _ = m(x)
        h, prev, parts = None, None, []
        for s in range(0, 300, 70):
            p, h = m(x[:, s:s + 70], h, prev)
            prev = x[:, s + 69] if s + 70 <= 300 else None
            parts.append(p)
    assert torch.allclose(full, torch.cat(parts, 1), atol=1e-5)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("d")
    synth.write(str(d / "train.parquet"), 4, with_scored=False, seed=1)
    synth.write(str(d / "valid.parquet"), 2, with_scored=True, seed=2)
    run("prepare_data.py", "--src", str(d / "train.parquet"), "--out", str(d / "train"), "--dtype", "float16")
    run("prepare_data.py", "--src", str(d / "valid.parquet"), "--out", str(d / "valid"))
    return d


def test_train_export_eval(data, tmp_path):
    runs = tmp_path / "run"
    out = run("train.py", "--train-dir", str(data / "train"), "--valid-dir", str(data / "valid"),
              "--out", str(runs), "--epochs", "2", "--batch", "2", "--chunk", "2000",
              "--proj", "16", "--hidden", "24", "--head", "8", "--event-weight", "1.0", "--lr", "5e-3")
    assert "VALID" in out and (runs / "best.pt").exists()
    hist = json.loads((runs / "history.json").read_text())
    assert hist[-1]["WP_mask_best"] > 0.05, hist        # learns the planted signal
    # resume continues from the saved epoch
    out2 = run("train.py", "--train-dir", str(data / "train"), "--valid-dir", str(data / "valid"),
               "--out", str(runs), "--epochs", "3", "--batch", "2", "--chunk", "2000",
               "--proj", "16", "--hidden", "24", "--head", "8", "--resume")
    assert "resumed from epoch 1" in out2
    sub = tmp_path / "sub"
    out3 = run("export.py", "--ckpt", str(runs / "best.pt"), "--valid-dir", str(data / "valid"),
               "--out", str(sub), "--check-steps", "1500")
    assert "OK" in out3 and (tmp_path / "sub.zip").exists()
    rep = tmp_path / "eval.json"
    run("local_eval.py", "--solution-dir", str(sub), "--data", str(data / "valid.parquet"),
        "--check-determinism", "2", "--report", str(rep))
    r = json.loads(rep.read_text())
    assert sum(r["errors"].values()) == 0
    assert r["determinism_max_abs_diff"] == 0 and r["reset_check_max_abs_diff"] == 0
    # harness score ~ training-time validation score (same model, same data)
    best = max(h["WP_mask_best"] for h in json.loads((runs / "history.json").read_text()))
    assert r["WP_mask_abs_target"]["score"] == pytest.approx(best, abs=0.02)


ONNX = Path(os.environ.get("BASELINE_ONNX", "baseline.onnx"))   # the organizers' model from the starter pack


@pytest.mark.skipif(not ONNX.exists(), reason="baseline.onnx not available")
def test_finetune_from_onnx(data, tmp_path):
    runs = tmp_path / "ft"
    out = run("train.py", "--train-dir", str(data / "train"), "--valid-dir", str(data / "valid"),
              "--out", str(runs), "--arch", "baseline", "--init-onnx", str(ONNX),
              "--epochs", "1", "--batch", "2", "--chunk", "2000", "--lr", "2e-4")
    assert "initialised from" in out and "VALID (start = initial weights)" in out
    hist = json.loads((runs / "history.json").read_text())
    assert hist[0]["epoch"] == -1 and len(hist) == 2
    sub = tmp_path / "ftsub"
    out3 = run("export.py", "--ckpt", str(runs / "best.pt"), "--valid-dir", str(data / "valid"),
               "--out", str(sub), "--check-steps", "1200")
    assert "OK" in out3
    import zipfile
    assert set(zipfile.ZipFile(tmp_path / "ftsub.zip").namelist()) == {"solution.py", "weights.npz", "config.json"}
    rep = tmp_path / "e.json"
    run("local_eval.py", "--solution-dir", str(sub), "--data", str(data / "valid.parquet"),
        "--check-determinism", "2", "--report", str(rep))
    r = json.loads(rep.read_text())
    assert sum(r["errors"].values()) == 0 and r["reset_check_max_abs_diff"] == 0


# ----------------------------------------------------------------------------- Mamba
def test_ssd_scan_matches_sequential():
    from mamba import ssd_scan
    torch.manual_seed(0)
    b, L, h, p, n = 2, 200, 3, 4, 5
    X = torch.randn(b, L, h, p, dtype=torch.float64)
    A = -torch.rand(b, L, h, dtype=torch.float64) * 0.5
    B = torch.randn(b, L, h, n, dtype=torch.float64)
    C = torch.randn(b, L, h, n, dtype=torch.float64)
    h0 = torch.randn(b, h, p, n, dtype=torch.float64)
    Y, fs = ssd_scan(X, A, B, C, 50, h0)
    hs, ref = h0.clone(), []
    for t in range(L):
        hs = torch.exp(A[:, t])[..., None, None] * hs + X[:, t][..., None] * B[:, t][:, :, None, :]
        ref.append((hs * C[:, t][:, :, None, :]).sum(-1))
    assert torch.allclose(Y, torch.stack(ref, 1), atol=1e-10)
    assert torch.allclose(fs, hs, atol=1e-10)


@pytest.mark.parametrize("out_norm", [False, True])
def test_mamba_chunked_equals_full(out_norm):
    from mamba import MambaNet
    torch.manual_seed(0)
    m = MambaNet(d_model=32, n_layers=2, d_state=8, headdim=8, block_len=50, out_norm=out_norm).double().eval()
    x = torch.randn(2, 600, 112, dtype=torch.float64)
    with torch.no_grad():
        full, _ = m(x)
        for ch in (100, 70):            # 70: not a multiple of block_len -> padded scan
            h, prev, parts = None, None, []
            for s in range(0, 600, ch):
                p, h = m(x[:, s:s + ch], h, prev)
                prev = x[:, min(s + ch, 600) - 1]
                parts.append(p)
            assert torch.allclose(torch.cat(parts, 1), full, atol=1e-9)
        # the carried state matters (otherwise the test above would be vacuous)
        p2, _ = m(x[:, 300:], None, x[:, 299])
        assert (p2 - full[:, 300:]).abs().max() > 1e-3


def test_mamba_train_export_eval(data, tmp_path):
    runs = tmp_path / "mamba"
    common = ["--train-dir", str(data / "train"), "--valid-dir", str(data / "valid"), "--out", str(runs),
              "--arch", "mamba", "--d-model", "16", "--d-state", "8", "--headdim", "8", "--head", "16",
              "--batch", "2", "--chunk", "2000", "--lr", "5e-3", "--wd", "0.01"]
    out = run("train.py", *common, "--epochs", "2", "--event-weight", "1.0")
    assert "VALID" in out
    hist = json.loads((runs / "history.json").read_text())
    assert hist[-1]["WP_mask_best"] > 0.05, hist          # learns the planted signal
    out2 = run("train.py", *common, "--epochs", "3", "--resume")
    assert "resumed from epoch 1" in out2
    sub = tmp_path / "msub"
    out3 = run("export.py", "--ckpt", str(runs / "best.pt"), "--valid-dir", str(data / "valid"),
               "--out", str(sub), "--check-steps", "1500")
    assert "OK" in out3 and (tmp_path / "msub.zip").exists()
    rep = tmp_path / "me.json"
    run("local_eval.py", "--solution-dir", str(sub), "--data", str(data / "valid.parquet"),
        "--check-determinism", "2", "--report", str(rep))
    r = json.loads(rep.read_text())
    assert sum(r["errors"].values()) == 0
    assert r["determinism_max_abs_diff"] == 0 and r["reset_check_max_abs_diff"] == 0
    best = max(h["WP_mask_best"] for h in json.loads((runs / "history.json").read_text()))
    assert r["WP_mask_abs_target"]["score"] == pytest.approx(best, abs=0.02)


# ----------------------------------------------------------------------------- parquet streaming
def test_parquet_source_equals_npy(data, tmp_path):
    from train import NpySeqs, ParquetSeqs
    run("prepare_data.py", "--src", str(data / "train.parquet"), "--out", str(tmp_path / "f32"))   # float32 copy
    a = NpySeqs(str(tmp_path / "f32"))
    b = ParquetSeqs(str(data / "train.parquet"), workers=2)
    assert a.n == b.n == 4
    for idx in ([0], [3, 1], [2, 0, 3, 1]):
        xa, ya = a.fetch(idx)
        xb, yb = b.fetch(idx)
        assert xa.dtype == xb.dtype == np.float32 and xa.shape == xb.shape
        assert np.array_equal(xa.astype(np.float32), xb) and np.array_equal(ya, yb)


def test_parquet_reader_error_reaches_training_loop(tmp_path):
    from train import Prefetcher

    def bad_fetch(idx):
        raise ValueError("broken row group")
    with pytest.raises(ValueError, match="broken row group"):
        for _ in Prefetcher(bad_fetch, [[0], [1]]):
            pass


def test_train_from_parquet_and_resume(data, tmp_path):
    runs = tmp_path / "pq"
    common = ["--train-parquet", str(data / "train.parquet"), "--valid-dir", str(data / "valid"), "--out", str(runs),
              "--batch", "2", "--chunk", "2000", "--proj", "16", "--hidden", "24", "--head", "8", "--lr", "5e-3"]
    out = run("train.py", *common, "--epochs", "2", "--keep-epochs")
    assert "parquet, read on the fly" in out and "data-wait" in out and "VALID" in out
    assert (runs / "epoch_0.pt").exists() and (runs / "epoch_1.pt").exists()
    hist = json.loads((runs / "history.json").read_text())
    assert hist[-1]["WP_mask_best"] > 0.05, hist
    out2 = run("train.py", *common, "--epochs", "3", "--resume")
    assert "resumed from epoch 1" in out2


@pytest.mark.skipif(not ONNX.exists(), reason="baseline.onnx not available")
def test_finetune_from_parquet(data, tmp_path):
    runs = tmp_path / "ftpq"
    out = run("train.py", "--train-parquet", str(data / "train.parquet"), "--valid-dir", str(data / "valid"),
              "--out", str(runs), "--arch", "baseline", "--init-onnx", str(ONNX), "--keep-epochs",
              "--epochs", "1", "--batch", "2", "--chunk", "2000", "--lr", "2e-4")
    assert "VALID (start = initial weights)" in out and (runs / "last.pt").exists() and (runs / "epoch_0.pt").exists()
    sub = tmp_path / "ftpq_sub"
    assert "OK" in run("export.py", "--ckpt", str(runs / "last.pt"), "--valid-dir", str(data / "valid"),
                       "--out", str(sub), "--check-steps", "1000")


def test_train_needs_exactly_one_source(data, tmp_path):
    r = subprocess.run([sys.executable, "train.py", "--valid-dir", str(data / "valid"), "--out", str(tmp_path / "x")],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "exactly one of --train-dir / --train-parquet" in (r.stderr + r.stdout)


# ----------------------------------------------------------------------------- ensemble evaluation
def test_ensemble_eval(data, tmp_path):
    from ensemble_eval import TeamGRU
    runs = tmp_path / "g"
    run("train.py", "--train-dir", str(data / "train"), "--valid-dir", str(data / "valid"), "--out", str(runs),
        "--epochs", "2", "--batch", "2", "--chunk", "2000", "--proj", "16", "--hidden", "24", "--head", "8",
        "--lr", "5e-3", "--keep-epochs")
    team = TeamGRU(120, 16)
    torch.save({"model": team.state_dict(), "args": {"hidden": 16},
                "target_mean": np.zeros(2, np.float32), "target_std": np.ones(2, np.float32)}, tmp_path / "team.pt")
    models = ["--model", f"gru={runs / 'best.pt'}",
              "--model", f"swa={runs / 'epoch_0.pt'}+{runs / 'epoch_1.pt'}",
              "--model", f"team={tmp_path / 'team.pt'}"]
    names = {"gru", "swa", "team"}
    if ONNX.exists():
        models += ["--model", f"baseline={ONNX}"]
        names.add("baseline")
    rep = tmp_path / "ens.json"
    args = ["ensemble_eval.py", "--valid-dir", str(data / "valid"), "--out", str(tmp_path / "preds"), *models,
            "--report", str(rep)]
    out = run(*args)
    assert "SWA of 2" in out and "team GRU" in out
    r = json.loads(rep.read_text())
    assert set(r["singles"]) == names and "cv_blend" in r and len(r["fit_weights"]) == len(names)
    assert r["singles"]["gru"][0] > 0.05                      # the trained model finds the planted signal
    assert "loaded from" in run(*args)                        # cached predictions are reused


# ----------------------------------------------------------------------------- ensemble submission
@pytest.fixture(scope="module")
def ens_models(data, tmp_path_factory):
    """A tiny GRU and a tiny model of the organizers' architecture (2 kept epochs -> SWA)."""
    d = tmp_path_factory.mktemp("ens_models")
    common = ["--train-dir", str(data / "train"), "--valid-dir", str(data / "valid"), "--batch", "2",
              "--chunk", "2000", "--lr", "5e-3"]
    run("train.py", *common, "--out", str(d / "g"), "--epochs", "1", "--proj", "16", "--hidden", "24", "--head", "8",
        "--dropout", "0.2", "--input-noise", "0.05", "--wd", "1e-2")       # regularized: inference must not change
    run("train.py", *common, "--out", str(d / "b"), "--arch", "baseline", "--hidden", "24", "--epochs", "2",
        "--keep-epochs", "--dropout", "0.2", "--input-noise", "0.05")
    return [f"gru={d / 'g' / 'best.pt'}@0.5", f"base={d / 'b' / 'epoch_0.pt'}+{d / 'b' / 'epoch_1.pt'}@0.5"]


@pytest.mark.parametrize("engine,weights", [("numba", "fp16"), ("numba", "fp32"), ("onnx", "fp32")])
def test_build_ensemble(engine, weights, data, ens_models, tmp_path):
    import re
    import zipfile
    pytest.importorskip("numba" if engine == "numba" else "onnxruntime")
    sub = tmp_path / "ens"
    out = run("build_ensemble.py", "--valid-dir", str(data / "valid"), "--out", str(sub), "--engine", engine,
              "--weights", weights, *[a for m in ens_models for a in ("--model", m)])
    diff = float(re.search(r"max \|diff\| = ([0-9.e+-]+)", out).group(1))
    assert diff < 1e-4, out                                   # the submission reproduces torch
    files = {"numba": {"solution.py", "weights.npz"}, "onnx": {"solution.py", "ensemble.onnx"}}[engine]
    assert set(zipfile.ZipFile(tmp_path / "ens.zip").namelist()) == files
    blend = float(re.search(r"WP on the validation mask ([0-9.-]+)", out).group(1))
    if engine == "numba" and weights == "fp16":
        assert np.load(sub / "weights.npz")["mats"].dtype == np.uint16
        blend16 = float(re.search(r"float16 weight matrices .*: WP ([0-9.-]+)", out).group(1))
        assert blend16 == pytest.approx(blend, abs=2e-3)      # rounding the weights costs ~nothing
        blend = blend16                                       # ... and is what the submission computes
    rep = tmp_path / "e.json"
    run("local_eval.py", "--solution-dir", str(sub), "--data", str(data / "valid.parquet"),
        "--check-determinism", "2", "--report", str(rep))
    r = json.loads(rep.read_text())
    assert sum(r["errors"].values()) == 0
    assert r["determinism_max_abs_diff"] == 0 and r["reset_check_max_abs_diff"] == 0
    assert r["WP_mask_abs_target"]["score"] == pytest.approx(blend, abs=1e-3)   # harness == offline blend


@pytest.mark.parametrize("arch", ["gru", "baseline"])
def test_regularization_only_in_training(arch):
    from model import BaselineGRU
    torch.manual_seed(0)
    if arch == "gru":
        m = GRUNet(proj=8, hidden=12, layers=2, head=6, dropout=0.5, input_noise=0.1)
    else:
        m = BaselineGRU(12, dropout=0.5, input_noise=0.1)
    x = torch.randn(2, 30, 112)
    m.train()
    a, _ = m(x)
    b, _ = m(x)
    assert not torch.allclose(a, b)                     # dropout + noise are active while training
    m.eval()
    with torch.no_grad():
        c, _ = m(x)
        e, _ = m(x)
    assert torch.equal(c, e)                            # and switched off for inference
    groups = m.param_groups(1e-2)
    assert all(p.dim() >= 2 for p in groups[0]["params"]) and all(p.dim() < 2 for p in groups[1]["params"])


def test_extra_parquet_and_init_from(data, tmp_path):
    """Train on train.parquet + valid.parquet minus the sequences of a held-out subset, then
    continue from that checkpoint: the starting validation score equals the checkpoint's."""
    import re
    held = tmp_path / "held.parquet"
    run("make_subset.py", "--src", str(data / "valid.parquet"), "--out", str(held), "--n-seqs", "1")
    common = ["--train-parquet", str(data / "train.parquet"), "--extra-parquet", str(data / "valid.parquet"),
              "--exclude-seqs-from", str(held), "--valid-dir", str(data / "valid"), "--batch", "2",
              "--chunk", "2000", "--proj", "16", "--hidden", "24", "--head", "8"]
    out = run("train.py", *common, "--out", str(tmp_path / "a"), "--epochs", "1", "--lr", "5e-3")
    assert re.search(r"train seqs 5 from .* \+ 1 of 2 sequences of .*without the 1 that are in held.parquet", out), out
    src = torch.load(tmp_path / "a" / "best.pt", map_location="cpu", weights_only=False)
    out = run("train.py", *common, "--out", str(tmp_path / "b"), "--epochs", "1", "--lr", "1e-3",
              "--init-from", f"{tmp_path / 'a' / 'best.pt'}+{tmp_path / 'a' / 'best.pt'}")
    start = json.loads(re.search(r"VALID \(start = initial weights\) (\{.*\})", out).group(1))
    assert start["WP_mask_best"] == pytest.approx(src["best_score"], abs=1e-6)
