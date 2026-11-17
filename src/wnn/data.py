"""Data splits, training-set selection and conversion of the parquet files to .npy (numpy + pyarrow)."""
import numpy as np

N_TRAIN, N_VALID, N_HOLDOUT = 10607, 1873, 300
META = ("seq_ix", "step_in_seq", "need_prediction", "is_scored", "t0", "t1")


def holdout_ids(n_valid=N_VALID, k=N_HOLDOUT, seed=1):
    """Row groups of valid.parquet kept out of training for local evaluation ("valid300")."""
    return np.sort(np.random.default_rng(seed).choice(n_valid, k, replace=False))


def valid_train_ids(n_valid=N_VALID, k=N_HOLDOUT, seed=1):
    """The remaining valid row groups (1573), used for training."""
    return np.setdiff1d(np.arange(n_valid), holdout_ids(n_valid, k, seed))


def select_active(rate, family, n=5000):
    """Indices of the `n` most active sequences, chosen inside each family in proportion to its share.

    rate: activity (share of steps with a price event) of each sequence; family: its family 0..3.
    Training on active sequences was the strongest transferable gain in the late stage of the project.
    """
    rate, family = np.asarray(rate), np.asarray(family)
    picks = []
    for f in range(4):
        cand = np.where(family == f)[0]
        k = round(n * float(np.mean(family == f)))
        picks.append(cand[np.argsort(-rate[cand], kind="stable")][:k])
    return np.sort(np.concatenate(picks))


def family_weights(family, mix):
    """Per-sequence loss weights that re-weight the training family mix to `mix` (mean weight = 1)."""
    family = np.asarray(family)
    share = np.bincount(family, minlength=4) / len(family)
    w = (np.asarray(mix) / np.maximum(share, 1e-12))[family]
    return (w / w.mean()).astype(np.float32)


def parquet_to_npy(parquet_path, out_prefix, rows, dtype=np.float16, with_mask=False):
    """Writes features (n, 20000, 112) as a memory-mapped .npy, targets (and the is_scored mask) as .npy."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(parquet_path)
    feat = [c for c in pf.schema_arrow.names if c not in META]
    n_steps = pf.metadata.row_group(0).num_rows
    X = np.lib.format.open_memmap(f"{out_prefix}_X.npy", "w+", dtype, (len(rows), n_steps, len(feat)))
    Y = np.zeros((len(rows), n_steps, 2), np.float32)
    M = np.zeros((len(rows), n_steps), bool) if with_mask else None
    for k, i in enumerate(rows):
        tb = pf.read_row_group(int(i), columns=feat + ["t0", "t1"] + (["is_scored"] if with_mask else []))
        X[k] = np.column_stack([tb[c].to_numpy() for c in feat]).astype(dtype)
        Y[k] = np.column_stack([tb["t0"].to_numpy(), tb["t1"].to_numpy()])
        if with_mask:
            M[k] = tb["is_scored"].to_numpy()
    X.flush()
    np.save(f"{out_prefix}_Y.npy", Y)
    if with_mask:
        np.save(f"{out_prefix}_M.npy", M)
