"""Shared constants for the Wunder Fund RNN challenge (Alpha Connectome)."""
from __future__ import annotations

SEQ_LEN = 20_000          # rows per sequence
WARMUP = 99               # steps 0..98 need no prediction
N_FEATURES = 112
TEST_ROWS = 39_400_000    # current test set size (FAQ)
TIME_LIMIT_S = 3600       # 60 minutes for the whole test set
CLIP = 2.0                # metric clips targets and predictions to [-2, 2]


def _group(inst: str, grp: str, n: int) -> list[str]:
    return [f"{inst}_{grp}{i}" for i in range(n)]


# Exact order from the Data overview (NOT alphabetical!).
FEATURES: list[str] = []
for _inst in ("i0", "i1"):
    FEATURES += _group(_inst, "p", 22)   # 0-10 bid, 11-21 ask
    FEATURES += _group(_inst, "v", 22)   # 0-10 bid, 11-21 ask
    FEATURES += _group(_inst, "dp", 4)   # trade price
    FEATURES += _group(_inst, "dv", 4)   # trade volume
FEATURES += [f"a{i}" for i in range(8)]
assert len(FEATURES) == N_FEATURES

TARGETS = ["t0", "t1"]

# Index helpers into the 112-dim state vector.
IDX = {}
for k, inst in enumerate(("i0", "i1")):
    base = 52 * k
    IDX[f"{inst}_bid_p"] = list(range(base + 0, base + 11))
    IDX[f"{inst}_ask_p"] = list(range(base + 11, base + 22))
    IDX[f"{inst}_bid_v"] = list(range(base + 22, base + 33))
    IDX[f"{inst}_ask_v"] = list(range(base + 33, base + 44))
    IDX[f"{inst}_dp"] = list(range(base + 44, base + 48))
    IDX[f"{inst}_dv"] = list(range(base + 48, base + 52))
IDX["extra"] = list(range(104, 112))

# Sanity: index helpers match the names
assert FEATURES[IDX["i1_ask_v"][0]] == "i1_v11"
assert FEATURES[IDX["i0_dv"][-1]] == "i0_dv3"
assert FEATURES[IDX["extra"][0]] == "a0"
