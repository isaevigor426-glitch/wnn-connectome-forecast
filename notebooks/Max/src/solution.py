"""Blend of recurrent models, one row at a time, compiled with numba (written by build_ensemble.py).

The scorer image has numba (its requirements.txt pins numba==0.67.0). One compiled call per row
runs every model and the blend, so there is no per-operation overhead of numpy/onnxruntime:
  * matrix-vector products are "axpy" loops over transposed weights, 8 input rows at a time - the
    inner loop runs over contiguous memory and is vectorised (AVX2/AVX-512);
  * sigmoid = 1 / (1 + exp(-x)) with a Cephes-style exp (relative error < 1e-7) built from the
    float's exponent bits, tanh = a rational approximation (abs error < 4e-7, as in Eigen); no
    libm calls, so the gate loops vectorise too. Sigmoid saturates to exactly 1.0 like float32
    torch does - a gate that should keep memory for thousands of steps does not leak;
  * error_model="numpy": no Python zero-division checks inside the loops (they block vectorising).
  * weight matrices are stored as float16 ("mats", uint16 bits; vectors stay float32 in "flat") and
    widened to float32 on load (vcvtph2ps): two models then take 0.8 MB, which stays in a 1 MB L2
    cache, instead of 1.6 MB streamed from L3 on every row. Arithmetic is float32. build_ensemble.py
    measures the effect of the rounding on the validation score (~1e-5).
Compilation happens once, in __init__ (a few seconds), followed by a self-test: a few rows with
outputs computed by torch at build time must be reproduced, otherwise the matrices are widened to
float32 storage (same values) and the test is repeated; if that fails too, __init__ raises.

Model types ("mods" rows: type, first row in meta, state offset, hidden size):
  0 = organizers' baseline architecture: GRU(112->H) -> GRU(H->H) -> Linear(H->2), raw features
  2 = type 0 on [raw features, 8 event features] (120 inputs), see _evfeat
  1 = our GRU: [x, x - prev] -> (f - mu) / sd, clip +-10 -> Linear+SiLU -> 2-layer GRU -> MLP head
Output = const + sum_k coef[k] * p_k   (blend weights, centring and scale are in coef/const)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from llvmlite import ir  # noqa: E402
from numba import njit, types  # noqa: E402
from numba.extending import intrinsic  # noqa: E402

_FM = {"contract"}            # allow fused multiply-add only; NaN/inf semantics stay exact
_KW = dict(cache=False, fastmath=_FM, error_model="numpy")
_f = np.float32
_ZERO, _HALF, _ONE, _CLIP, _TC = _f(0.0), _f(0.5), _f(1.0), _f(10.0), _f(7.90531110763549805)
_LOG2E, _LN2HI, _LN2LO, _XMAX = _f(1.44269504088896341), _f(0.693359375), _f(-2.12194440e-4), _f(80.0)
_E0, _E1, _E2, _E3, _E4, _E5 = (_f(1.9875691500e-4), _f(1.3981999507e-3), _f(8.3334519073e-3),
                                _f(4.1665795894e-2), _f(1.6666665459e-1), _f(5.0000001201e-1))
_A1, _A3, _A5, _A7 = _f(4.89352455891786e-03), _f(6.37261928875436e-04), _f(1.48572235717979e-05), _f(5.12229709037114e-08)
_A9, _A11, _A13 = _f(-8.60467152213735e-11), _f(2.00018790482477e-13), _f(-2.76076847742355e-16)
_B0, _B2, _B4, _B6 = _f(4.89352518554385e-03), _f(2.26843463243900e-03), _f(1.18534705686654e-04), _f(1.19825839466702e-06)


@intrinsic
def _pow2i(typingctx, k):
    """2.0 ** k as float32 for an integer k in [-126, 127], written straight into the exponent bits."""
    if not isinstance(k, types.Integer):
        return None

    def codegen(context, builder, sig, args):
        v = context.cast(builder, args[0], sig.args[0], types.int64)
        e = builder.shl(builder.add(v, ir.Constant(ir.IntType(64), 127)), ir.Constant(ir.IntType(64), 23))
        return builder.bitcast(builder.trunc(e, ir.IntType(32)), ir.FloatType())
    return types.float32(k), codegen


@intrinsic
def _w2f(typingctx, w):
    """A weight as float32: float32 as is, uint16 = IEEE half bits (vcvtph2ps when vectorised)."""
    if w == types.float32:
        return types.float32(w), lambda context, builder, sig, args: args[0]
    if w == types.uint16:
        def codegen(context, builder, sig, args):
            return builder.fpext(builder.bitcast(args[0], ir.HalfType()), ir.FloatType())
        return types.float32(w), codegen
    return None


@njit(inline="always", **_KW)
def _exp(v):
    x = min(max(v, -_XMAX), _XMAX)
    fx = np.floor(x * _LOG2E + _HALF)
    r = x - fx * _LN2HI
    r = r - fx * _LN2LO
    y = ((((_E0 * r + _E1) * r + _E2) * r + _E3) * r + _E4) * r + _E5
    y = y * (r * r) + r + _ONE
    return y * _pow2i(np.int64(fx))


@njit(inline="always", **_KW)
def _sigmoid(v):
    return _ONE / (_ONE + _exp(-v))


@njit(inline="always", **_KW)
def _tanh(v):
    x = min(max(v, -_TC), _TC)
    x2 = x * x
    p = ((((((_A13 * x2 + _A11) * x2 + _A9) * x2 + _A7) * x2 + _A5) * x2 + _A3) * x2 + _A1) * x
    q = ((_B6 * x2 + _B4) * x2 + _B2) * x2 + _B0
    return p / q


@njit(**_KW)
def _gemv(x, WT, b, out):
    """out = b + x @ WT, WT: (n, m) C-contiguous. 8 rows per pass over out."""
    n = x.shape[0]
    m = out.shape[0]
    for j in range(m):
        out[j] = b[j]
    i = 0
    while i + 8 <= n:
        x0, x1, x2, x3 = x[i], x[i + 1], x[i + 2], x[i + 3]
        x4, x5, x6, x7 = x[i + 4], x[i + 5], x[i + 6], x[i + 7]
        for j in range(m):
            out[j] += ((_w2f(WT[i, j]) * x0 + _w2f(WT[i + 1, j]) * x1) + (_w2f(WT[i + 2, j]) * x2 + _w2f(WT[i + 3, j]) * x3)) + \
                      ((_w2f(WT[i + 4, j]) * x4 + _w2f(WT[i + 5, j]) * x5) + (_w2f(WT[i + 6, j]) * x6 + _w2f(WT[i + 7, j]) * x7))
        i += 8
    while i < n:
        xi = x[i]
        for j in range(m):
            out[j] += _w2f(WT[i, j]) * xi
        i += 1


@njit(**_KW)
def _gru(x, h, WiT, WhT, bi, bh, gi, gh):
    """PyTorch GRU cell (gate order r, z, n); updates h in place."""
    H = h.shape[0]
    _gemv(x, WiT, bi, gi)
    _gemv(h, WhT, bh, gh)
    for j in range(H):
        r = _sigmoid(gi[j] + gh[j])
        z = _sigmoid(gi[H + j] + gh[H + j])
        n = _tanh(gi[2 * H + j] + r * gh[2 * H + j])
        h[j] = n + z * (h[j] - n)


@njit(**_KW)
def _lstm(x, h, c, WiT, WhT, bi, bh, gi, gh):
    """PyTorch i,f,g,o gates; h and c updated in place after both GEMVs."""
    H = h.shape[0]
    _gemv(x, WiT, bi, gi)
    _gemv(h, WhT, bh, gh)
    for j in range(H):
        i = _sigmoid(gi[j] + gh[j])
        f = _sigmoid(gi[H+j] + gh[H+j])
        g = _tanh(gi[2*H+j] + gh[2*H+j])
        o = _sigmoid(gi[3*H+j] + gh[3*H+j])
        c[j] = f*c[j] + i*g
        h[j] = o*_tanh(c[j])


@njit(**_KW)
def _mat(mats, meta, r):
    o, n, m = meta[r, 0], meta[r, 1], meta[r, 2]
    return mats[o:o + n * m].reshape((n, m))


@njit(**_KW)
def _vec(flat, meta, r):
    o = meta[r, 0]
    return flat[o:o + meta[r, 1] * meta[r, 2]]


_KS = np.array([1, 2, 3, 5, 10], dtype=np.int64)


@njit(**_KW)
def _evfeat(x, prev, first, ei, mbuf, fl, xe, emu, esd):
    """Causal event features for type-2 models (price-change events of columns 0-21); writes [x, (f - mu) / sd] into xe.
    ei: step, events so far, step of last event, events in last 50 steps, events in last 200 steps."""
    t = ei[0]
    cnt = ei[1]
    ch = False
    s = 0.0
    if not first:
        for i in range(22):
            d = x[i] - prev[i]
            if d != _ZERO:
                ch = True
            s += d
    flag = 0
    if ch:
        cnt += 1
        mbuf[cnt % 10] = s / 22.0
        ei[2] = t
        flag = 1
    j = t % 200
    ei[4] += flag - fl[j]
    ei[3] += flag - fl[(t + 150) % 200]
    fl[j] = flag
    for q in range(5):
        k = _KS[q]
        m = k if k < cnt else cnt
        acc = 0.0
        for jj in range(m):
            acc += mbuf[(cnt - jj) % 10]
        xe[112 + q] = (acc - emu[q]) / esd[q]
    v5 = math.log1p(t - ei[2]) if cnt > 0 else math.log1p(20000.0)
    xe[117] = (v5 - emu[5]) / esd[5]
    xe[118] = (ei[3] - emu[6]) / esd[6]
    xe[119] = (ei[4] - emu[7]) / esd[7]
    for i in range(112):
        xe[i] = x[i]
    ei[0] = t + 1
    ei[1] = cnt


@njit(**_KW)
def _step(x, prev, first, flat, mats, meta, mods, S, gi, gh, f, zb, cat, hid, o2, coef, const, out, ei, mbuf, fl, xe, emu, esd):
    """One row for every model; writes the blend into out and copies x into prev."""
    _evfeat(x, prev, first, ei, mbuf, fl, xe, emu, esd)
    out[0] = const[0]
    out[1] = const[1]
    for k in range(mods.shape[0]):
        typ, r, so, H = mods[k, 0], mods[k, 1], mods[k, 2], mods[k, 3]
        h0 = S[so:so + H]
        h1 = S[so + H:so + 2 * H]
        g_i = gi[:3 * H]
        g_h = gh[:3 * H]
        if typ == 0 or typ == 2:
            if typ == 0:
                _gru(x, h0, _mat(mats, meta, r), _mat(mats, meta, r + 1), _vec(flat, meta, r + 2), _vec(flat, meta, r + 3), g_i, g_h)
            else:
                _gru(xe, h0, _mat(mats, meta, r), _mat(mats, meta, r + 1), _vec(flat, meta, r + 2), _vec(flat, meta, r + 3), g_i, g_h)
            _gru(h0, h1, _mat(mats, meta, r + 4), _mat(mats, meta, r + 5), _vec(flat, meta, r + 6), _vec(flat, meta, r + 7), g_i, g_h)
            _gemv(h1, _mat(mats, meta, r + 8), _vec(flat, meta, r + 9), o2)
        elif typ == 3:
            # State layout: h0,c0,h1,c1. Ten metadata rows, same order as GRU.
            c0 = S[so+H:so+2*H]
            h1 = S[so+2*H:so+3*H]
            c1 = S[so+3*H:so+4*H]
            _lstm(x, h0, c0, _mat(mats, meta, r), _mat(mats, meta, r+1), _vec(flat, meta, r+2), _vec(flat, meta, r+3), gi[:4*H], gh[:4*H])
            _lstm(h0, h1, c1, _mat(mats, meta, r+4), _mat(mats, meta, r+5), _vec(flat, meta, r+6), _vec(flat, meta, r+7), gi[:4*H], gh[:4*H])
            _gemv(h1, _mat(mats, meta, r+8), _vec(flat, meta, r+9), o2)
        elif typ == 1:
            nx = x.shape[0]
            mu = _vec(flat, meta, r)
            sd = _vec(flat, meta, r + 1)
            for i in range(nx):
                f[i] = min(max((x[i] - mu[i]) / sd[i], -_CLIP), _CLIP)
                d = _ZERO if first else x[i] - prev[i]
                f[nx + i] = min(max((d - mu[nx + i]) / sd[nx + i], -_CLIP), _CLIP)
            P = meta[r + 2, 2]
            z = zb[:P]
            _gemv(f[:2 * nx], _mat(mats, meta, r + 2), _vec(flat, meta, r + 3), z)
            for j in range(P):
                z[j] = z[j] * _sigmoid(z[j])                                   # SiLU
            _gru(z, h0, _mat(mats, meta, r + 4), _mat(mats, meta, r + 5), _vec(flat, meta, r + 6), _vec(flat, meta, r + 7), g_i, g_h)
            _gru(h0, h1, _mat(mats, meta, r + 8), _mat(mats, meta, r + 9), _vec(flat, meta, r + 10), _vec(flat, meta, r + 11), g_i, g_h)
            c = cat[:H + P]
            for j in range(H):
                c[j] = h1[j]
            for j in range(P):
                c[H + j] = z[j]
            D = meta[r + 12, 2]
            hd = hid[:D]
            _gemv(c, _mat(mats, meta, r + 12), _vec(flat, meta, r + 13), hd)
            for j in range(D):
                hd[j] = hd[j] * _sigmoid(hd[j])
            _gemv(hd, _mat(mats, meta, r + 14), _vec(flat, meta, r + 15), o2)
        else:
            raise ValueError("Unknown model type")
        out[0] += coef[k, 0] * o2[0]
        out[1] += coef[k, 1] * o2[1]
    for i in range(x.shape[0]):
        prev[i] = x[i]
    for t in range(2):
        if not math.isfinite(out[t]):
            out[t] = 0.0 if math.isnan(out[t]) else (2.0 if out[t] > 0 else -2.0)


class PredictionModel:
    def __init__(self, weights_path=None):
        w = np.load(weights_path or Path(__file__).resolve().parent / "weights.npz", allow_pickle=False)
        self.cross_mix = float(w["cross_mix"]) if "cross_mix" in w.files else 0.07
        self.flat = np.ascontiguousarray(w["flat"], dtype=np.float32)
        self.mats = np.ascontiguousarray(w["mats"])                 # uint16 = float16 bits, or float32
        self.meta = np.ascontiguousarray(w["meta"], dtype=np.int64)
        self.mods = np.ascontiguousarray(w["mods"], dtype=np.int64)
        self.coef = np.ascontiguousarray(w["coef"], dtype=np.float32)
        self.const = np.ascontiguousarray(w["const"], dtype=np.float32)
        if not np.isin(self.mods[:, 0], [0,1,2,3]).all():
            raise ValueError("Unknown model type")
        ends = self.mods[:,2] + np.where(self.mods[:,0] == 3, 4, 2)*self.mods[:,3]
        spans = sorted(zip(self.mods[:,2], ends))
        if any(a < 0 or b <= a for a,b in spans) or any(spans[i][1] > spans[i+1][0] for i in range(len(spans)-1)):
            raise ValueError("Invalid or overlapping state offsets")
        Hm = int(self.mods[:, 3].max())
        self.S = np.zeros(int(ends.max()), np.float32)
        self.gi, self.gh = np.zeros(4 * Hm, np.float32), np.zeros(4 * Hm, np.float32)
        self.f = np.zeros(2 * 112, np.float32)
        big = int(self.meta[:, 1:].max())
        self.zb, self.cat, self.hid = np.zeros(big, np.float32), np.zeros(Hm + big, np.float32), np.zeros(big, np.float32)
        self.o2, self.out = np.zeros(2, np.float32), np.zeros(2, np.float32)
        self.x = np.zeros(112, np.float32)          # every row is copied here: one compiled signature
        self.prev = np.zeros(112, np.float32)
        self.ei, self.mbuf, self.fl = np.zeros(5, np.int64), np.zeros(10, np.float64), np.zeros(200, np.int64)
        self.xe = np.zeros(120, np.float32)
        self.emu = np.ascontiguousarray(w["ev_mu"], dtype=np.float64) if "ev_mu" in w.files else np.zeros(8)
        self.esd = np.ascontiguousarray(w["ev_sd"], dtype=np.float64) if "ev_sd" in w.files else np.ones(8)
        self._selftest(np.asarray(w["selftest_x"], np.float32), np.asarray(w["selftest_y"], np.float32))

    def _ev_reset(self):
        self.ei[:] = 0
        self.mbuf[:] = 0.0
        self.fl[:] = 0

    def _reset(self):
        self.S[:] = 0.0
        self._ev_reset()
        self.prev[:] = 0.0
        self.first = True
        self.seq_ix = None

    def _run(self):
        _step(self.x, self.prev, self.first, self.flat, self.mats, self.meta, self.mods, self.S, self.gi, self.gh,
              self.f, self.zb, self.cat, self.hid, self.o2, self.coef, self.const, self.out,
              self.ei, self.mbuf, self.fl, self.xe, self.emu, self.esd)

    def _probe(self, X, Y):
        """Compile (first call) and run the self-test rows from a fresh state; max |out - expected|."""
        if not np.isfinite(X).all() or not np.isfinite(Y).all():
            raise ValueError('Nonfinite selftest')
        self._reset()
        err = 0.0
        for t in range(len(X)):
            self.x[:] = X[t]
            self._run()
            self.first = False
            err = max(err, float(np.abs(self.out - Y[t]).max()))
        return err

    def _selftest(self, X, Y, tol=1e-3):
        try:
            err, problem = self._probe(X, Y), None
        except Exception as e:  # noqa: BLE001  (e.g. a CPU without float16 conversion instructions)
            err, problem = float("inf"), e
        if not err <= tol and self.mats.dtype == np.uint16:
            self.mats = np.ascontiguousarray(self.mats.view(np.float16).astype(np.float32))   # same values
            err, problem = self._probe(X, Y), None
        if problem is not None:
            raise problem
        if not err <= tol:
            raise RuntimeError(f"self-test failed: compiled model differs from the reference by {err:.2e}")
        self._reset()

    def predict(self, data_point):
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            self.S[:] = 0.0
            self.first = True
            self._ev_reset()
        self.x[:] = data_point.state
        self._run()
        self.first = False
        if not data_point.need_prediction:
            return None
        p, q = self.out
        a = self.cross_mix
        return np.array([(1-a)*p-a*q, (1-a)*q-a*p], dtype=np.float32)


