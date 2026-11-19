import os, sys, time, subprocess
import numpy as np, torch, torch.nn as nn, pyarrow.parquet as pq

dev = 'cuda'
L, NF, WARM, SAT, U3 = 20000, 112, 99, 99, 0.6569
ROOT = '/content/wnn_connectome_starterpack/datasets'
CKPT_DIR = '/content/drive/MyDrive/wnn' if os.path.isdir('/content/drive/MyDrive/wnn') else '/content'
UN = np.array([0.1254, 0.2413, 0.4685, 0.6569])
VALMIX = np.array([0.2125, 0.1853, 0.2301, 0.3721])
post = np.zeros(L, bool); post[WARM:] = True

# ---------- модели ----------
class Net(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.gru = nn.GRU(NF, h, num_layers=2, batch_first=True)
        self.head = nn.Linear(h, 2)
    def forward(self, x, h=None):
        y, h = self.gru(x, h)
        return self.head(y), h

class Net2(nn.Module):                      # то же, но с произвольным числом входов
    def __init__(self, nin, h=128):
        super().__init__()
        self.gru = nn.GRU(nin, h, num_layers=2, batch_first=True)
        self.head = nn.Linear(h, 2)
    def forward(self, x, h=None):
        y, h = self.gru(x, h)
        return self.head(y), h

class Step(nn.Module):                      # обёртка для экспорта в ONNX (интерфейс как у baseline)
    def __init__(self, net): super().__init__(); self.net = net
    def forward(self, features, hidden_0, hidden_1):
        o, h = self.net(features, torch.cat([hidden_0, hidden_1], 0))
        return o, h[0:1], h[1:2]
StepH = Step

# ---------- данные ----------
def build(src, dst, n, seed, with_mask):
    pf = pq.ParquetFile(f'{ROOT}/{src}.parquet')
    meta = {'seq_ix', 'step_in_seq', 'need_prediction', 'is_scored', 't0', 't1'}
    feat = [c for c in pf.schema_arrow.names if c not in meta]
    assert len(feat) == NF
    idx = np.sort(np.random.default_rng(seed).choice(pf.num_row_groups, n, replace=False))
    X = np.lib.format.open_memmap(f'/content/data/{dst}_X.npy', 'w+', np.float32, (n, L, NF))
    Y = np.zeros((n, L, 2), np.float32); M = np.zeros((n, L), bool) if with_mask else None
    t = time.time()
    for k, i in enumerate(idx):
        tb = pf.read_row_group(int(i), columns=feat + ['t0', 't1'] + (['is_scored'] if with_mask else []))
        X[k] = np.column_stack([tb[c].to_numpy() for c in feat])
        Y[k] = np.column_stack([tb['t0'].to_numpy(), tb['t1'].to_numpy()])
        if with_mask: M[k] = tb['is_scored'].to_numpy()
        if k % 500 == 0: print(dst, k, f'{time.time() - t:.0f} с', flush=True)
    X.flush(); np.save(f'/content/data/{dst}_Y.npy', Y)
    if with_mask: np.save(f'/content/data/{dst}_M.npy', M)

def fam_of(Y, verbose=True):                # семейство = ближайшая единица к наименьшему ненулевому |t0|
    a = np.abs(np.clip(Y[..., 0], -2, 2)).astype(np.float64)
    mn = np.where(a > 0, a, 9.0).min(1)
    f = np.abs(mn[:, None] - UN[None]).argmin(1)
    if verbose: print('макс. отклонение от единицы:', np.abs(mn - UN[f]).max().round(4), '| классы:', np.bincount(f, minlength=4))
    return f

# ---------- обучение и метрика ----------
def loss_fn(p, y, mask):
    y = y.clamp(-2, 2)
    return (y.abs() * (p - y) ** 2)[mask].mean()

@torch.no_grad()
def evaluate(model, X, Y, M, bs=50, ch=2000):
    model.eval()
    S = {k: torch.zeros(2, 6, dtype=torch.float64, device=dev) for k in ('все строки', 'is_scored')}
    pm = torch.from_numpy(post).to(dev)
    for s in range(0, len(X), bs):
        x = torch.from_numpy(np.ascontiguousarray(X[s:s + bs])).to(dev)
        y = torch.from_numpy(np.ascontiguousarray(Y[s:s + bs])).to(dev).clamp(-2, 2)
        m_sc = torch.from_numpy(np.ascontiguousarray(M[s:s + bs])).to(dev)
        h, outs = None, []
        for t in range(0, L, ch):
            o, h = model(x[:, t:t + ch], h); outs.append(o)
        p = torch.cat(outs, 1).clamp(-2, 2)
        for name, mask in (('все строки', pm[None].expand_as(m_sc)), ('is_scored', m_sc & pm[None])):
            for j in (0, 1):
                yy, pp = y[..., j][mask].double(), p[..., j][mask].double(); a = yy.abs()
                S[name][j] += torch.stack([a.sum(), (a*yy).sum(), (a*pp).sum(), (a*yy*yy).sum(), (a*pp*pp).sum(), (a*yy*pp).sum()])
    out = {}
    for k, T in S.items():
        sw = T[:, 0]; my, mp = T[:, 1] / sw, T[:, 2] / sw
        cov = T[:, 5] / sw - my * mp
        den = (T[:, 3] / sw - my ** 2) * (T[:, 4] / sw - mp ** 2)
        c = torch.where(den > 1e-16, cov / den.clamp_min(1e-16).sqrt(), torch.zeros_like(cov)).clamp(-1, 1)
        out[k] = (c.mean().item(), c.cpu().numpy().round(3))
    model.train(); return out

def train(model, epochs, bs=32, ch=1000, lr=2e-3, tag='gru'):     # использует глобальные Xtr, Ytr, Xva, Yva, Mva
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    steps_total = epochs * (len(Xtr) // bs) * (L // ch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps_total, pct_start=0.05)
    rng = np.random.default_rng(0); best = -1
    for ep in range(epochs):
        t0 = time.time(); order = rng.permutation(len(Xtr)); run = []
        for g in range(len(Xtr) // bs):
            idx = np.sort(order[g * bs:(g + 1) * bs])
            x = torch.from_numpy(np.ascontiguousarray(Xtr[idx])).to(dev); y = torch.from_numpy(Ytr[idx]).to(dev)
            h = None
            for t in range(0, L, ch):
                o, h = model(x[:, t:t + ch], h); h = h.detach()
                mask = (torch.arange(t, t + ch, device=dev) >= WARM)[None].expand(len(idx), -1)
                loss = loss_fn(o, y[:, t:t + ch], mask)
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
                run.append(loss.item())
        r = evaluate(model, Xva, Yva, Mva)
        print(f'эпоха {ep + 1}/{epochs} | {time.time() - t0:.0f} с | loss {np.mean(run):.4f} | '
              f'valid все строки {r["все строки"][0]:.3f} {r["все строки"][1]} | is_scored {r["is_scored"][0]:.3f} {r["is_scored"][1]}', flush=True)
        if r['is_scored'][0] > best:
            best = r['is_scored'][0]; torch.save(model.state_dict(), f'{CKPT_DIR}/{tag}_best.pt')
    return model

# ---------- анализ по семействам и валидационной смеси ----------
@torch.no_grad()
def predict(model, X, idx, ch=2000, bs=49):
    model.eval(); out = []
    for s in range(0, len(idx), bs):
        x = torch.from_numpy(np.ascontiguousarray(X[idx[s:s + bs]])).to(dev); h = None; o_ = []
        for t in range(0, L, ch):
            o, h = model(x[:, t:t + ch], h); o_.append(o)
        out.append(torch.cat(o_, 1).clamp(-2, 2).cpu().numpy())
    return np.concatenate(out)

def sums_np(P, Y, M):
    post_m = np.broadcast_to(post, M.shape); out = {}
    for name, m in (('все строки', post_m), ('is_scored', M & post_m)):
        T = np.zeros((2, 6))
        for j in (0, 1):
            y = np.clip(Y[..., j][m], -2, 2).astype(np.float64); p = np.clip(P[..., j][m], -2, 2).astype(np.float64); a = np.abs(y)
            T[j] = [a.sum(), (a*y).sum(), (a*p).sum(), (a*y*y).sum(), (a*p*p).sum(), (a*y*p).sum()]
        out[name] = T
    return out

def corr(T):
    sw = T[:, 0]; my, mp = T[:, 1] / sw, T[:, 2] / sw
    cov = T[:, 5] / sw - my * mp
    den = (T[:, 3] / sw - my ** 2) * (T[:, 4] / sw - mp ** 2)
    return float(np.mean(np.where(den > 1e-16, cov / np.sqrt(np.maximum(den, 1e-16)), 0.0)))

def ens_mix(models):                        # скор по валидационной смеси семейств; нужны Xva, Yva, Mva, fam_v, have
    m = VALMIX / have; Sm = {}
    for f in range(4):
        ix = np.where(fam_v == f)[0]
        P = np.mean([predict(mod, Xva, ix) for mod in models], axis=0)
        Sm[f] = sums_np(P, Yva[ix], Mva[ix])
    return {k: round(corr(sum(m[f] * Sm[f][k] for f in range(4))), 3) for k in ('все строки', 'is_scored')}
mix_score = lambda model: ens_mix([model])

def code3(y):                               # код k в единицах семейства 4; насыщение = ±99
    k = np.round(y / U3).astype(np.int32); k[y >= 2 - 1e-6] = SAT; k[y <= -2 + 1e-6] = -SAT; return k

def wc(y, p, m):                            # взвешенная корреляция по маске m
    y, p = y[m].astype(np.float64), p[m].astype(np.float64); w = np.abs(y); sw = w.sum()
    my, mp = (w * y).sum() / sw, (w * p).sum() / sw
    cov, vy, vp = (w * (y - my) * (p - mp)).sum(), (w * (y - my) ** 2).sum(), (w * (p - mp) ** 2).sum()
    return cov / np.sqrt(vy * vp) if vp > 1e-12 else 0.0
