"""Export native recurrent models. Matrix storage is uint16 half bits or float32."""
import copy
import numpy as np
import torch

def export(models, path, half=None, coefficients=None, cross_mix=0., ev_mu=None, ev_sd=None):
    # Preserve trained LSTM precision by default. Half rounding can accumulate
    # through long cell-state trajectories; a quantized selftest cannot detect it.
    # Explicit half=True remains available for separately validated exports.
    if half is None:
        half = not any(model.kind == 'lstm' for model in models)
    flat, mats, meta, mods = [], [], [], []
    foff = moff = soff = 0
    clones = []
    for model in models:
        model = copy.deepcopy(model).cpu().eval()
        typ = 3 if model.kind == 'lstm' else (2 if model.kind == 'gru_ev' else 0)
        mods.append([typ, len(meta), soff, model.hidden])
        soff += (4 if typ == 3 else 2)*model.hidden
        params = [getattr(model.rnn, f'{name}_l{layer}') for layer in range(2) for name in ('weight_ih','weight_hh','bias_ih','bias_hh')]
        params += [model.head.weight, model.head.bias]
        for p in params:
            a = p.detach().numpy()
            if a.ndim == 2:
                if half:
                    a = a.astype(np.float16).astype(np.float32)
                    with torch.no_grad(): p.copy_(torch.from_numpy(a))
                a = np.ascontiguousarray(a.T)
                meta.append([moff, *a.shape]); moff += a.size
                mats.append(a.ravel().astype(np.float16).view(np.uint16) if half else a.ravel())
            else:
                meta.append([foff, a.size, 1]); foff += a.size; flat.append(a.ravel())
        clones.append(model)
    coef = np.full((len(models),2), 1/len(models), np.float32) if coefficients is None else np.asarray(coefficients,np.float32)
    x = np.random.default_rng(17).normal(size=(8,112)).astype(np.float32)
    y = np.zeros((8,2),np.float32)
    from data import event_features
    with torch.no_grad():
        for k,m in enumerate(clones):
            xx = x if m.kind != 'gru_ev' else np.concatenate([x,(event_features(x)-ev_mu)/ev_sd],axis=1).astype(np.float32)
            y += m(torch.from_numpy(xx[None]))[0][0].numpy()*coef[k]
    np.savez(path, flat=np.concatenate(flat), mats=np.concatenate(mats), meta=np.array(meta,np.int64), mods=np.array(mods,np.int64), coef=coef, const=np.zeros(2,np.float32), selftest_x=x,selftest_y=y,cross_mix=np.float32(cross_mix),ev_mu=np.zeros(8) if ev_mu is None else ev_mu,ev_sd=np.ones(8) if ev_sd is None else ev_sd)
    return clones
