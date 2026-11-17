"""Export of a trained GRU model to a one-row-per-call ONNX graph, with a numerical check against PyTorch.

    python -m wnn.export_onnx --weights gru.pt --out model.onnx
"""
import argparse

import numpy as np
import torch

from .models import GRUStep, RecurrentNet


def export(weights, out, hidden=128, check_rows=500):
    net = RecurrentNet("gru", 112, hidden); net.load_state_dict(torch.load(weights, map_location="cpu")); net.eval()
    z = torch.zeros(1, 1, hidden)
    torch.onnx.export(GRUStep(net), (torch.zeros(1, 1, 112), z, z.clone()), out,
                      input_names=["features", "hidden_0", "hidden_1"], output_names=["prediction", "hidden_0_out", "hidden_1_out"],
                      opset_version=17, dynamo=False)
    import onnxruntime as ort
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).standard_normal((check_rows, 112)).astype(np.float32)
    h0 = np.zeros((1, 1, hidden), np.float32); h1 = h0.copy(); got = []
    for t in range(check_rows):
        p, h0, h1 = sess.run(None, {"features": x[t][None, None], "hidden_0": h0, "hidden_1": h1}); got.append(p[0, 0])
    with torch.no_grad():
        ref = net(torch.from_numpy(x[None]))[0][0].numpy()
    err = float(np.abs(np.array(got) - ref).max())
    print(f"max |ONNX - PyTorch| over {check_rows} steps: {err:.2e}")
    return err


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--weights", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); export(a.weights, a.out)
