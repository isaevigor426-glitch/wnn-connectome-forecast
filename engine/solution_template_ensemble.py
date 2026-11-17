"""Blend of several recurrent models in ONE ONNX graph (written by build_ensemble.py).

Per row: the graph gets the 112 features, the previous row (for the delta features of our GRU)
and every model's hidden state, and returns the blended [t0, t1] (blend weights, centring and
scale are inside the graph) plus the new hidden states. One session.run per row.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402


class PredictionModel:
    def __init__(self):
        o = ort.SessionOptions()
        o.intra_op_num_threads = 1
        o.inter_op_num_threads = 1
        o.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        o.add_session_config_entry("session.intra_op.allow_spinning", "0")
        o.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.sess = ort.InferenceSession(str(Path(__file__).resolve().parent / "ensemble.onnx"),
                                         sess_options=o, providers=["CPUExecutionProvider"])
        ins = self.sess.get_inputs()
        self.h_names = [i.name for i in ins[2:]]
        self.h_shapes = [tuple(i.shape) for i in ins[2:]]
        self.x = np.zeros((1, 1, 112), np.float32)
        self.prev = np.zeros((1, 1, 112), np.float32)
        self.seq_ix = None
        self.h = None
        self.first = True

    def _reset(self):
        self.h = [np.zeros(s, np.float32) for s in self.h_shapes]
        self.first = True

    def predict(self, data_point):
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            self._reset()
        s = np.asarray(data_point.state, dtype=np.float32)
        self.x[0, 0] = s
        if self.first:                       # delta = 0 on the first row, as in training
            self.prev[0, 0] = s
            self.first = False
        feeds = {"state": self.x, "prev": self.prev}
        for name, h in zip(self.h_names, self.h):
            feeds[name] = h
        out = self.sess.run(None, feeds)
        self.h = out[1:]
        self.prev[0, 0] = s
        if not data_point.need_prediction:
            return None
        p = out[0].reshape(2).astype(np.float32)
        if not np.isfinite(p).all():
            p = np.nan_to_num(p, nan=0.0, posinf=2.0, neginf=-2.0)
        return p
