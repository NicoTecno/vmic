"""Backends de cambio de voz. Todos exponen process(x) -> x (mismo largo, mono float32)."""
import numpy as np
from .dsp import _take


class ChunkedConverter:
    """Adapta un modelo NO streaming (RVC, etc.) a tiempo real.

    Acumula chunks, le pasa audio con contexto previo al modelo (infer_fn) y
    empalma las salidas con crossfade. Latencia ≈ chunk_ms + xfade_ms + tiempo de inferencia.
    """

    def __init__(self, infer_fn, sr, chunk_ms=250, context_ms=200, xfade_ms=30):
        self.fn = infer_fn
        self.chunk = int(sr * chunk_ms / 1000)
        self.ctx = int(sr * context_ms / 1000)
        self.xf = int(sr * xfade_ms / 1000)
        assert self.chunk > self.xf
        self.inbuf = np.zeros(0, np.float32)
        self.hist = np.zeros(0, np.float32)
        self.tail = np.zeros(self.xf, np.float32)
        self.pending = np.zeros(self.chunk, np.float32)
        self.fade = np.linspace(0, 1, self.xf, dtype=np.float32)
        self.latency_ms = chunk_ms + xfade_ms

    def process(self, x):
        self.inbuf = np.concatenate([self.inbuf, x])
        need = self.chunk + self.xf
        while len(self.inbuf) >= self.chunk:
            new, self.inbuf = self.inbuf[: self.chunk], self.inbuf[self.chunk:]
            self.hist = np.concatenate([self.hist, new])[-(self.ctx + need):]
            seg = np.asarray(self.fn(self.hist), np.float32)[-need:]
            if len(seg) < need:
                seg = np.pad(seg, (need - len(seg), 0))
            head = seg[: self.xf] * self.fade + self.tail * (1 - self.fade)
            emit = np.concatenate([head, seg[self.xf: self.chunk]])
            self.tail = seg[self.chunk:]
            self.pending = np.concatenate([self.pending, emit])
        return _take(self, len(x))
