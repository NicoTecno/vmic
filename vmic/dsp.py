"""Procesamiento de audio en numpy puro (mono, float32)."""
import numpy as np


def db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


class NoiseGate:
    """Puerta de ruido con attack/release y hold."""

    def __init__(self, sr, threshold_db=-50.0, attack_ms=5.0, release_ms=150.0, hold_ms=120.0):
        self.sr = sr
        self.threshold = db_to_lin(threshold_db)
        self.attack, self.release = attack_ms / 1000, release_ms / 1000
        self.hold_samples = int(sr * hold_ms / 1000)
        self.hold = 0
        self.gain = 0.0

    def process(self, x):
        n = len(x)
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        is_open = rms > self.threshold
        if is_open:
            self.hold = self.hold_samples
        elif self.hold > 0:
            self.hold -= n
            is_open = True
        target = 1.0 if is_open else 0.0
        tau = self.attack if target > self.gain else self.release
        coef = np.exp(-n / (self.sr * tau))
        new_gain = target + (self.gain - target) * coef
        g = np.linspace(self.gain, new_gain, n, endpoint=False, dtype=np.float32)
        self.gain = float(new_gain)
        return x * g


class SpectralDenoiser:
    """Supresor de ruido espectral liviano (resta espectral + estadística de mínimos).

    El piso de ruido por banda se estima como el mínimo del espectro suavizado en ~0.8 s, así que
    funciona bien con ruido estacionario (ventilador, PC, aire) y habla con pausas. Sonidos
    sostenidos por más de ~1 s (ej. cantar una nota larga) pueden atenuarse.
    Es una primera versión en CPU. Para calidad tipo NVIDIA Broadcast, la Fase 2 integra
    RNNoise / DeepFilterNet (también locales).
    """

    def __init__(self, n=512, over=1.3, floor=0.1, bias=1.7, window_s=0.8, sr=48000):
        self.n, self.h = n, n // 2
        self.over, self.floor, self.bias = over, floor, bias
        k = np.arange(n)
        self.win = np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * k / n)).astype(np.float32)
        self.inbuf = np.zeros(0, np.float32)
        self.ola = np.zeros(n, np.float32)
        self.K = max(8, int(window_s * sr / self.h))
        self.ring = None
        self.ri = 0
        self.smooth = None
        self.gprev = None
        self.pending = np.zeros(n + self.h, np.float32)  # latencia ~16 ms

    def process(self, x):
        self.inbuf = np.concatenate([self.inbuf, x.astype(np.float32, copy=False)])
        out = []
        while len(self.inbuf) >= self.n:
            sp = np.fft.rfft(self.inbuf[: self.n] * self.win)
            mag = np.abs(sp)
            if self.ring is None:
                self.smooth = mag.copy()
                self.ring = np.tile(self.smooth, (self.K, 1))
                self.gprev = np.ones_like(mag)
            self.smooth = 0.8 * self.smooth + 0.2 * mag
            self.ring[self.ri] = self.smooth
            self.ri = (self.ri + 1) % self.K
            noise = self.bias * self.ring.min(axis=0)
            gain = np.maximum(1.0 - self.over * noise / (mag + 1e-9), self.floor)
            gain = 0.5 * gain + 0.5 * self.gprev   # suaviza en el tiempo (menos "ruido musical")
            self.gprev = gain
            y = np.fft.irfft(sp * gain, self.n).astype(np.float32) * self.win
            self.ola += y
            out.append(self.ola[: self.h].copy())
            self.ola = np.concatenate([self.ola[self.h:], np.zeros(self.h, np.float32)])
            self.inbuf = self.inbuf[self.h:]
        if out:
            self.pending = np.concatenate([self.pending] + out)
        return _take(self, len(x))


class PitchShifter:
    """Cambio de tono en tiempo real (línea de retardo con dos lecturas y crossfade)."""

    def __init__(self, sr, semitones=0.0, window_ms=50):
        self.W = int(sr * window_ms / 1000)
        self.hist = np.zeros(self.W + 4, np.float32)
        self.phase = 0.0
        self.ratio = 1.0
        self.set_semitones(semitones)

    def set_semitones(self, st):
        self.ratio = float(2.0 ** (st / 12.0))

    def process(self, x):
        n = len(x)
        buf = np.concatenate([self.hist, x.astype(np.float32, copy=False)])
        self.hist = buf[-len(self.hist):]
        if abs(self.ratio - 1.0) < 1e-4:
            return x
        base = len(buf) - n
        i = np.arange(n)
        step = (1.0 - self.ratio) / self.W
        ph = (self.phase + i * step) % 1.0
        self.phase = (self.phase + n * step) % 1.0
        out = np.zeros(n, np.float32)
        for off in (0.0, 0.5):
            p = (ph + off) % 1.0
            pos = base + i - p * self.W
            i0 = np.floor(pos).astype(np.int64)
            frac = (pos - i0).astype(np.float32)
            i1 = np.minimum(i0 + 1, len(buf) - 1)
            out += (buf[i0] * (1 - frac) + buf[i1] * frac) * (np.sin(np.pi * p) ** 2).astype(np.float32)
        return out


def _take(obj, n):
    """Devuelve exactamente n muestras de obj.pending (rellena con ceros si falta)."""
    if len(obj.pending) < n:
        y = np.concatenate([np.zeros(n - len(obj.pending), np.float32), obj.pending])
        obj.pending = np.zeros(0, np.float32)
        return y
    y, obj.pending = obj.pending[:n], obj.pending[n:]
    return y
