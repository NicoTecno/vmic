import threading
import numpy as np

MAX_SECONDS = 120
MAX_VOICES = 16


class Soundboard:
    """Biblioteca de sonidos + mezcla polifónica (permite 'spamear' el mismo sonido)."""

    def __init__(self, sr):
        self.sr = sr
        self.volume = 0.8
        self.library = {}   # name -> (data, gain)
        self.voices = []    # [data, pos, gain]
        self.lock = threading.Lock()

    def _load(self, path):
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=True, frames=MAX_SECONDS * 192000)
        mono = data.mean(axis=1)
        if sr != self.sr and len(mono) > 1:
            new_len = int(len(mono) * self.sr / sr)
            mono = np.interp(np.linspace(0, len(mono) - 1, new_len), np.arange(len(mono)), mono)
        return mono.astype(np.float32)[: MAX_SECONDS * self.sr]

    def add(self, name, path, gain=1.0):
        self.library[name] = (self._load(path), float(gain))

    def remove(self, name):
        self.library.pop(name, None)

    def play(self, name):
        item = self.library.get(name)
        if item is None:
            return
        with self.lock:
            if len(self.voices) >= MAX_VOICES:
                self.voices.pop(0)
            self.voices.append([item[0], 0, item[1]])

    def stop_all(self):
        with self.lock:
            self.voices.clear()

    def mix(self, n):
        out = np.zeros(n, np.float32)
        with self.lock:
            alive = []
            for v in self.voices:
                seg = v[0][v[1]: v[1] + n]
                out[: len(seg)] += seg * v[2]
                v[1] += n
                if v[1] < len(v[0]):
                    alive.append(v)
            self.voices = alive
        return out * self.volume
