import queue
import threading
import traceback
from collections import deque

import numpy as np

from .dsp import NoiseGate, PitchShifter, SpectralDenoiser, db_to_lin
from . import devices


class Engine:
    """mic -> [ganancia -> denoise -> gate -> voz] -> (+ soundboard) -> cable virtual (+ monitor)."""

    def __init__(self, cfg, soundboard):
        self.cfg, self.sb = cfg, soundboard
        sr = cfg.sample_rate
        self.gate = NoiseGate(sr)
        self.denoiser = SpectralDenoiser()
        self.pitch = PitchShifter(sr)
        self.rvc = None
        self.rvc_loading = False
        self.rvc_status = ""   # "" | "loading" | "ready" | "error: <msg>"
        self.in_q = queue.Queue(maxsize=16)
        self.out_q = deque(maxlen=8)
        self.mon_q = deque(maxlen=8)
        self.muted = False
        self.running = False
        self.level_in = self.level_out = 0.0
        self.last_error = ""
        self._streams = []
        self._thread = None
        self.mon_enabled = False

    # ---- procesamiento (puro, testeable sin placa de audio) ----
    def process_block(self, x):
        c = self.cfg
        x = x.astype(np.float32, copy=False) * db_to_lin(c.mic_gain_db)
        self.level_in = max(self.level_in, float(np.max(np.abs(x))))
        if c.denoise_enabled:
            x = self.denoiser.process(x)
        if c.gate_enabled:
            self.gate.threshold = db_to_lin(c.gate_threshold_db)
            x = self.gate.process(x)
        if c.voice_mode == "pitch":
            self.pitch.set_semitones(c.pitch_semitones)
            x = self.pitch.process(x)
        elif c.voice_mode == "rvc" and self.rvc is not None:
            x = self.rvc.process(x)
        return x

    def _worker(self):
        while self.running:
            try:
                x = self.in_q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                y = self.process_block(x)
            except Exception:
                self.last_error = traceback.format_exc()
                y = np.zeros_like(x)
            self.out_q.append(y)

    # ---- callbacks de audio ----
    def _in_cb(self, indata, frames, t, status):
        try:
            self.in_q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    @staticmethod
    def _fit(a, n):
        if len(a) == n:
            return a
        return np.pad(a, (0, n - len(a))) if len(a) < n else a[:n]

    def _out_cb(self, outdata, frames, t, status):
        try:
            voice = self._fit(self.out_q.popleft(), frames)
        except IndexError:
            voice = np.zeros(frames, np.float32)
        if self.muted:
            voice = np.zeros(frames, np.float32)
        self.sb.volume = self.cfg.sounds_volume
        snd = self.sb.mix(frames)
        mix = np.clip(voice + snd, -1.0, 1.0)
        self.level_out = max(self.level_out, float(np.max(np.abs(mix))))
        outdata[:] = mix[:, None]
        if self.mon_enabled:
            self.mon_q.append(voice * float(self.cfg.monitor_voice) + snd * float(self.cfg.monitor_sounds))

    def _mon_cb(self, outdata, frames, t, status):
        try:
            m = self._fit(self.mon_q.popleft(), frames)
        except IndexError:
            m = np.zeros(frames, np.float32)
        outdata[:] = m[:, None]

    # ---- ciclo de vida ----
    def start(self, in_name, out_name, mon_name=""):
        import sounddevice as sd
        if self.running:
            return
        c = self.cfg
        i_in = devices.find_by_name("input", in_name)
        i_out = devices.find_by_name("output", out_name)
        kw = dict(samplerate=c.sample_rate, blocksize=c.block_size, dtype="float32", latency="low")
        try:
            self._streams = [
                sd.InputStream(device=i_in, channels=1, callback=self._in_cb,
                               extra_settings=devices.extra_settings(i_in), **kw),
                sd.OutputStream(device=i_out, channels=devices.out_channels(i_out), callback=self._out_cb,
                                extra_settings=devices.extra_settings(i_out), **kw),
            ]
            self.mon_enabled = False
            if mon_name:
                i_mon = devices.find_by_name("output", mon_name)
                self._streams.append(sd.OutputStream(
                    device=i_mon, channels=devices.out_channels(i_mon), callback=self._mon_cb,
                    extra_settings=devices.extra_settings(i_mon), **kw))
                self.mon_enabled = True
            self.running = True
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()
            for s in self._streams:
                s.start()
        except Exception:
            self.stop()
            raise

    def load_rvc(self, model_path, index_path, pitch_shift=0, index_rate=0.66):
        """Carga (o recarga) el backend RVC en un hilo para no bloquear la UI."""
        from . import rvc_backend
        self.rvc_loading = True
        self.rvc_status = "loading"
        self.rvc = None

        def _load():
            try:
                backend = rvc_backend.build_backend(
                    model_path, index_path, self.cfg.sample_rate,
                    pitch_shift=pitch_shift, index_rate=index_rate)
                self.rvc = backend
                self.rvc_status = "ready"
            except Exception:
                import traceback
                self.rvc = None
                self.rvc_status = "error: " + traceback.format_exc()
            finally:
                self.rvc_loading = False

        threading.Thread(target=_load, daemon=True).start()

    def unload_rvc(self):
        self.rvc = None
        self.rvc_status = ""
        self.rvc_loading = False

    def stop(self):
        self.running = False
        for s in self._streams:
            try:
                s.stop()
                s.close()
            except Exception:
                pass
        self._streams = []
        self.in_q = queue.Queue(maxsize=16)
        self.out_q.clear()
        self.mon_q.clear()
