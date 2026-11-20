"""Backend de cambio de voz usando RVC (Retrieval-based Voice Conversion) en GPU local."""
import os
import tempfile
import threading

import numpy as np
import soundfile as sf


def available() -> bool:
    """Devuelve True si rvc-python está instalado y listo."""
    try:
        import rvc_python  # noqa
        return True
    except ImportError:
        return False


class RVCBackend:
    """Wrapper del modelo RVC adaptado a la interfaz process(x) del engine.

    La inferencia trabaja con archivos WAV temporales (en disco) porque la
    API de rvc-python es file-based. Los archivos se crean y eliminan en cada
    chunk (~300 ms), así que el overhead es mínimo comparado con la GPU.
    """

    def __init__(self, model_path: str, index_path: str, sr: int,
                 pitch_shift: int = 0, index_rate: float = 0.66):
        from rvc_python.infer import RVCInference
        import torch

        self.sr = sr
        self._lock = threading.Lock()
        device = "cuda:0" if torch.cuda.is_available() else "cpu:0"

        # Inicializa RVC (descarga HuBERT + RMVPE la primera vez, ~400 MB)
        self._rvc = RVCInference(device=device)

        # Carga el modelo .pth (versión v2)
        idx = index_path if index_path and os.path.isfile(index_path) else ""
        self._rvc.load_model(model_path, version="v2", index_path=idx)

        # Parámetros de inferencia
        self._rvc.set_params(
            f0up_key=pitch_shift,     # semitones de desplazamiento de tono
            f0method="rmvpe",         # mejor calidad para voz
            index_rate=index_rate,    # influencia del índice FAISS
            resample_sr=sr,           # re-muestrea al SR del engine
            protect=0.33,             # protege consonantes sin voz
            filter_radius=3,
            rms_mix_rate=1,
        )

        from .voice import ChunkedConverter
        self._conv = ChunkedConverter(
            self._infer_chunk, sr,
            chunk_ms=300, context_ms=200, xfade_ms=30
        )

    # ---- inferencia (un chunk a la vez, con archivos tmp) ----
    def _infer_chunk(self, audio: np.ndarray) -> np.ndarray:
        fd_in,  path_in  = tempfile.mkstemp(suffix=".wav")
        fd_out, path_out = tempfile.mkstemp(suffix=".wav")
        try:
            os.close(fd_in)
            os.close(fd_out)
            sf.write(path_in, audio, self.sr)
            with self._lock:
                self._rvc.infer_file(path_in, path_out)
            result, out_sr = sf.read(path_out, dtype="float32")
            if result.ndim > 1:
                result = result.mean(axis=1)
            # Re-muestrea si el SR devuelto difiere (no debería con resample_sr)
            if out_sr != self.sr and len(result) > 1:
                new_len = int(len(result) * self.sr / out_sr)
                result = np.interp(
                    np.linspace(0, len(result) - 1, new_len),
                    np.arange(len(result)), result
                ).astype(np.float32)
            return result
        finally:
            for p in (path_in, path_out):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    # ---- interfaz pública ----
    def process(self, x: np.ndarray) -> np.ndarray:
        return self._conv.process(x)

    def set_pitch(self, semitones: int) -> None:
        self._rvc.set_params(f0up_key=int(semitones))

    def set_index_rate(self, rate: float) -> None:
        self._rvc.set_params(index_rate=float(rate))


def build_backend(model_path: str, index_path: str, sr: int,
                  pitch_shift: int = 0, index_rate: float = 0.66) -> RVCBackend:
    """Construye y devuelve el backend RVC listo para usar."""
    return RVCBackend(model_path, index_path, sr, pitch_shift, index_rate)
