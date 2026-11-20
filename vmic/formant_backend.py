"""Backend de transformación de voz (femenina/masculina) por formantes y tono usando Praat."""
import numpy as np
import parselmouth
from parselmouth.praat import call
from .voice import ChunkedConverter


class FormantVoiceModifier:
    """Modificador de voz basado en fonética acústica (PSOLA de Praat).

    Separa el tono (pitch) del tamaño del tracto vocal (formantes).
    Permite feminizar o masculinizar la voz manteniendo 100% de inteligibilidad
    y muy baja latencia, sin usar redes neuronales.
    """

    def __init__(self, sr: int, pitch_semitones: float = 4.5, formant_shift: float = 1.18):
        self.sr = sr
        self.pitch_semitones = float(pitch_semitones)
        self.formant_shift = float(formant_shift)

        # Chunk de 120 ms + contexto de 80 ms + crossfade de 25 ms
        # Latencia total acústica ≈ 145 ms
        self._conv = ChunkedConverter(
            self._infer_chunk, sr,
            chunk_ms=120, context_ms=80, xfade_ms=25
        )

    def _infer_chunk(self, audio: np.ndarray) -> np.ndarray:
        # Si es silencio absoluto o muy bajo, retornar directamente
        rms = float(np.sqrt(np.mean(audio * audio) + 1e-12))
        if rms < 0.001:
            return audio

        try:
            sound = parselmouth.Sound(audio, sampling_frequency=self.sr)
            pitch_factor = float(2.0 ** (self.pitch_semitones / 12.0))
            # Praat: Change speaker
            # pitch_min (75), pitch_max (600), formant_factor, pitch_factor, pitch_range_factor (1.05), duration_factor (1.0)
            res = call(sound, "Change speaker", 75, 600, self.formant_shift, pitch_factor, 1.05, 1.0)
            out = res.values[0].astype(np.float32)

            if len(out) != len(audio):
                out = np.interp(
                    np.linspace(0, len(out) - 1, len(audio)),
                    np.arange(len(out)), out
                ).astype(np.float32)
            return out
        except Exception:
            return audio

    def process(self, x: np.ndarray) -> np.ndarray:
        return self._conv.process(x)

    def set_pitch(self, semitones: float) -> None:
        self.pitch_semitones = float(semitones)

    def set_formant(self, ratio: float) -> None:
        self.formant_shift = float(ratio)
