"""Prueba el motor sin placa de audio: python selftest.py"""
import numpy as np
from vmic.config import Config
from vmic.dsp import NoiseGate, PitchShifter, SpectralDenoiser
from vmic.engine import Engine
from vmic.soundboard import Soundboard
from vmic.voice import ChunkedConverter

SR, N = 48000, 480
rng = np.random.default_rng(0)


def blocks(x):
    return [x[i:i + N] for i in range(0, len(x) - N + 1, N)]


def peak_hz(x):
    sp = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.argmax(sp) * SR / len(x)


t = np.arange(SR * 2) / SR
tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

# 1) pitch +12 st => 880 Hz, -12 st => 220 Hz
for st, want in ((12, 880), (-12, 220)):
    ps = PitchShifter(SR, st)
    y = np.concatenate([ps.process(b) for b in blocks(tone)])[SR:]
    hz = peak_hz(y)
    assert abs(hz - want) < 15, (st, hz)
    assert np.isfinite(y).all() and np.max(np.abs(y)) < 1.0
    print(f"pitch {st:+d} st -> {hz:.0f} Hz  OK")

# 2) denoiser: baja ruido blanco y conserva "habla" sintética (armónicos con pausas)
L = SR * 6
noise = (0.01 * rng.standard_normal(L)).astype(np.float32)
tt = np.arange(L) / SR
env = ((tt % 0.6) < 0.3).astype(np.float32)
voice = sum(0.1 / k * np.sin(2 * np.pi * 150 * k * tt) for k in range(1, 8)).astype(np.float32) * env


def run_dn(x):
    d = SpectralDenoiser()
    return np.concatenate([d.process(b) for b in blocks(x)])


y = run_dn(noise)[SR * 2:]
red = 20 * np.log10(np.std(noise[SR * 2:SR * 2 + len(y)]) / np.std(y))
print(f"denoiser: ruido reducido {red:.1f} dB")
assert red > 8
y2, lat = run_dn(voice + noise), 768
seg = slice(SR * 2, len(y2) - lat)
on = env[seg] > 0
v_in, v_out = np.std(voice[seg][on]), np.std(y2[lat:][seg][on])
print(f"denoiser: voz conservada ({v_out:.3f} vs {v_in:.3f})")
assert v_out > 0.8 * v_in
tone = tone  # (se usa abajo)

# 3) gate: silencia ruido bajo, deja pasar tono
g = NoiseGate(SR, -30)
quiet = np.concatenate([g.process(b) for b in blocks(noise[: SR])])[SR // 2:]
loud = np.concatenate([g.process(b) for b in blocks(tone[: SR])])[SR // 2:]
assert np.std(quiet) < 1e-3 and np.std(loud) > 0.15
print("gate OK")

# 4) ChunkedConverter (identidad): conserva la señal con retardo constante
cc = ChunkedConverter(lambda a: a, SR)
y = np.concatenate([cc.process(b) for b in blocks(tone)])
lag = cc.chunk + cc.xf  # retardo esperado
body = y[SR:SR * 2 - 1000]
ref = tone[SR - lag: SR - lag + len(body)]
err = np.max(np.abs(body - ref))
print(f"chunked: retardo {lag / SR * 1000:.0f} ms, error máx {err:.4f}")
assert err < 0.02

# 5) soundboard polifónico
sb = Soundboard(SR)
sb.library["x"] = (np.ones(1000, np.float32) * 0.1, 1.0)
sb.volume = 1.0
sb.play("x"); sb.play("x")
m = sb.mix(500)
assert abs(m[0] - 0.2) < 1e-6 and len(sb.voices) == 2
sb.mix(600)
assert len(sb.voices) == 0
print("soundboard OK")

# 6) cadena completa
cfg = Config(voice_mode="pitch", pitch_semitones=-4)
eng = Engine(cfg, Soundboard(SR))
out = np.concatenate([eng.process_block(b) for b in blocks(voice + noise)])
assert len(out) == len(blocks(voice + noise)) * N and np.isfinite(out).all()
print("engine.process_block OK\n\nTodo OK ✔")
