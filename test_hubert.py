"""Test rápido del mapeo de pesos de HuBERT."""
import sys
import os

class FakeConfig:
    is_half = False
    device = "cpu"

lib_dir = r"C:\Users\nico\AppData\Local\Programs\Python\Python313\Lib\site-packages\rvc_python"

from rvc_python.modules.vc.utils import load_hubert

print("Cargando HuBERT...")
hubert = load_hubert(FakeConfig(), lib_dir)
print("Wrapper creado OK:", type(hubert))

import torch
audio = torch.randn(1, 16000)  # 1 seg de audio de prueba a 16kHz
padding_mask = torch.BoolTensor(audio.shape).fill_(False)
result = hubert.extract_features(source=audio, padding_mask=padding_mask, output_layer=12)
print("Output shape:", result[0].shape)   # esperado: [1, ~49, 768]
print("Output dtype:", result[0].dtype)
print(f"Output range: min={result[0].min():.4f} max={result[0].max():.4f} mean={result[0].mean():.4f}")
print("TEST PASS ✓")
