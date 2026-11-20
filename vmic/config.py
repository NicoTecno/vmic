import json
import os
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("VMIC_HOME", Path.home() / ".vmic"))
CONFIG_PATH = CONFIG_DIR / "config.json"


@dataclass
class Config:
    input_device: str = ""
    output_device: str = ""
    monitor_device: str = ""
    sample_rate: int = 48000
    block_size: int = 480          # 10 ms a 48 kHz
    mic_gain_db: float = 0.0
    gate_enabled: bool = True
    gate_threshold_db: float = -50.0
    denoise_enabled: bool = True
    voice_mode: str = "off"        # off | pitch | rvc
    pitch_semitones: float = 0.0
    female_pitch: float = 4.5      # semitonos para voz femenina
    female_formant: float = 1.18   # factor de formantes (1.10 - 1.25)
    rvc_model_path: str = ""
    rvc_index_path: str = ""
    rvc_pitch: int = 0             # semitones de desplazamiento para RVC
    rvc_index_rate: float = 0.66   # influencia del índice FAISS (0.0–1.0)
    monitor_voice: bool = False
    monitor_sounds: bool = True
    sounds_volume: float = 0.8
    stop_hotkey: str = "<ctrl>+<alt>+0"
    sounds: list = field(default_factory=list)  # [{name, path, hotkey, volume}]


def load_config() -> Config:
    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        known = {f.name for f in fields(Config)}
        return Config(**{k: v for k, v in raw.items() if k in known})
    except Exception:
        return Config()


def save_config(cfg: Config) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")
