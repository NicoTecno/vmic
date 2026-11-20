import sys


def _sd():
    import sounddevice as sd
    return sd


def list_devices(kind):
    """kind: 'input' | 'output'. En Windows se prefiere WASAPI (baja latencia, sin duplicados)."""
    sd = _sd()
    apis = sd.query_hostapis()
    prefer_wasapi = sys.platform == "win32" and any("WASAPI" in a["name"] for a in apis)
    out = []
    for i, d in enumerate(sd.query_devices()):
        ch = d["max_input_channels"] if kind == "input" else d["max_output_channels"]
        if ch <= 0:
            continue
        if prefer_wasapi and "WASAPI" not in apis[d["hostapi"]]["name"]:
            continue
        out.append((i, d["name"]))
    return out


def find_by_name(kind, name):
    for i, n in list_devices(kind):
        if n == name:
            return i
    raise ValueError(f"No encuentro el dispositivo: {name!r}")


def guess_virtual_output(devs):
    for _, name in devs:
        low = name.lower()
        if "cable input" in low or "voicemeeter input" in low:
            return name
    return ""


def extra_settings(index):
    sd = _sd()
    d = sd.query_devices(index)
    if "WASAPI" in sd.query_hostapis(d["hostapi"])["name"] and hasattr(sd, "WasapiSettings"):
        return sd.WasapiSettings(auto_convert=True)
    return None


def out_channels(index):
    return max(1, min(2, _sd().query_devices(index)["max_output_channels"]))
