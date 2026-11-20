import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from . import devices, rvc_backend
from .config import load_config, save_config
from .engine import Engine
from .hotkeys import HotkeyManager
from .soundboard import Soundboard

MODES = {"Sin cambio": "off", "Cambio de tono (pitch)": "pitch", "IA / RVC (Messi, etc.)": "rvc"}
MODE_BY_VALUE = {v: k for k, v in MODES.items()}
NONE = "(ninguno)"
PAD = dict(padx=8, pady=4)


class App:
    def __init__(self):
        self.cfg = load_config()
        self.sb = Soundboard(self.cfg.sample_rate)
        self.engine = Engine(self.cfg, self.sb)
        self.hk = HotkeyManager()
        self.root = tk.Tk()
        self.root.title("VMic - micrófono virtual local")
        self.root.geometry("740x800")
        ttk.Style().theme_use("clam")
        self._load_sounds()
        self._build()
        self.refresh_devices()
        self.refresh_sounds()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.tick()

    # ------------------------------------------------------------ UI
    def _build(self):
        c, r = self.cfg, self.root

        # ---- Dispositivos ----
        f = ttk.LabelFrame(r, text="Dispositivos")
        f.pack(fill="x", **PAD)
        self.in_var  = tk.StringVar(value=c.input_device)
        self.out_var = tk.StringVar(value=c.output_device)
        self.mon_var = tk.StringVar(value=c.monitor_device or NONE)
        self.combos  = {}
        for i, (label, var) in enumerate([
            ("Tu micrófono real",              self.in_var),
            ("Salida → cable virtual (CABLE Input)", self.out_var),
            ("Escucharte en (auriculares)",    self.mon_var),
        ]):
            ttk.Label(f, text=label).grid(row=i, column=0, sticky="w", padx=6, pady=2)
            cb = ttk.Combobox(f, textvariable=var, state="readonly", width=55)
            cb.grid(row=i, column=1, padx=6, pady=2, sticky="ew")
            self.combos[label] = cb
        f.columnconfigure(1, weight=1)
        row = ttk.Frame(f)
        row.grid(row=3, column=0, columnspan=2, sticky="ew", pady=4)
        self.btn = ttk.Button(row, text="▶ Iniciar", command=self.toggle)
        self.btn.pack(side="left", padx=6)
        ttk.Button(row, text="Refrescar", command=self.refresh_devices).pack(side="left")
        self.mute = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="Mute", variable=self.mute).pack(side="left", padx=12)
        self.meter = ttk.Progressbar(row, maximum=100, length=180)
        self.meter.pack(side="right", padx=6)
        ttk.Label(row, text="Nivel salida").pack(side="right")

        # ---- Micrófono ----
        f = ttk.LabelFrame(r, text="Micrófono")
        f.pack(fill="x", **PAD)
        self.gain   = tk.DoubleVar(value=c.mic_gain_db)
        self.denoise = tk.BooleanVar(value=c.denoise_enabled)
        self.gate   = tk.BooleanVar(value=c.gate_enabled)
        self.thr    = tk.DoubleVar(value=c.gate_threshold_db)
        self._slider(f, 0, "Ganancia (dB)", self.gain, -20, 20)
        ttk.Checkbutton(f, text="Reducción de ruido", variable=self.denoise).grid(row=1, column=0, sticky="w", padx=6)
        ttk.Checkbutton(f, text="Noise gate",         variable=self.gate).grid(row=2, column=0, sticky="w", padx=6)
        self._slider(f, 2, "Umbral gate (dB)", self.thr, -80, -20, col=1)

        # ---- Voz ----
        self._voice_lf = ttk.LabelFrame(r, text="Voz")
        self._voice_lf.pack(fill="x", **PAD)
        f = self._voice_lf
        self.mode = tk.StringVar(value=MODE_BY_VALUE.get(c.voice_mode, "Sin cambio"))
        ttk.Label(f, text="Modo").grid(row=0, column=0, sticky="w", padx=6)
        ttk.Combobox(f, textvariable=self.mode, values=list(MODES),
                     state="readonly", width=30).grid(row=0, column=1, sticky="w")
        self.pitch = tk.DoubleVar(value=c.pitch_semitones)
        self._slider(f, 1, "Tono (semitonos)", self.pitch, -12, 12)
        self.monv = tk.BooleanVar(value=c.monitor_voice)
        ttk.Checkbutton(f, text="Escuchar mi voz procesada",
                        variable=self.monv).grid(row=2, column=0, columnspan=2, sticky="w", padx=6)

        # ---- Configuración RVC (oculto hasta que se elija ese modo) ----
        self._rvc_lf = ttk.LabelFrame(r, text="Configuración RVC (IA)")
        self._rvc_visible = False

        rf = self._rvc_lf

        # Fila: modelo .pth
        row0 = ttk.Frame(rf)
        row0.grid(row=0, column=0, columnspan=4, sticky="ew", padx=6, pady=3)
        ttk.Label(row0, text="Modelo (.pth):").pack(side="left")
        self.rvc_model_var = tk.StringVar(value=c.rvc_model_path)
        ttk.Entry(row0, textvariable=self.rvc_model_var, width=48).pack(side="left", padx=4)
        ttk.Button(row0, text="…", width=3, command=self._browse_rvc_model).pack(side="left")

        # Fila: índice .index
        row1 = ttk.Frame(rf)
        row1.grid(row=1, column=0, columnspan=4, sticky="ew", padx=6, pady=3)
        ttk.Label(row1, text="Índice (.index):").pack(side="left")
        self.rvc_index_var = tk.StringVar(value=c.rvc_index_path)
        ttk.Entry(row1, textvariable=self.rvc_index_var, width=48).pack(side="left", padx=4)
        ttk.Button(row1, text="…", width=3, command=self._browse_rvc_index).pack(side="left")

        # Sliders: pitch RVC + index rate (en la misma fila)
        sf2 = ttk.Frame(rf)
        sf2.grid(row=2, column=0, columnspan=4, sticky="ew", padx=6, pady=4)
        self.rvc_pitch_var = tk.DoubleVar(value=float(c.rvc_pitch))
        self.rvc_rate_var  = tk.DoubleVar(value=c.rvc_index_rate)
        self._slider(sf2, 0, "Tono RVC (st)",  self.rvc_pitch_var, -12, 12, col=0)
        self._slider(sf2, 0, "Index rate",      self.rvc_rate_var,  0.0, 1.0, col=1)

        # Botón cargar + estado
        row3 = ttk.Frame(rf)
        row3.grid(row=3, column=0, columnspan=4, sticky="ew", padx=6, pady=4)
        self.rvc_load_btn = ttk.Button(row3, text="⚡ Cargar modelo", command=self._load_rvc)
        self.rvc_load_btn.pack(side="left")
        ttk.Button(row3, text="✕ Descargar", command=self._unload_rvc).pack(side="left", padx=6)
        self.rvc_status_lbl = ttk.Label(row3, text="Sin cargar", foreground="gray")
        self.rvc_status_lbl.pack(side="left", padx=10)

        rf.columnconfigure(0, weight=1)

        # ---- Soundboard ----
        self._sb_lf = ttk.LabelFrame(r, text="Soundboard  (clic = reproducir · clic derecho = opciones)")
        self._sb_lf.pack(fill="both", expand=True, **PAD)
        f = self._sb_lf
        top = ttk.Frame(f)
        top.pack(fill="x")
        self.vol = tk.DoubleVar(value=c.sounds_volume)
        self._slider(top, 0, "Volumen sonidos", self.vol, 0, 1.5)
        self.mons = tk.BooleanVar(value=c.monitor_sounds)
        ttk.Checkbutton(top, text="Escucharlos yo también", variable=self.mons).grid(row=0, column=3, padx=6)
        ttk.Button(top, text="+ Agregar",    command=self.add_sounds).grid(row=0, column=4, padx=4)
        ttk.Button(top, text="■ Parar todos", command=self.sb.stop_all).grid(row=0, column=5, padx=4)
        self.grid_frame = ttk.Frame(f)
        self.grid_frame.pack(fill="both", expand=True, pady=6)

        # Traces — llaman a apply() ante cualquier cambio
        for v in (self.gain, self.denoise, self.gate, self.thr, self.mode, self.pitch,
                  self.monv, self.mons, self.vol, self.mute,
                  self.rvc_pitch_var, self.rvc_rate_var):
            v.trace_add("write", lambda *a: self.apply())

        self.apply()

    def _slider(self, parent, row, text, var, lo, hi, col=0):
        ttk.Label(parent, text=text).grid(row=row, column=col * 3, sticky="w", padx=6)
        ttk.Scale(parent, from_=lo, to=hi, variable=var, length=160).grid(row=row, column=col * 3 + 1, padx=4)
        lbl = ttk.Label(parent, width=6)
        lbl.grid(row=row, column=col * 3 + 2)
        var.trace_add("write", lambda *a: lbl.config(text=f"{var.get():.1f}"))
        lbl.config(text=f"{var.get():.1f}")

    # ------------------------------------------------------------ lógica
    def apply(self):
        c = self.cfg
        c.mic_gain_db, c.denoise_enabled = self.gain.get(), self.denoise.get()
        c.gate_enabled, c.gate_threshold_db = self.gate.get(), self.thr.get()
        mode = MODES.get(self.mode.get(), "off")
        if mode == "rvc" and not rvc_backend.available():
            mode = "off"
        c.voice_mode, c.pitch_semitones = mode, self.pitch.get()
        c.monitor_voice, c.monitor_sounds, c.sounds_volume = self.monv.get(), self.mons.get(), self.vol.get()
        c.rvc_pitch       = int(round(self.rvc_pitch_var.get()))
        c.rvc_index_rate  = self.rvc_rate_var.get()
        self.engine.muted = self.mute.get()

        # Actualizar parámetros RVC en caliente si el modelo ya está cargado
        if self.engine.rvc is not None:
            self.engine.rvc.set_pitch(c.rvc_pitch)
            self.engine.rvc.set_index_rate(c.rvc_index_rate)

        # Mostrar/ocultar sección RVC
        self._set_rvc_visible(mode == "rvc")

    def _set_rvc_visible(self, show: bool):
        if show and not self._rvc_visible:
            self._rvc_lf.pack(fill="x", before=self._sb_lf, **PAD)
            self._rvc_visible = True
        elif not show and self._rvc_visible:
            self._rvc_lf.pack_forget()
            self._rvc_visible = False

    def refresh_devices(self):
        try:
            ins, outs = devices.list_devices("input"), devices.list_devices("output")
        except Exception as e:
            messagebox.showerror("Audio", f"No se pudo inicializar el audio:\n{e}")
            return
        in_names, out_names = [n for _, n in ins], [n for _, n in outs]
        labels = list(self.combos)
        self.combos[labels[0]]["values"] = in_names
        self.combos[labels[1]]["values"] = out_names
        self.combos[labels[2]]["values"] = [NONE] + out_names
        if self.in_var.get() not in in_names and in_names:
            self.in_var.set(in_names[0])
        if self.out_var.get() not in out_names:
            self.out_var.set(devices.guess_virtual_output(outs) or (out_names[0] if out_names else ""))
        if self.mon_var.get() not in out_names:
            self.mon_var.set(NONE)

    def toggle(self):
        if self.engine.running:
            self.engine.stop()
            self.btn.config(text="▶ Iniciar")
            return
        out = self.out_var.get()
        if "cable" not in out.lower() and "voicemeeter" not in out.lower():
            if not messagebox.askyesno("Ojo",
                "La salida elegida no parece un cable virtual (VB-Cable).\n"
                "Si es tu parlante, vas a escucharte con eco.\n¿Continuar igual?"):
                return
        mon = self.mon_var.get()
        try:
            self.engine.start(self.in_var.get(), out, "" if mon == NONE else mon)
        except Exception as e:
            messagebox.showerror("Error al iniciar", str(e))
            return
        c = self.cfg
        c.input_device, c.output_device, c.monitor_device = self.in_var.get(), out, "" if mon == NONE else mon
        save_config(c)
        self.btn.config(text="■ Detener")

    # ---- RVC ----
    def _browse_rvc_model(self):
        p = filedialog.askopenfilename(title="Seleccioná el modelo RVC (.pth)",
                                       filetypes=[("PyTorch model", "*.pth"), ("Todos", "*.*")])
        if p:
            self.rvc_model_var.set(p)

    def _browse_rvc_index(self):
        p = filedialog.askopenfilename(title="Seleccioná el índice FAISS (.index)",
                                       filetypes=[("FAISS index", "*.index"), ("Todos", "*.*")])
        if p:
            self.rvc_index_var.set(p)

    def _load_rvc(self):
        model = self.rvc_model_var.get().strip()
        index = self.rvc_index_var.get().strip()
        if not model or not Path(model).is_file():
            messagebox.showerror("RVC", "Seleccioná un archivo .pth válido primero.")
            return
        if not rvc_backend.available():
            messagebox.showerror("RVC", "rvc-python no está instalado.\nEjecutá: pip install rvc-python")
            return
        self.cfg.rvc_model_path = model
        self.cfg.rvc_index_path = index
        self.rvc_status_lbl.config(text="⏳ Cargando…  (descarga HuBERT si es primera vez)", foreground="orange")
        self.rvc_load_btn.state(["disabled"])
        self.engine.load_rvc(model, index,
                             pitch_shift=int(round(self.rvc_pitch_var.get())),
                             index_rate=self.rvc_rate_var.get())

    def _unload_rvc(self):
        self.engine.unload_rvc()
        self.rvc_status_lbl.config(text="Sin cargar", foreground="gray")
        self.rvc_load_btn.state(["!disabled"])

    # ---- soundboard ----
    def _load_sounds(self):
        for s in list(self.cfg.sounds):
            try:
                self.sb.add(s["name"], s["path"], s.get("volume", 1.0))
            except Exception:
                pass

    def add_sounds(self):
        paths = filedialog.askopenfilenames(title="Elegí audios",
                                            filetypes=[("Audio", "*.wav *.mp3 *.ogg *.flac"), ("Todos", "*.*")])
        for p in paths:
            name, k = Path(p).stem, 2
            while any(s["name"] == name for s in self.cfg.sounds):
                name, k = f"{Path(p).stem} ({k})", k + 1
            try:
                self.sb.add(name, p)
            except Exception as e:
                messagebox.showerror("Audio", f"No pude cargar {p}:\n{e}")
                continue
            self.cfg.sounds.append({"name": name, "path": p, "hotkey": "", "volume": 1.0})
        self.refresh_sounds()

    def refresh_sounds(self):
        for w in self.grid_frame.winfo_children():
            w.destroy()
        for i, s in enumerate(self.cfg.sounds):
            hk = s.get("hotkey", "")
            b = ttk.Button(self.grid_frame, text=f"{s['name']}\n{hk}", width=16,
                           command=lambda n=s["name"]: self.sb.play(n))
            b.grid(row=i // 4, column=i % 4, padx=3, pady=3, sticky="ew")
            b.bind("<Button-3>", lambda e, n=s["name"]: self.sound_menu(e, n))
        mapping = {s["hotkey"]: (lambda n=s["name"]: self.sb.play(n))
                   for s in self.cfg.sounds if s.get("hotkey")}
        mapping[self.cfg.stop_hotkey] = self.sb.stop_all
        try:
            self.hk.set(mapping)
        except Exception as e:
            print("Hotkeys no disponibles:", e)

    def sound_menu(self, event, name):
        m = tk.Menu(self.root, tearoff=0)
        m.add_command(label="Asignar atajo global…", command=lambda: self.set_hotkey(name))
        m.add_command(label="Quitar de la lista",    command=lambda: self.remove_sound(name))
        m.tk_popup(event.x_root, event.y_root)

    def set_hotkey(self, name):
        combo = simpledialog.askstring("Atajo global",
            "Ej: <ctrl>+<alt>+1   o   <f9>\n(vacío = sin atajo)")
        if combo is None:
            return
        if combo and not HotkeyManager.valid(combo):
            messagebox.showerror("Atajo", "Formato inválido.")
            return
        for s in self.cfg.sounds:
            if s["name"] == name:
                s["hotkey"] = combo.strip()
        self.refresh_sounds()

    def remove_sound(self, name):
        self.cfg.sounds = [s for s in self.cfg.sounds if s["name"] != name]
        self.sb.remove(name)
        self.refresh_sounds()

    # ---- loop principal ----
    def tick(self):
        # VU meter
        lvl = self.engine.level_out
        self.engine.level_out = self.engine.level_in = 0.0
        self.meter["value"] = min(100, lvl * 100)

        # Errores generales
        if self.engine.last_error:
            print(self.engine.last_error)
            self.engine.last_error = ""

        # Estado del modelo RVC
        status = self.engine.rvc_status
        if status == "loading":
            self.rvc_status_lbl.config(text="⏳ Cargando…", foreground="orange")
        elif status == "ready":
            self.rvc_status_lbl.config(text="✅ Modelo listo", foreground="green")
            self.rvc_load_btn.state(["!disabled"])
        elif status.startswith("error"):
            short = status.split("\n")[0].replace("error: ", "")[:60]
            self.rvc_status_lbl.config(text=f"❌ {short}", foreground="red")
            self.rvc_load_btn.state(["!disabled"])

        self.root.after(50, self.tick)

    def close(self):
        self.engine.stop()
        self.hk.stop()
        save_config(self.cfg)
        self.root.destroy()


def run():
    App().root.mainloop()
