# VMic – micrófono virtual 100 % local

Tu micrófono → reducción de ruido → noise gate → cambio de voz → **+ soundboard** → cable virtual → Discord.
Sin nube, sin cuentas, sin ningún código de red.

## Instalación (Windows)
1. Instalá **VB-Cable** (gratis): https://vb-audio.com/Cable/ y reiniciá la PC.
2. Instalá Python 3.10+ (tildá "Add to PATH").
3. En esta carpeta:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   python selftest.py      # prueba el motor sin placa de audio
   python main.py
   ```
4. En la app: elegí tu micrófono real, la salida **CABLE Input**, y (opcional) tus auriculares para escucharte.
5. En Discord → Voz y video → Micrófono: **CABLE Output**. Desactivá ahí "Supresión de ruido", "Control de ganancia automático" y "Cancelación de eco" (la app ya se encarga y se pisarían).
6. Apretá **Iniciar**.

## Soundboard
- "+ Agregar" (wav, mp3, ogg, flac). Clic en el botón = suena; podés spamear el mismo sonido (hasta 16 a la vez).
- Clic derecho → atajo global, formato `<ctrl>+<alt>+1`, `<f9>`, etc. Funcionan con el juego/Discord en foco.
- `<ctrl>+<alt>+0` = parar todos los sonidos.
- Es el reemplazo local del soundboard de Nitro: el sonido sale por tu micrófono, así que no depende de Discord.

## Estado del proyecto
| Fase | Qué | Estado |
|---|---|---|
| 1 | Mic virtual, soundboard, atajos, ruido, gate, cambio de tono, monitor | ✅ hecho (motor verificado con `selftest.py`; la GUI y el audio real hay que probarlos en tu PC) |
| 2 | Cambio de voz con IA (RVC) en GPU, con `ChunkedConverter` ya listo | ⏳ siguiente |
| 3 | RNNoise / DeepFilterNet para calidad tipo NVIDIA Broadcast, EQ, compresor | ⏳ |

## Seguridad
- La app no abre sockets ni hace requests. Podés correrla con el firewall bloqueándole todo.
- Instalá las dependencias en un entorno virtual (`.venv`).
- **Modelos de voz (Fase 2):** los `.pth` de RVC son *pickles* de PyTorch y pueden ejecutar código al cargarse. Usá solo modelos de fuentes que conozcas, cargalos con `torch.load(..., weights_only=True)` y, de ser posible, entrená/convertí los tuyos.
- Configuración guardada en `~/.vmic/config.json` (solo rutas y ajustes).

## Notas honestas
- Latencia Fase 1: ~30-50 ms (más ~5 ms en el modo "pitch"). Con RVC se espera ~250-400 ms en una RTX decente.
- Si la salida suena entrecortada, subí `block_size` a 960 en `~/.vmic/config.json`.
- Con voces de personas reales (ej. Messi): usalo para joda con amigos, sin hacerte pasar por la persona para engañar a nadie, y no distribuyas modelos de voz de terceros.
