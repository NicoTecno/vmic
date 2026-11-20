class HotkeyManager:
    """Atajos globales (funcionan aunque Discord/juego tenga el foco). Formato pynput: '<ctrl>+<alt>+1'."""

    def __init__(self):
        self.listener = None

    @staticmethod
    def valid(combo: str) -> bool:
        try:
            from pynput import keyboard
            keyboard.HotKey.parse(combo)
            return True
        except Exception:
            return False

    def set(self, mapping):
        self.stop()
        good = {k: v for k, v in mapping.items() if k and self.valid(k)}
        if not good:
            return
        from pynput import keyboard
        self.listener = keyboard.GlobalHotKeys(good)
        self.listener.daemon = True
        self.listener.start()

    def stop(self):
        if self.listener:
            self.listener.stop()
            self.listener = None
