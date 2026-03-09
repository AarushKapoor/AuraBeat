# src/audio/bus.py
try:
    import pyo
except ImportError:
    pyo = None

class MasterBus:

    def __init__(self, master_gain: float = 0.9):
        self.master_gain = float(master_gain)

    def add_voice(self, idx: int, sigL, sigR):

        return

    def set_param(self, name: str, value):
        if name == "master_gain":
            self.master_gain = float(value)