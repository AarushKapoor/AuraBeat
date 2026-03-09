# src/audio/voices/keyboard.py
from __future__ import annotations
from ..utils import midi_to_hz, clamp01

try:
    import pyo
except ImportError:
    pyo = None

class KeyboardVoice:

    def __init__(self, master_bus, index: int, pan: float = 0.0):
        self.master = master_bus
        self.index = int(index)
        self.pan = max(-1.0, min(1.0, float(pan)))
        self.note = 60
        self.freq = midi_to_hz(self.note)

        if pyo is None:
            self._env = self._osc = self._pan = None
            return


        master_mul = getattr(self.master, "master_gain", 0.9)

        self._env = pyo.Adsr(
            attack=0.002,  # 2 ms
            decay=0.06,  # 60 ms
            sustain=1.0,  # no hold
            release=0.05,  # 50 ms
            mul=0.0
        )
        self._osc = pyo.Sine(freq=self.freq, mul=self._env * master_mul)


        pan_pos = (self.pan + 1.0) * 0.5
        self._pan = pyo.Pan(self._osc, outs=2, pan=pan_pos)


        self._out = self._pan.out()

    def set_pitch(self, midi_note: int):
        self.note = int(midi_note)
        self.freq = midi_to_hz(self.note)
        if pyo is None or self._osc is None:
            return
        self._osc.freq = self.freq

    def gate_on(self, velocity: int):
        if pyo is None or self._env is None:
            print("[KeyboardVoice] gate_on skipped (no pyo).")
            return
        v = clamp01(velocity / 127.0)

        self._env.mul = 0.3 + 0.7 * v
        print(f"[KeyboardVoice] gate_on freq={self.freq:.1f}Hz mul={self._env.mul:.2f}")
        self._env.play()

    def gate_off(self):
        if pyo is None or self._env is None:
            return
        self._env.stop()

    def panic(self):
        if pyo is None or self._env is None:
            return
        self._env.mul = 0.0
        self._env.stop()