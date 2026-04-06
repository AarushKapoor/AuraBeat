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

        if pyo is None: return

        master_mul = getattr(self.master, "master_gain", 0.9)

        # 1. Amplitude Envelope
        self._env = pyo.Adsr(attack=0.005, decay=0.1, sustain=0.8, release=0.3)

        # 2. Richer Oscillator: SuperSaw (Built-in detuned saw waves)
        self._osc = pyo.SuperSaw(freq=self.freq, detune=0.5, mul=self._env * master_mul)

        # 3. The "Piano" Filter
        self._filt = pyo.Biquad(self._osc, freq=1000, q=1, type=0)

        # 4. Output
        self._pan = pyo.Pan(self._filt, outs=2, pan=(self.pan + 1.0) * 0.5).out()

    def set_pitch(self, midi_note: int):
        self.note = int(midi_note)
        self.freq = midi_to_hz(self.note)
        if pyo:
            self._osc.freq = self.freq

    def gate_on(self, velocity: int):
        if pyo is None: return
        v = clamp01(velocity / 127.0)

        # Velocity-to-Brightness: Harder hits = higher filter cutoff
        self._filt.freq = 400 + (3000 * v)
        self._env.mul = 0.5 + (0.5 * v)

        self._env.play()

    def gate_off(self):
        if pyo: self._env.stop()

    def panic(self):
        if pyo: self._env.stop()

    def panic(self):
        if pyo is None or self._env is None:
            return
        self._env.mul = 0.0
        self._env.stop()