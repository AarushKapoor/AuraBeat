# src/audio/utils.py
import math

A4_FREQ = 440.0
A4_MIDI = 69

def midi_to_hz(m: int) -> float:
    return A4_FREQ * (2.0 ** ((m - A4_MIDI) / 12.0))

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x
