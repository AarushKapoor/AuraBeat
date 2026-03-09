# src/audio/engine.py
from __future__ import annotations

import threading
import time
from typing import Dict

try:
    import pyo
except ImportError:
    pyo = None

from .voices.keyboard import KeyboardVoice
from .bus import MasterBus

FINGER_TAGS = [
    "L-Thumb","L-Index","L-Middle","L-Ring","L-Pinky",
    "R-Thumb","R-Index","R-Middle","R-Ring","R-Pinky",
]

class AudioEngine:

    def __init__(self, sr: int = 48000, buffersize: int = 256):
        self.sr = sr
        self.buffersize = buffersize
        self.server = None
        self.master = None
        self.voices: Dict[str, KeyboardVoice] = {}
        self.staccato_mode = True
        self.staccato_ms = 120
        self._on_times = {}
        self._timers = {}

    def start(self):
        if pyo is None:
            print("[AudioEngine] pyo not installed; audio disabled.")
            return
        if self.server is None:
            print("[AudioEngine] booting server...")
            self.server = pyo.Server(sr=self.sr, buffersize=self.buffersize, nchnls=2, duplex=0).boot()
            self.server.start()
            print("[AudioEngine] server started.")


            self.master = MasterBus(master_gain=0.9)


            for idx, tag in enumerate(FINGER_TAGS):
                pan = -0.35 if tag.startswith("L-") else +0.35
                self.voices[tag] = KeyboardVoice(self.master, index=idx, pan=pan)

            print(f"[AudioEngine] voices ready: {list(self.voices.keys())}")

    def stop(self):
        if self.server:
            try:
                self.server.stop()
                self.server.shutdown()
            except Exception:
                pass
            self.server = None

    def note_on(self, note: int, velocity: int, tag: str = "R-Index", behavior: str = "gate"):
        print(f"[AudioEngine] note_on note={note} vel={velocity} tag={tag}")
        v = self.voices.get(tag) or self.voices.get("R-Index") or next(iter(self.voices.values()), None)
        if v is None:
            return
        v.set_pitch(note)
        v.gate_on(velocity)

        key = (tag, note)
        RETRIGGER_GUARD_MS = 40
        prev = self._on_times.get(key)
        now = time.perf_counter()
        if prev and (now - prev) * 1000.0 < RETRIGGER_GUARD_MS:
            return

        self._on_times[key] = time.perf_counter()


        t = self._timers.pop(key, None)
        if t and t.is_alive():
            try:
                t.cancel()
            except:
                pass


        if self.staccato_mode:
            delay = max(0.01, self.staccato_ms / 1000.0)

            def _auto_off():

                if key not in self._on_times:
                    return
                v.gate_off()
                self._on_times.pop(key, None)
                self._timers.pop(key, None)

            timer = threading.Timer(delay, _auto_off)
            timer.daemon = True
            self._timers[key] = timer
            timer.start()

    def note_off(self, note: int, tag: str):
        key = (tag, note)

        t = self._timers.pop(key, None)
        if t and t.is_alive():
            try:
                t.cancel()
            except:
                pass
        self._on_times.pop(key, None)

        v = self.voices.get(tag) or self.voices.get("R-Index") or next(iter(self.voices.values()), None)
        if v is None:
            return
        v.gate_off()

    def panic(self):
        for t in list(self._timers.values()):
            try:
                if t.is_alive(): t.cancel()
            except:
                pass
        self._timers.clear()
        self._on_times.clear()
        for v in self.voices.values():
            v.panic()