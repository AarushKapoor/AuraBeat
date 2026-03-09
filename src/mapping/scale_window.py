# src/mapping/scale_window.py
from dataclasses import dataclass
from typing import List

SHIFT_STEP = 5  # set 4 if you want 4-note windows instead

NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
def midi_to_name(m: int) -> str:
    return f"{NOTE_NAMES[m % 12]}{(m // 12) - 1}"

def build_c_major(root_midi=60, octaves=6) -> List[int]:
    pattern = [0,2,4,5,7,9,11]
    base = root_midi - 24   # room below middle C
    notes = sorted({base + 12*o + p for o in range(octaves+4) for p in pattern})
    return [n for n in notes if 0 <= n <= 127]

def _clamp(v, lo, hi): return max(lo, min(hi, v))

@dataclass
class ScaleWindow:
    notes: List[int]
    right_base_idx: int
    left_base_idx: int

    @staticmethod
    def create_c_major():
        notes = build_c_major(60, 6)
        i_c4 = notes.index(60)
        return ScaleWindow(notes=notes, right_base_idx=i_c4, left_base_idx=i_c4)

    def right_block(self) -> List[int]:  # ascending 5 notes
        i = _clamp(self.right_base_idx, 0, len(self.notes)-SHIFT_STEP)
        return [self.notes[i+d] for d in range(SHIFT_STEP)]

    def left_block(self) -> List[int]:   # descending 5 notes
        i = _clamp(self.left_base_idx, SHIFT_STEP-1, len(self.notes)-1)
        return [self.notes[i-d] for d in range(SHIFT_STEP)]

    def right_scale_up(self):    self._shift_right(+SHIFT_STEP)
    def right_scale_down(self):  self._shift_right(-SHIFT_STEP)
    def left_scale_up(self):     self._shift_left(+SHIFT_STEP)
    def left_scale_down(self):   self._shift_left(-SHIFT_STEP)

    def _shift_right(self, steps):
        self.right_base_idx = _clamp(self.right_base_idx + steps, 0, len(self.notes)-SHIFT_STEP)

    def _shift_left(self, steps):
        self.left_base_idx = _clamp(self.left_base_idx + steps, SHIFT_STEP-1, len(self.notes)-1)