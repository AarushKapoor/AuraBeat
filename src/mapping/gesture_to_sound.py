# src/mapping/gesture_to_sound.py
from __future__ import annotations

from typing import Optional

# Re-export midi_to_name for convenience in callers importing from here
from scale_window import midi_to_name


def note_on(audio_engine: Optional[object], note: int, velocity: int):
    """Send a NOTE ON to the audio layer (or log)."""
    print(f"[MIDI] note_on {note} vel={velocity}")
    if audio_engine and hasattr(audio_engine, "note_on"):
        try:
            audio_engine.note_on(note, velocity)
        except Exception as e:
            print(f"[Audio] note_on failed: {e}")


def note_off(audio_engine: Optional[object], note: int):
    """Send a NOTE OFF to the audio layer (or log)."""
    print(f"[MIDI] note_off {note}")
    if audio_engine and hasattr(audio_engine, "note_off"):
        try:
            audio_engine.note_off(note)
        except Exception as e:
            print(f"[Audio] note_off failed: {e}")
