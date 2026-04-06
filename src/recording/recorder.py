import time
from recording.note_event import NoteEvent


class Recorder:
    """
    Records finger presses into NoteEvents.
    Produces: pitch, start_time, end_time, hand, finger
    """
    def __init__(self, pitch_mapper):
        self.pitch_mapper = pitch_mapper
        self.is_recording = False
        self.start_time = 0
        self.active_notes = {}  # (hand,finger) → NoteEvent
        self.events = []

    # ---------------------------------------------------------
    # Start recording
    # ---------------------------------------------------------
    def start(self):
        self.is_recording = True
        self.start_time = time.time()
        self.events = []
        self.active_notes = {}

    # ---------------------------------------------------------
    # Stop recording and finalize events
    # ---------------------------------------------------------
    def stop(self):
        self.is_recording = False
        now = time.time() - self.start_time

        # Close any open notes
        for ev in self.active_notes.values():
            ev.end = now
            self.events.append(ev)

        self.active_notes.clear()
        return self.events

    # ---------------------------------------------------------
    # Update per finger event
    # ---------------------------------------------------------
    def update(self, hand, finger, is_down):
        if not self.is_recording:
            return

            # Normalize hand name
        hand = hand.lower()  # "Left" → "left"

        now = time.time() - self.start_time
        key = (hand, finger)

        # Finger pressed → start note
        if is_down and key not in self.active_notes:
            pitch = self.pitch_mapper.get_pitch(hand, finger)
            ev = NoteEvent(pitch, start=now, hand=hand, finger=finger)
            self.active_notes[key] = ev

        # Finger released → end note
        elif not is_down and key in self.active_notes:
            ev = self.active_notes.pop(key)
            ev.end = now
            self.events.append(ev)

    def get_events(self):
        return self.events


