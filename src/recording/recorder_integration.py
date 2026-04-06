from recording.note_event import NoteEvent


class RecorderToPianoRollBridge:
    def __init__(self, recorder, piano_roll, time_grid, pitch_mapper):
        self.recorder = recorder
        self.piano_roll = piano_roll
        self.time_grid = time_grid
        self.pitch_mapper = pitch_mapper

    def apply_recorded_events(self):
        events = self.recorder.events
        if not events:
            return

        # Convert events to canvas notes
        notes = []
        for ev in events:
            notes.append(
                NoteEvent(
                    pitch=ev.pitch,
                    start=ev.start,
                    end=ev.end,
                    velocity=getattr(ev, "velocity", 1.0),
                    hand=getattr(ev, "hand", None),
                    finger=getattr(ev, "finger", None)
                )
            )

        self.piano_roll.note_canvas.notes = notes

        # Resize scroll region
        self.piano_roll.update_scroll_region()

        # Redraw notes using correct pitch mapper
        self.piano_roll.redraw_notes(self.time_grid)
