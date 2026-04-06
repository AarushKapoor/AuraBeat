from kivy.clock import Clock


class PlaybackEngine:
    """
    Drives falling-note playback for the Piano Roll.
    Handles:
        - time advancement
        - scroll position
        - note_on / note_off
        - loop mode (optional)
    """

    def __init__(self, time_grid, note_canvas, scroll_view, audio_interface):
        """
        time_grid: TimeGrid instance
        note_canvas: NoteCanvas instance
        scroll_view: ScrollView containing the note canvas
        audio_interface: wrapper with note_on / note_off
        """
        self.time_grid = time_grid
        self.note_canvas = note_canvas
        self.scroll = scroll_view
        self.audio = audio_interface

        self.current_time = 0.0
        self.is_playing = False
        self.loop_enabled = False
        self.loop_end_time = None

        # Track which notes are currently sounding
        self.active_notes = set()

        self._clock_event = None

    # ---------------------------------------------------------
    # Playback control
    # ---------------------------------------------------------

    def play(self):
        if self.is_playing:
            return

        self.is_playing = True
        self._clock_event = Clock.schedule_interval(self._update, 1 / 60.0)

    def pause(self):
        if not self.is_playing:
            return

        self.is_playing = False
        if self._clock_event is not None:
            self._clock_event.cancel()
            self._clock_event = None

        # Stop all sounding notes
        for pitch in list(self.active_notes):
            self.audio.note_off(pitch)
        self.active_notes.clear()

    def stop(self):
        self.pause()
        self.current_time = 0.0
        self._update_scroll()

    # ---------------------------------------------------------
    # Looping
    # ---------------------------------------------------------

    def enable_loop(self, end_time):
        self.loop_enabled = True
        self.loop_end_time = float(end_time)

    def disable_loop(self):
        self.loop_enabled = False
        self.loop_end_time = None

    # ---------------------------------------------------------
    # Main update loop
    # ---------------------------------------------------------

    def _update(self, dt):
        if not self.is_playing:
            return

        # Advance time
        self.current_time += dt

        # Looping logic
        if self.loop_enabled and self.loop_end_time is not None:
            if self.current_time >= self.loop_end_time:
                self.current_time = 0.0
                for pitch in list(self.active_notes):
                    self.audio.note_off(pitch)
                self.active_notes.clear()

        # Update scroll position
        self._update_scroll()

        # Trigger notes based on time
        self._trigger_notes()

    # ---------------------------------------------------------
    # ScrollView position update
    # ---------------------------------------------------------

    def _update_scroll(self):
        """
        ScrollView.scroll_y is normalized (0–1).
        We convert current_time → pixel_y → normalized scroll.
        """
        y = self.time_grid.time_to_y(self.current_time)

        max_scroll_px = max(1, self.note_canvas.height - self.scroll.height)
        norm = y / max_scroll_px

        # Clamp
        norm = max(0.0, min(1.0, norm))

        # Kivy: 1.0 = top, 0.0 = bottom → invert
        self.scroll.scroll_y = 1.0 - norm

    # ---------------------------------------------------------
    # Time-based note triggering
    # ---------------------------------------------------------

    def _trigger_notes(self):
        t = self.current_time

        for note in self.note_canvas.notes:
            # Support both NoteEvent objects and dicts
            start = note.start if hasattr(note, "start") else note["start"]
            end = note.end if hasattr(note, "end") else note["end"]
            pitch = note.pitch if hasattr(note, "pitch") else note["pitch"]
            velocity = getattr(note, "velocity", 1.0)

            tag = pitch  # simple, stable tag

            # Already sounding
            if pitch in self.active_notes:
                if t > end:
                    self.audio.note_off(pitch, tag)
                    self.active_notes.remove(pitch)
                continue

            # Not yet sounding
            if start <= t <= end:
                self.audio.note_on(pitch, velocity, tag)
                self.active_notes.add(pitch)


