# src/playback/playback_engine.py
from kivy.clock import Clock


class PlaybackEngine:
    def __init__(self, time_grid, note_canvas, scroll_view, audio_interface):
        self.time_grid = time_grid
        self.note_canvas = note_canvas
        self.scroll = scroll_view
        self.audio = audio_interface

        self.current_time = 0.0
        self.is_playing = False
        self.loop_enabled = False
        self.loop_end_time = None

        self.active_notes = set()
        self._clock_event = None

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
        for pitch in list(self.active_notes):
            try:
                self.audio.note_off(pitch, tag="R-Index")
            except Exception:
                pass
        self.active_notes.clear()

    def stop(self):
        self.pause()
        self.current_time = 0.0
        self.scroll.scroll_y = 1.0  # reset to top

    def enable_loop(self, end_time):
        self.loop_enabled = True
        self.loop_end_time = float(end_time)

    def disable_loop(self):
        self.loop_enabled = False
        self.loop_end_time = None

    def _update(self, dt):
        if not self.is_playing:
            return

        self.current_time += dt

        if self.loop_enabled and self.loop_end_time is not None:
            if self.current_time >= self.loop_end_time:
                self.current_time = 0.0
                for pitch in list(self.active_notes):
                    try:
                        self.audio.note_off(pitch, tag="R-Index")
                    except Exception:
                        pass
                self.active_notes.clear()

        # Auto-stop when past all notes
        if self.note_canvas.notes:
            max_end = max(
                (n.end if hasattr(n, "end") else n["end"])
                for n in self.note_canvas.notes
            )
            if self.current_time > max_end + 0.5:
                self.stop()
                return

        self._update_scroll()
        self._trigger_notes()

    def _update_scroll(self):
        """
        Move the viewport downward over time so notes fall toward the keyboard.
        At t=0, scroll is at top (scroll_y=1.0).
        As time advances, viewport moves down so notes arrive at the strike line.
        """
        canvas_h = self.note_canvas.height
        view_h = self.scroll.height

        max_scroll_px = max(1, canvas_h - view_h)

        # How far down the canvas we are in pixels
        # Notes are drawn flipped: a note at time t is at y = canvas_h - time_to_y(t)
        # We want that note to be at the TOP of the viewport when it's time to play it
        scroll_px = canvas_h - self.time_grid.time_to_y(self.current_time) - view_h

        norm = scroll_px / max_scroll_px
        norm = max(0.0, min(1.0, norm))

        # Kivy scroll_y: 1.0 = top, 0.0 = bottom
        self.scroll.scroll_y = 1.0 - norm

    def _trigger_notes(self):
        t = self.current_time

        for note in self.note_canvas.notes:
            start  = note.start  if hasattr(note, "start")  else note["start"]
            end    = note.end    if hasattr(note, "end")    else note["end"]
            pitch  = note.pitch  if hasattr(note, "pitch")  else note["pitch"]
            hand   = getattr(note, "hand",   "right") or "right"
            finger = getattr(note, "finger", "Index") or "Index"

            hand_char  = "R" if hand.lower() == "right" else "L"
            finger_cap = finger.capitalize()
            tag = f"{hand_char}-{finger_cap}"

            if pitch in self.active_notes:
                if t > end:
                    try:
                        self.audio.note_off(pitch, tag=tag)
                    except Exception:
                        pass
                    self.active_notes.remove(pitch)
                continue

            if start <= t <= end:
                velocity = getattr(note, "velocity", 1.0)
                vel_int = int(velocity * 127) if velocity <= 1.0 else int(velocity)
                try:
                    self.audio.note_on(pitch, vel_int, tag=tag)
                except Exception:
                    pass
                self.active_notes.add(pitch)