# ui/widgets/expanded_piano_roll.py

from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.metrics import dp
from kivy.clock import Clock

from ui.widgets.piano_roll import PianoRollPanel
from recording.note_event import NoteEvent


# ui/widgets/expanded_piano_roll.py

from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.metrics import dp
from kivy.clock import Clock

from ui.widgets.piano_roll import PianoRollPanel
from recording.note_event import NoteEvent


class ExpandedPianoRollPopup(Popup):
    def __init__(self, piano_roll, time_grid, pitch_mapper, **kwargs):
        print("DEBUG: ExpandedPianoRollPopup __init__ called")
        super().__init__(**kwargs)

        self.title = "Full Piano Roll"
        self.size_hint = (0.95, 0.95)
        self.auto_dismiss = False
        self.background_color = (0, 0, 0, 0.85)

        # References to minimized panel + shared systems
        self.original_panel = piano_roll
        self.time_grid = time_grid
        self.pitch_mapper = pitch_mapper

        # Root layout
        root = BoxLayout(orientation="vertical", size_hint=(1, 1))
        print("DEBUG: Created root BoxLayout")

        # Create a NEW expanded panel (do NOT share canvas)
        self.full_panel = PianoRollPanel(
            expanded=True,
            size_hint=(1, 1),
            height=dp(800)
        )
        print("DEBUG: Created full_panel:", self.full_panel)

        # Copy notes + scroll height from minimized panel
        self._copy_notes_from_original()

        # Attach time grid + pitch mapper
        self.full_panel.time_grid = self.time_grid
        self.full_panel.pitch_mapper = self.pitch_mapper
        print("DEBUG: Shared time_grid and pitch_mapper")

        root.add_widget(self.full_panel)
        print("DEBUG: Added full_panel to root")

        self.add_widget(root)
        print("DEBUG: Added root to popup")

        # Finish setup after layout
        Clock.schedule_once(self._finish_setup, 0)

    # -----------------------------------------------------------
    # Copy notes + scroll height from minimized panel
    # -----------------------------------------------------------
    def _copy_notes_from_original(self):
        print("DEBUG: Copying notes from minimized panel")

        # Convert dicts or objects → NoteEvent objects
        self.full_panel.note_canvas.notes = [
            NoteEvent(
                pitch=n.pitch if hasattr(n, "pitch") else n["pitch"],
                start=n.start if hasattr(n, "start") else n["start"],
                end=n.end if hasattr(n, "end") else n["end"],
                velocity=getattr(n, "velocity", 1.0),
                hand=getattr(n, "hand", None),
                finger=getattr(n, "finger", None)
            )
            for n in self.original_panel.note_canvas.notes
        ]

        # Copy scrollable height
        self.full_panel.note_canvas.height = self.original_panel.note_canvas.height

        print(f"DEBUG: Copied {len(self.full_panel.note_canvas.notes)} notes")

    # -----------------------------------------------------------
    # Final redraw after layout
    # -----------------------------------------------------------
    def _finish_setup(self, dt):
        print("DEBUG: _finish_setup called")
        print("DEBUG: full_panel size =", self.full_panel.size)
        print("DEBUG: popup size =", self.size)

        try:
            # Redraw using expanded pitch mapper
            self.full_panel.redraw_notes(self.time_grid)
            print("DEBUG: full_panel.redraw_notes() succeeded")
        except Exception as e:
            print("ERROR: full_panel.redraw_notes() failed:", e)

    def on_open(self):
        print("DEBUG: Popup opened")
# ui/widgets/expanded_piano_roll.py

from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.metrics import dp
from kivy.clock import Clock

from ui.widgets.piano_roll import PianoRollPanel
from recording.note_event import NoteEvent


class ExpandedPianoRollPopup(Popup):
    def __init__(self, piano_roll, time_grid, pitch_mapper, **kwargs):
        print("DEBUG: ExpandedPianoRollPopup __init__ called")
        super().__init__(**kwargs)

        self.title = "Full Piano Roll"
        self.size_hint = (0.95, 0.95)
        self.auto_dismiss = False
        self.background_color = (0, 0, 0, 0.85)

        # References to minimized panel + shared systems
        self.original_panel = piano_roll
        self.time_grid = time_grid
        self.pitch_mapper = pitch_mapper

        # Root layout
        root = BoxLayout(orientation="vertical", size_hint=(1, 1))
        print("DEBUG: Created root BoxLayout")

        # Create a NEW expanded panel (do NOT share canvas)
        self.full_panel = PianoRollPanel(
            expanded=True,
            size_hint=(1, 1),
            height=dp(800)
        )
        print("DEBUG: Created full_panel:", self.full_panel)

        # Copy notes + scroll height from minimized panel
        self._copy_notes_from_original()

        # Attach time grid + pitch mapper
        self.full_panel.time_grid = self.time_grid
        self.full_panel.pitch_mapper = self.pitch_mapper
        print("DEBUG: Shared time_grid and pitch_mapper")

        root.add_widget(self.full_panel)
        print("DEBUG: Added full_panel to root")

        self.add_widget(root)
        print("DEBUG: Added root to popup")

        # Finish setup after layout
        Clock.schedule_once(self._finish_setup, 0)

    # -----------------------------------------------------------
    # Copy notes + scroll height from minimized panel
    # -----------------------------------------------------------
    def _copy_notes_from_original(self):
        print("DEBUG: Copying notes from minimized panel")

        # Convert dicts or objects → NoteEvent objects
        self.full_panel.note_canvas.notes = [
            NoteEvent(
                pitch=n.pitch if hasattr(n, "pitch") else n["pitch"],
                start=n.start if hasattr(n, "start") else n["start"],
                end=n.end if hasattr(n, "end") else n["end"],
                velocity=getattr(n, "velocity", 1.0),
                hand=getattr(n, "hand", None),
                finger=getattr(n, "finger", None)
            )
            for n in self.original_panel.note_canvas.notes
        ]

        # Copy scrollable height
        self.full_panel.note_canvas.height = self.original_panel.note_canvas.height

        print(f"DEBUG: Copied {len(self.full_panel.note_canvas.notes)} notes")

    # -----------------------------------------------------------
    # Final redraw after layout
    # -----------------------------------------------------------
    def _finish_setup(self, dt):
        print("DEBUG: _finish_setup called")
        print("DEBUG: full_panel size =", self.full_panel.size)
        print("DEBUG: popup size =", self.size)

        try:
            # Redraw using expanded pitch mapper
            self.full_panel.redraw_notes(self.time_grid)
            print("DEBUG: full_panel.redraw_notes() succeeded")
        except Exception as e:
            print("ERROR: full_panel.redraw_notes() failed:", e)

    def on_open(self):
        print("DEBUG: Popup opened")

