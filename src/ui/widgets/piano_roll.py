# src/ui/widgets/piano_roll.py
from kivy.uix.widget import Widget
from kivy.uix.scrollview import ScrollView
from kivy.graphics import (
    Color, Rectangle, Line, InstructionGroup
)
from kivy.properties import (
    NumericProperty, BooleanProperty, ColorProperty
)
from kivy.metrics import dp
from kivy.graphics.texture import Texture
import numpy as np


# ============================================================
#  NOTE CANVAS (scrollable falling-note surface)
# ============================================================

class NoteCanvas(Widget):
    """
    Scrollable drawing surface for notes.
    Height is controlled externally (TimeGrid).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.notes = []

        with self.canvas:
            Color(0, 0, 0, 0)
            self.bg = Rectangle(pos=self.pos, size=self.size)

        self.bind(pos=self._update_bg, size=self._update_bg)

    def _update_bg(self, *_):
        self.bg.pos = self.pos
        self.bg.size = self.size

    def redraw_notes(self, time_grid, pitch_to_x):
        self.canvas.clear()

        with self.canvas:
            Color(0, 0, 0, 0)
            Rectangle(pos=self.pos, size=self.size)

        canvas_h = self.height

        for note in self.notes:
            pitch = note.pitch if hasattr(note, "pitch") else note["pitch"]
            start = note.start if hasattr(note, "start") else note["start"]
            end   = note.end   if hasattr(note, "end")   else note["end"]

            h = time_grid.duration_to_height(end - start)
            y_end = time_grid.time_to_y(end)

            # Flip: notes fall from top down toward the strike line
            y = canvas_h - y_end
            x = pitch_to_x(pitch, self.width)

            with self.canvas:
                Color(0.2, 0.7, 1.0, 1.0)
                Rectangle(pos=(self.x + x, self.y + y), size=(dp(12), h))

    def add_note(self, *, x, y, height, pitch=None, start_ms=None, end_ms=None, color=(0.2, 0.7, 1.0, 1.0)):
        width = dp(12)

        self.notes.append({
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "pitch": pitch,
            "start": start_ms,
            "end": end_ms,
        })

        with self.canvas:
            Color(*color)
            Rectangle(pos=(x, y), size=(width, height))


# ============================================================
#  PIANO ROLL PANEL (compact + expanded)
# ============================================================

class PianoRollPanel(Widget):
    """
    Synthesia-style piano roll:
    - Compact 7-key keyboard in normal mode
    - Full 88-key keyboard in expanded popup mode
    Notes fall downward toward the strike line.
    """
    expanded = BooleanProperty(False)

    keyboard_height_ratio = NumericProperty(0.18)
    show_chevrons = BooleanProperty(False)

    strike_line_color = ColorProperty((0x32/255., 0xCD/255., 0x32/255., 1.0))
    strike_line_thickness_dp = NumericProperty(dp(2))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.scroll = ScrollView(
            size_hint=(1, 1),
            bar_width=0,
            do_scroll_x=False,
            do_scroll_y=True,
            effect_cls='ScrollEffect'
        )

        self.note_canvas = NoteCanvas(size_hint=(1, None))
        self.scroll.add_widget(self.note_canvas)
        self.add_widget(self.scroll)

        self._g = InstructionGroup()
        self.canvas.add(self._g)
        self._border_tex = None

        self.bind(
            pos=self._redraw, size=self._redraw,
            expanded=self._redraw,
            keyboard_height_ratio=self._redraw,
            show_chevrons=self._redraw,
            strike_line_color=self._redraw,
            strike_line_thickness_dp=self._redraw
        )

    # ============================================================
    #  PUBLIC API
    # ============================================================

    @property
    def strike_line_y(self):
        kb_h = max(dp(40), self.height * float(self.keyboard_height_ratio))
        return self.y + kb_h

    def set_canvas_height(self, height_px):
        self.note_canvas.height = height_px

    def redraw_notes(self, time_grid, pitch_to_x=None):
        mapper = self.get_pitch_to_x() if pitch_to_x is None else pitch_to_x
        self.note_canvas.redraw_notes(time_grid, mapper)

    # ============================================================
    #  INTERNAL RENDERING
    # ============================================================

    def _ensure_border_texture(self):
        if self._border_tex is not None:
            return

        h, w = 256, 8
        y = np.linspace(0.0, 1.0, h, dtype=np.float32)
        alpha_line = 1.0 - np.abs(2.0 * y - 1.0)

        arr = np.zeros((h, w, 4), dtype=np.uint8)
        arr[..., 0:3] = 255
        arr[..., 3] = (alpha_line[:, None] * 0.85 * 255).astype(np.uint8)
        arr = np.flipud(arr)

        tex = Texture.create(size=(w, h))
        tex.blit_buffer(arr.tobytes(), colorfmt="rgba", bufferfmt="ubyte")
        tex.wrap = 'clamp_to_edge'
        self._border_tex = tex

    def _redraw(self, *args):
        self._g.clear()
        self._ensure_border_texture()

        x0, y0 = self.x, self.y
        W, H = self.width, self.height

        kb_h = max(dp(40), H * float(self.keyboard_height_ratio))
        track_y = y0 + kb_h
        track_h = max(0, H - kb_h)

        # ScrollView sits above the keyboard
        self.scroll.pos = (x0, track_y)
        self.scroll.size = (W, track_h)

        self.note_canvas.width = W

        # Panel background
        self._g.add(Color(0x12/255., 0x1C/255., 0x2A/255., 0.72))
        self._g.add(Rectangle(pos=(x0, y0), size=(W, H)))

        # Feathered vertical accents
        if H > 0:
            accent_len = max(dp(30), H / 3.0)
            yc = y0 + H / 2.0
            accent_y = yc - accent_len / 2.0
            razor = max(1.0, dp(1.2))

            self._g.add(Color(1, 1, 1, 1))
            self._g.add(Rectangle(
                pos=(x0, accent_y),
                size=(razor, accent_len),
                texture=self._border_tex
            ))
            self._g.add(Rectangle(
                pos=(x0 + W - razor, accent_y),
                size=(razor, accent_len),
                texture=self._border_tex
            ))

        # Keyboard background
        self._g.add(Color(0x12/255., 0x1C/255., 0x2A/255., 0.72))
        self._g.add(Rectangle(pos=(x0, y0), size=(W, kb_h)))

        # Strike line
        line_th = float(self.strike_line_thickness_dp)
        self._g.add(Color(*self.strike_line_color))
        self._g.add(Rectangle(
            pos=(x0, y0 + kb_h - line_th / 2.0),
            size=(W, line_th)
        ))

        # ============================================================
        #  KEYBOARD RENDERING
        # ============================================================

        if not self.expanded:
            # Compact 7-key keyboard
            key_w = W / 7.0
            kb_color = (0.90, 0.92, 0.96, 1)

            self._g.add(Color(*kb_color))
            self._g.add(Rectangle(pos=(x0, y0), size=(W, kb_h)))

            # Separators
            self._g.add(Color(0.07, 0.08, 0.10, 0.90))
            sep_w = max(1, int(round(dp(1))))
            for i in range(1, 7):
                sep_x = x0 + i * key_w
                sep_x_int = int(round(sep_x))
                self._g.add(Rectangle(
                    pos=(sep_x_int - sep_w // 2, y0),
                    size=(sep_w, kb_h)
                ))

            # Black keys
            black_boundaries = [0, 1, 3, 4, 5]
            bw = key_w * 0.56
            bh = kb_h * 0.62

            self._g.add(Color(0.06, 0.06, 0.09, 1))
            for j in black_boundaries:
                cx = x0 + (j + 1) * key_w
                bx = int(round(cx - bw / 2.0))
                by = y0 + kb_h - bh
                self._g.add(Rectangle(
                    pos=(bx, by),
                    size=(int(round(bw)), bh)
                ))

        else:
            # Full 88-key keyboard
            WHITE_KEY_COUNT = 52
            BLACK_KEY_OFFSETS = [1, 3, 6, 8, 10]

            white_w = W / float(WHITE_KEY_COUNT)

            # White keys
            self._g.add(Color(0.90, 0.92, 0.96, 1))
            for i in range(WHITE_KEY_COUNT):
                wx = x0 + i * white_w
                self._g.add(Rectangle(pos=(wx, y0), size=(white_w, kb_h)))

            # Separators
            self._g.add(Color(0.07, 0.08, 0.10, 0.90))
            sep_w = max(1, int(round(dp(1))))
            for i in range(1, WHITE_KEY_COUNT):
                sx = x0 + i * white_w
                self._g.add(Rectangle(
                    pos=(sx - sep_w // 2, y0),
                    size=(sep_w, kb_h)
                ))

            # Black keys
            self._g.add(Color(0.06, 0.06, 0.09, 1))
            black_w = white_w * 0.65
            black_h = kb_h * 0.62

            white_index = 0
            for midi in range(21, 109):
                semitone = midi % 12
                if semitone in BLACK_KEY_OFFSETS:
                    bx = x0 + white_index * white_w - black_w / 2.0
                    by = y0 + kb_h - black_h
                    self._g.add(Rectangle(pos=(bx, by), size=(black_w, black_h)))
                else:
                    white_index += 1

    # ============================================================
    #  PITCH → X MAPPERS
    # ============================================================

    def pitch_to_x_compact(self, pitch, width):
        key_w = width / 7.0
        scale_index = pitch % 7
        return scale_index * key_w

    def pitch_to_x_expanded(self, pitch, width):
        WHITE_KEY_COUNT = 52
        BLACK_KEY_OFFSETS = [1, 3, 6, 8, 10]

        white_w = width / float(WHITE_KEY_COUNT)

        white_index = 0
        for midi in range(21, pitch + 1):
            semitone = midi % 12
            if semitone not in BLACK_KEY_OFFSETS:
                white_index += 1

        return white_index * white_w

    def get_pitch_to_x(self):
        return self.pitch_to_x_expanded if self.expanded else self.pitch_to_x_compact

    # ============================================================
    #  SCROLL REGION UPDATE
    # ============================================================

    def update_scroll_region(self):
        if not self.note_canvas.notes:
            self.note_canvas.height = self.scroll.height + dp(200)
            return

        max_y = 0
        for n in self.note_canvas.notes:
            start = n.start if hasattr(n, "start") else n["start"]
            end   = n.end   if hasattr(n, "end")   else n["end"]
            y_end = self.time_grid.time_to_y(end)
            max_y = max(max_y, y_end)

        min_height = self.scroll.height + dp(200)
        target_h = max_y + dp(200)

        self.note_canvas.height = max(target_h, min_height)

        # Start scrolled to top so notes fall into view from above
        self.scroll.scroll_y = 1.0