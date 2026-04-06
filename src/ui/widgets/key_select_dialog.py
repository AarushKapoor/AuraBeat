from kivy.uix.popup import Popup
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.graphics import Color, Rectangle
from kivy.metrics import dp
from kivy.properties import StringProperty, ObjectProperty


class KeySelectDialog(Popup):
    hand = StringProperty()
    finger = StringProperty()
    pitch_mapper = ObjectProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title = ""  # empty string is allowed
        self.title_size = 0  # hides the title bar height
        self.separator_height = 0  # removes the blue separator line
        #self.background = ""  # removes default popup background image

        self.size_hint = (0.95, 0.40)

        root = FloatLayout()

        # ---------------------------------------------------------
        # LABEL ABOVE KEYBOARD
        # ---------------------------------------------------------
        lbl = Label(
            text=f"[b]{self.hand} – Finger {self.finger}[/b]",
            markup=True,
            size_hint=(1, None),
            height=dp(40),
            pos_hint={"top": 1},
            halign="center",
            valign="middle",
        )
        lbl.bind(size=lambda *_: setattr(lbl, "text_size", lbl.size))
        root.add_widget(lbl)

        # ---------------------------------------------------------
        # KEYBOARD AREA
        # ---------------------------------------------------------
        kb = KeyboardWidget(
            size_hint=(1, None),
            height=dp(160),
            pos_hint={"x": 0, "y": 0},
            on_key_press=self._apply
        )
        root.add_widget(kb)

        self.add_widget(root)

    # ---------------------------------------------------------
    # CALLBACK
    # ---------------------------------------------------------
    def _apply(self, midi_pitch):
        if self.pitch_mapper:
            self.pitch_mapper.set_custom_pitch(self.hand, self.finger, midi_pitch)

        # Update the overlay immediately
        from kivy.app import App
        app = App.get_running_app()
        if app and app.root and "overlay" in app.root.ids:
            app.root.ids["overlay"].refresh_labels()

        self.dismiss()


# =====================================================================
#  CUSTOM KEYBOARD WIDGET (clean aesthetic, no labels)
# =====================================================================

class KeyboardWidget(FloatLayout):
    def __init__(self, on_key_press=None, **kwargs):
        super().__init__(**kwargs)
        self.on_key_press = on_key_press
        self.selected_midi = None  # <-- NEW
        self.bind(pos=self.redraw, size=self.redraw)

    def _note_name(self, midi):
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        octave = (midi // 12) - 1
        return f"{names[midi % 12]}{octave}"

    def _is_black(self, midi):
        return "#" in self._note_name(midi)[:-1]

    def _select(self, midi):
        self.selected_midi = midi
        if self.on_key_press:
            self.on_key_press(midi)
        self.redraw()

    def redraw(self, *args):
        self.canvas.clear()
        self.clear_widgets()

        x0, y0 = self.x, self.y
        W, H = self.width, self.height

        if W <= 0 or H <= 0:
            return

        white_midis = [m for m in range(21, 109) if not self._is_black(m)]
        num_white = len(white_midis)

        white_w = W / float(num_white)
        white_h = H
        black_w = white_w * 0.60
        black_h = H * 0.60

        white_positions = {}

        # ---------------------------------------------------------
        # 1. Draw white keys
        # ---------------------------------------------------------
        with self.canvas:
            # Base white keys
            Color(0.92, 0.94, 0.97, 1)
            Rectangle(pos=(x0, y0), size=(W, H))

            # Separators
            Color(0.07, 0.08, 0.10, 0.9)
            for i in range(1, num_white):
                sx = x0 + i * white_w
                Rectangle(pos=(sx - 1, y0), size=(2, H))

        # ---------------------------------------------------------
        # 2. Add white buttons + highlight
        # ---------------------------------------------------------
        for idx, midi in enumerate(white_midis):
            bx = x0 + idx * white_w
            white_positions[midi] = bx

            # Highlight if selected
            if midi == self.selected_midi:
                with self.canvas:
                    Color(0.3, 0.55, 1.0, 0.35)  # soft blue overlay
                    Rectangle(pos=(bx, y0), size=(white_w, white_h))

            # Invisible button
            btn = Button(
                background_color=(0, 0, 0, 0),
                background_normal="",
                background_down="",
                size_hint=(None, None),
                pos=(bx, y0),
                width=white_w,
                height=white_h,
            )
            btn.bind(on_release=lambda b, m=midi: self._select(m))
            self.add_widget(btn)

        # ---------------------------------------------------------
        # 3. Draw black keys
        # ---------------------------------------------------------
        black_rects = []

        with self.canvas:
            Color(0.06, 0.06, 0.09, 1)

            for midi in range(21, 109):
                if not self._is_black(midi):
                    continue

                # Find nearest lower white key
                lower = midi - 1
                while lower >= 21 and self._is_black(lower):
                    lower -= 1

                # Find nearest upper white key
                upper = midi + 1
                while upper <= 108 and self._is_black(upper):
                    upper += 1

                if lower not in white_positions or upper not in white_positions:
                    continue

                lx = white_positions[lower]
                ux = white_positions[upper]

                center = (lx + ux + white_w) / 2.0
                bx = center - black_w / 2.0
                by = y0 + white_h - black_h

                # Draw black key
                Rectangle(pos=(bx, by), size=(black_w, black_h))

                # Store for button placement
                black_rects.append((midi, bx, by))

        # ---------------------------------------------------------
        # 4. Add black buttons + highlight
        # ---------------------------------------------------------
        for midi, bx, by in black_rects:

            # Highlight if selected
            if midi == self.selected_midi:
                with self.canvas:
                    Color(1, 1, 1, 0.25)  # light overlay for black keys
                    Rectangle(pos=(bx, by), size=(black_w, black_h))

            btn = Button(
                background_color=(0, 0, 0, 0),
                background_normal="",
                background_down="",
                size_hint=(None, None),
                pos=(bx, by),
                width=black_w,
                height=black_h,
            )
            btn.bind(on_release=lambda b, m=midi: self._select(m))
            self.add_widget(btn)
