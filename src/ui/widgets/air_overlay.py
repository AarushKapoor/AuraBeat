# src/ui/widgets/air_overlay.py
from __future__ import annotations

from kivy.uix.widget import Widget
from kivy.properties import (
    DictProperty, BooleanProperty, NumericProperty, ListProperty, ColorProperty
)
from kivy.graphics import Color, InstructionGroup, Ellipse, Line
from kivy.uix.label import Label
from kivy.metrics import dp
from ui.widgets.finger_label import FingerLabel


FINGERS_ORDER = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


class AirOverlayPanel(Widget):
    """
    Middle column air keyboard: 10 dots laid out horizontally.
    (Full implementation preserved exactly from original file.)
    """

    # --- Inputs from controller ---
    left_labels = DictProperty({})
    right_labels = DictProperty({})
    left_pressed = DictProperty({})
    right_pressed = DictProperty({})
    left_fist = BooleanProperty(False)
    left_thumbup = BooleanProperty(False)
    right_fist = BooleanProperty(False)
    right_thumbup = BooleanProperty(False)

    left_present = BooleanProperty(False)
    right_present = BooleanProperty(False)

    # --- Horizontal spacing ---
    dot_spacing = NumericProperty(None)
    horizontal_margin_dp = NumericProperty(dp(24))
    spacing_factor = NumericProperty(0.90)
    mid_gap_factor = NumericProperty(1.4)

    left_gap_weights = ListProperty([0.78, 0.92, 1.04, 0.95])
    right_gap_weights = ListProperty([0.78, 0.92, 1.04, 0.95])
    mirror_right_from_left = BooleanProperty(True)

    # --- Colors ---
    base_color = ColorProperty((0x4E/255., 0x64/255., 0x94/255., 1.0))
    pressed_color = ColorProperty((0xF0/255., 0x8C/255., 0xFF/255., 1.0))
    sustain_ring = BooleanProperty(True)

    # --- Sizes ---
    min_dot_diam_dp = NumericProperty(dp(14))
    max_dot_diam_dp = NumericProperty(dp(96))
    label_offset_dp = NumericProperty(dp(8))

    # --- Arc controls ---
    left_edge_start_y = NumericProperty(0.55)
    left_center_y = NumericProperty(0.80)
    left_edge_end_y = NumericProperty(0.67)

    lock_right_arc_to_left = BooleanProperty(True)
    right_edge_start_y = NumericProperty(0.67)
    right_center_y = NumericProperty(0.80)
    right_edge_end_y = NumericProperty(0.55)

    # --- Presence dots ---
    presence_dot_color = ColorProperty((0.20, 1.00, 0.75, 1.0))
    presence_dot_radius_dp = NumericProperty(dp(4))
    presence_dot_offset_dp = NumericProperty(dp(12))

    # --- Vertical centering ---
    auto_center_vertical = BooleanProperty(True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._g = InstructionGroup()
        self.canvas.add(self._g)

        # Bind everything to redraw
        for prop in (
            "pos", "size",
            "left_labels", "right_labels",
            "left_pressed", "right_pressed",
            "left_fist", "left_thumbup",
            "right_fist", "right_thumbup",
            "dot_spacing", "horizontal_margin_dp",
            "spacing_factor", "mid_gap_factor",
            "left_gap_weights", "right_gap_weights",
            "mirror_right_from_left",
            "base_color", "pressed_color", "sustain_ring",
            "min_dot_diam_dp", "max_dot_diam_dp",
            "label_offset_dp",
            "left_edge_start_y", "left_center_y", "left_edge_end_y",
            "right_edge_start_y", "right_center_y", "right_edge_end_y",
            "lock_right_arc_to_left",
            "left_present", "right_present",
            "presence_dot_color", "presence_dot_radius_dp", "presence_dot_offset_dp",
            "auto_center_vertical",
        ):
            self.bind(**{prop: self._redraw})

    # ---------------------------------------------------------
    # NEW: Refresh labels after popup mapping change
    # ---------------------------------------------------------
    def refresh_labels(self):
        """
        Pull updated labels from the pitch mapper and redraw.
        """
        app = self.get_app()
        if not app or not app.pitch_mapper:
            return

        mapper = app.pitch_mapper

        # Import the global midi_to_name function
        from mapping.scale_window import midi_to_name

        left = {}
        right = {}

        for finger in FINGERS_ORDER:
            # LEFT HAND
            midi = mapper.get_pitch("left", finger)
            if midi is None:
                left[finger] = "—"
            else:
                left[finger] = midi_to_name(midi)

            # RIGHT HAND
            midi = mapper.get_pitch("right", finger)
            if midi is None:
                right[finger] = "—"
            else:
                right[finger] = midi_to_name(midi)

        # Trigger redraw using existing update_model()
        self.update_model(
            left_labels=left,
            left_pressed=self.left_pressed,
            right_labels=right,
            right_pressed=self.right_pressed,
            left_fist=self.left_fist,
            left_thumbup=self.left_thumbup,
            right_fist=self.right_fist,
            right_thumbup=self.right_thumbup,
            left_present=self.left_present,
            right_present=self.right_present,
        )

    # ---------------------------------------------------------

    def update_model(
        self,
        left_labels,
        left_pressed,
        right_labels,
        right_pressed,
        left_fist=False,
        left_thumbup=False,
        right_fist=False,
        right_thumbup=False,
        left_present=False,
        right_present=False,
    ):
        self.left_labels = left_labels or {}
        self.right_labels = right_labels or {}
        self.left_pressed = left_pressed or {}
        self.right_pressed = right_pressed or {}
        self.left_fist = bool(left_fist)
        self.left_thumbup = bool(left_thumbup)
        self.right_fist = bool(right_fist)
        self.right_thumbup = bool(right_thumbup)
        self.left_present = bool(left_present)
        self.right_present = bool(right_present)

    @staticmethod
    def _quad_bezier_y(u, y0, yc, y1):
        u = max(0.0, min(1.0, float(u)))
        om = 1.0 - u
        return om * om * y0 + 2.0 * om * u * yc + u * u * y1

    def _redraw(self, *args):
        self._g.clear()
        x0, y0 = self.x, self.y
        W, H = self.width, self.height

        # Remove old labels to prevent accumulation
        for c in list(self.children):
            if getattr(c, "_air_label", False):
                self.remove_widget(c)

        # Visual order
        names_left_vis = ["Pinky", "Ring", "Middle", "Index", "Thumb"]
        names_right_vis = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

        all_labels = [self.left_labels.get(nm, "—") for nm in names_left_vis] + \
                     [self.right_labels.get(nm, "—") for nm in names_right_vis]
        all_pressed = [bool(self.left_pressed.get(nm, False)) for nm in names_left_vis] + \
                      [bool(self.right_pressed.get(nm, False)) for nm in names_right_vis]

        # ---------------- Horizontal positions ----------------
        margin = float(self.horizontal_margin_dp)
        usable_w = max(0.0, W - 2.0 * margin)
        N = 10

        if N <= 1 or usable_w <= 0:
            xs = [x0 + W / 2.0] * N
        else:
            Lw = list(self.left_gap_weights or [1, 1, 1, 1])
            if self.mirror_right_from_left:
                Rw = list(reversed(Lw))
            else:
                Rw = list(self.right_gap_weights or [1, 1, 1, 1])

            Lsum = sum(float(w) for w in Lw)
            Rsum = sum(float(w) for w in Rw)
            effective_gaps = Lsum + float(self.mid_gap_factor) + Rsum

            if self.dot_spacing is not None:
                s = float(self.dot_spacing) * float(self.spacing_factor)
                total_span = s * effective_gaps
            else:
                s = (usable_w * float(self.spacing_factor)) / effective_gaps
                total_span = s * effective_gaps

            left_start = x0 + (W - total_span) / 2.0

            xs = []
            acc = left_start

            for i in range(5):
                xs.append(acc)
                if i < 4:
                    acc += s * float(Lw[i])

            acc += s * float(self.mid_gap_factor)

            for i in range(5):
                xs.append(acc)
                if i < 4:
                    acc += s * float(Rw[i])

        # ---------------- Dot size ----------------
        diam = max(float(self.min_dot_diam_dp),
                   min(float(self.max_dot_diam_dp), H / 12.0 if H > 0 else float(self.min_dot_diam_dp)))
        r = diam / 2.0

        # ---------------- Vertical positions ----------------
        if self.lock_right_arc_to_left:
            right_edge_start_y = self.left_edge_end_y
            right_center_y = self.left_center_y
            right_edge_end_y = self.left_edge_start_y
        else:
            right_edge_start_y = self.right_edge_start_y
            right_center_y = self.right_center_y
            right_edge_end_y = self.right_edge_end_y

        y_norms = []
        for i in range(5):
            u = i / 4.0
            y_norm = self._quad_bezier_y(u, self.left_edge_start_y, self.left_center_y, self.left_edge_end_y)
            y_norms.append(max(0.0, min(1.0, y_norm)))

        for j in range(5):
            u = j / 4.0
            y_norm = self._quad_bezier_y(u, right_edge_start_y, right_center_y, right_edge_end_y)
            y_norms.append(max(0.0, min(1.0, y_norm)))

        if self.auto_center_vertical and len(y_norms) == 10:
            mean_y = sum(y_norms) / 10.0
            delta = 0.5 - mean_y
            for k in range(10):
                y_norms[k] = max(0.0, min(1.0, y_norms[k] + delta))

        ys = [y0 + yn * H for yn in y_norms]

        # ---------------- Draw dots + labels ----------------
        pr = float(self.presence_dot_radius_dp)
        poff = float(self.presence_dot_offset_dp)
        for i in range(N):
            cx, cy = xs[i], ys[i]
            on = all_pressed[i]
            color = self.pressed_color if on else self.base_color

            self._g.add(Color(*color))
            self._g.add(Ellipse(pos=(cx - r, cy - r), size=(2 * r, 2 * r)))

            is_left_hand = (i < 5)
            if ((self.left_present and is_left_hand) or (self.right_present and not is_left_hand)) and pr > 0.0:
                extra = r * 0.15
                py = max(y0 + pr, cy - r - (poff + extra))
                self._g.add(Color(*self.presence_dot_color))
                self._g.add(Ellipse(pos=(cx - pr, py - pr), size=(2 * pr, 2 * pr)))

            if self.sustain_ring and on:
                self._g.add(Color(color[0], color[1], color[2], 0.25))
                self._g.add(Line(circle=(cx, cy, r + dp(4)), width=dp(1.6)))

            label_text = str(all_labels[i]) if all_labels[i] else "—"

            if i < 5:
                hand = "left"
                finger = ["Pinky", "Ring", "Middle", "Index", "Thumb"][i]
            else:
                hand = "right"
                finger = ["Thumb", "Index", "Middle", "Ring", "Pinky"][i - 5]

            lbl = FingerLabel(
                text=label_text,
                font_size=dp(14),
                color=(0.92, 0.95, 1, 1),
                size_hint=(None, None),
                size=(dp(90), dp(22)),
                pos=(cx - dp(45), cy + r + float(self.label_offset_dp)),
                halign="center",
                valign="middle",
                hand=hand,
                finger=finger,
            )
            lbl.text_size = lbl.size
            lbl._air_label = True

            lbl.bind(on_release=lambda w: self._on_finger_label_clicked(w.hand, w.finger))

            self.add_widget(lbl)

    def _on_finger_label_clicked(self, hand, finger):
        app = self.get_app()
        if app and hasattr(app, "open_finger_mapping_dialog"):
            app.open_finger_mapping_dialog(hand, finger)

    def get_app(self):
        from kivy.app import App
        return App.get_running_app()
