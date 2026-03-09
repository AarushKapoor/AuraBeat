# src/ui/widgets.py
from __future__ import annotations

from typing import Dict

import numpy as np
from kivy.uix.relativelayout import RelativeLayout

from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label

from kivy.properties import (
    DictProperty,
    BooleanProperty,
    NumericProperty,
)

from kivy.metrics import dp
from kivy.clock import Clock
from kivy.animation import Animation

from kivy.graphics import (
    Color, Rectangle, InstructionGroup, Line, Ellipse, RoundedRectangle
)
from kivy.graphics.texture import Texture


# -----------------------------------------------------------------------------
# Root container
# -----------------------------------------------------------------------------
class RootView(BoxLayout):
    """Top-level Kivy container used by KV."""
    pass


# src/ui/widgets.py (only the VideoFeed class changes shown)
from kivy.properties import BooleanProperty

class VideoFeed(Image):
    """Image widget with rounded corners using a stencil mask."""
    draw_border = BooleanProperty(True)   # <-- make it configurable from KV

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.corner_radius_dp = dp(14)

        self._pre = InstructionGroup()
        self.canvas.before.add(self._pre)

        self._post = InstructionGroup()
        self.canvas.after.add(self._post)

        self._texture = None
        # Rebuild stencil and border when geometry or draw_border changes
        self.bind(pos=self._rebuild_stencil, size=self._rebuild_stencil, draw_border=self._rebuild_stencil)
        self._rebuild_stencil()

    def _rebuild_stencil(self, *args):
        self._pre.clear()
        self._post.clear()

        from kivy.graphics import StencilPush, StencilUse, StencilUnUse, StencilPop

        # Stencil start
        self._pre.add(StencilPush())
        self._pre.add(Color(1, 1, 1, 1))
        self._mask = RoundedRectangle(
            pos=self.pos,
            size=self.size,
            radius=[self.corner_radius_dp] * 4
        )
        self._pre.add(self._mask)
        self._pre.add(StencilUse())

        # Stencil end
        self._post.add(StencilUnUse())
        self._post.add(Color(1, 1, 1, 1))
        self._post.add(StencilPop())

        # Optional border line atop — render only if draw_border is True
        if self.draw_border:
            self._post.add(Color(1, 1, 1, 0.18))
            self._post.add(Line(
                rounded_rectangle=(self.x, self.y, self.width, self.height, self.corner_radius_dp),
                width=dp(1.2))
            )

    def set_frame(self, rgb_frame: np.ndarray):
        """Accepts an RGB numpy array and uploads to a Kivy texture."""
        # Safety guards
        if rgb_frame is None or not hasattr(rgb_frame, "shape") or len(rgb_frame.shape) != 3:
            return

        h, w = rgb_frame.shape[:2]
        if h <= 1 or w <= 1:
            return

        # (Re)create texture if dimensions changed or first time
        if (self._texture is None) or (self._texture.width != w) or (self._texture.height != h):
            tex = Texture.create(size=(w, h))
            tex.flip_vertical()
            self._texture = tex

        self.texture = self._texture
        self.texture.blit_buffer(rgb_frame.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
        self.canvas.ask_update()


# -----------------------------------------------------------------------------
# CircleButton: minimal circular gear button
# -----------------------------------------------------------------------------
class CircleButton(Button):
    """Minimal circular button (e.g., gear)."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_normal = ""
        self.background_down = ""
        self.background_color = (0, 0, 0, 0)
        self.color = (0.9, 0.95, 1, 1)
        self.font_size = dp(20)
        self.bold = True

        self._bg_instr = InstructionGroup()
        self.canvas.before.add(self._bg_instr)

        self.bind(pos=self._redraw_bg, size=self._redraw_bg)
        self._redraw_bg()

    def _redraw_bg(self, *args):
        self._bg_instr.clear()
        r = max(1.0, min(self.width, self.height) / 2.0)

        # Base disk
        self._bg_instr.add(Color(0.12, 0.14, 0.18, 1))
        self._bg_instr.add(Ellipse(pos=(self.x, self.y), size=(2*r, 2*r)))

        # Subtle border ring
        self._bg_instr.add(Color(1, 1, 1, 0.18))
        self._bg_instr.add(Line(circle=(self.center_x, self.center_y, r), width=dp(1.2)))


# -----------------------------------------------------------------------------
# QuickMenu: rounded panel with vertical items; show/hide via .visible
# -----------------------------------------------------------------------------
class QuickMenu(Widget):
    """Rounded rectangular quick menu that appears next to the gear button."""
    visible = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._bg_instr = InstructionGroup()
        self.canvas.before.add(self._bg_instr)

        self._items = BoxLayout(
            orientation="vertical",
            spacing=dp(6),
            padding=[dp(10), dp(10), dp(10), dp(10)]
        )
        self.add_widget(self._items)

        # Placeholder actions – replace with real menu items
        for i in range(1, 6):
            btn = Button(text=f"Placeholder {i}", size_hint_y=None, height=dp(36))
            btn.background_normal = ""
            btn.background_down = ""
            btn.background_color = (0.18, 0.20, 0.25, 1)
            btn.color = (0.9, 0.95, 1, 1)
            self._items.add_widget(btn)

        self.bind(pos=self._redraw_bg, size=self._redraw_bg, visible=self._apply_visibility)
        self._apply_visibility()

    def _apply_visibility(self, *args):
        self.opacity = 1.0 if self.visible else 0.0
        self.disabled = not self.visible

    def _redraw_bg(self, *args):
        self._bg_instr.clear()

        # Background panel with rounded corners
        self._bg_instr.add(Color(0.13, 0.14, 0.18, 0.98))
        self._bg_instr.add(RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(12)] * 4))

        # Layout internal items
        self._items.pos = (self.x, self.y)
        self._items.size = (self.width, self.height)


# -----------------------------------------------------------------------------
# GestureHUD: badges with icons + labels; fades out after inactivity
# -----------------------------------------------------------------------------
GESTURE_META: Dict[str, Dict[str, object]] = {
    "Open Hand": {"icon": "🖐", "rgb_f": (0.24, 0.82, 0.24)},
    "Fist": {"icon": "✊", "rgb_f": (0.98, 0.55, 0.20)},
    "Point": {"icon": "☝️", "rgb_f": (0.16, 0.67, 0.90)},
    "Thumbs Up": {"icon": "👍", "rgb_f": (0.16, 0.82, 0.82)},
    "": {"icon": "❓", "rgb_f": (0.55, 0.55, 0.55)},
}


class GestureHUD(Widget):
    """
    A compact heads-up display that sits below the video preview.
    Shows one badge per detected hand with:
      - A colored glow/fill that matches the gesture
      - An emoji icon + gesture name label
      - A smooth fade-out after FADE_DELAY seconds of no detections
    Call `update_gestures(gestures: list[str])` from the UI thread.
    """

    FADE_DELAY = 1.8   # seconds before fading when hands disappear
    BADGE_H = 38       # row height
    BADGE_PAD_X = 12
    BADGE_PAD_Y = 6
    MAX_HANDS = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._gestures: list[str] = []
        self._fade_event = None

        self._g = InstructionGroup()
        self.canvas.add(self._g)

        self.opacity = 0.0   # hidden until first gesture
        self.bind(pos=self._redraw, size=self._redraw)

    def update_gestures(self, gestures: list[str]):
        """Call on UI thread with the latest gesture list."""
        changed = (gestures != self._gestures)
        self._gestures = list(gestures)

        if self._fade_event:
            self._fade_event.cancel()
            self._fade_event = None

        if gestures:
            Animation.cancel_all(self, "opacity")
            self.opacity = 1.0
            if changed:
                self._redraw()
            self._fade_event = Clock.schedule_once(self._start_fade, self.FADE_DELAY)
        else:
            self._start_fade()

    def _start_fade(self, *args):
        Animation(opacity=0.0, duration=0.5, t="out_quad").start(self)

    def _redraw(self, *args):
        self._g.clear()
        if not self._gestures:
            return

        x0, y0 = self.x, self.y
        W = self.width
        row_h = dp(self.BADGE_H)
        total_h = row_h * len(self._gestures) + dp(self.BADGE_PAD_Y) * (len(self._gestures) + 1)

        # HUD background
        self._g.add(Color(0.09, 0.10, 0.13, 0.92))
        self._g.add(RoundedRectangle(
            pos=(x0, y0 + self.height - total_h),
            size=(W, total_h),
            radius=[dp(10)] * 4
        ))

        # Remove old labels before re-adding
        for child in list(self.children):
            self.remove_widget(child)

        # Per-gesture rows
        for i, gesture in enumerate(self._gestures):
            meta = GESTURE_META.get(gesture, GESTURE_META[""])
            r, g_c, b = meta["rgb_f"]

            row_y = y0 + self.height - dp(self.BADGE_PAD_Y) * (i + 1) - row_h * (i + 1)

            # Glow fill
            self._g.add(Color(r, g_c, b, 0.18))
            self._g.add(RoundedRectangle(
                pos=(x0 + dp(self.BADGE_PAD_X), row_y),
                size=(W - dp(self.BADGE_PAD_X) * 2, row_h),
                radius=[dp(7)] * 4
            ))

            # Accent stripe
            self._g.add(Color(r, g_c, b, 0.90))
            self._g.add(RoundedRectangle(
                pos=(x0 + dp(self.BADGE_PAD_X), row_y),
                size=(dp(4), row_h),
                radius=[dp(3)] * 4
            ))

            # Label (emoji-capable)
            icon = meta["icon"]
            name = gesture if gesture else "—"
            lbl = Label(
                text=f"[b]{icon}[/b]  [color={int(r*255):02x}{int(g_c*255):02x}{int(b*255):02x}ff]{name}[/color]"
                     f"  [color=606878ff]Hand {i + 1}[/color]",
                markup=True,
                font_size=dp(14),
                halign="left",
                valign="middle",
                size=(W - dp(self.BADGE_PAD_X) * 2 - dp(16), row_h),
                pos=(x0 + dp(self.BADGE_PAD_X) + dp(16), row_y),
            )
            lbl.text_size = lbl.size
            self.add_widget(lbl)


# -----------------------------------------------------------------------------
# PianoRollPanel: right-side column with dusk-gray bg and feathered accents
# -----------------------------------------------------------------------------


from kivy.uix.widget import Widget
from kivy.properties import NumericProperty, BooleanProperty, ColorProperty
from kivy.graphics import Color, Rectangle, InstructionGroup
from kivy.metrics import dp
import numpy as np
from kivy.graphics.texture import Texture


class PianoRollPanel(Widget):
    """Right-side piano roll panel (Synthesia-inspired), no chevrons."""
    keyboard_height_ratio = NumericProperty(0.18)
    show_chevrons = BooleanProperty(False)  # default off

    # NEW: strike zone line properties
    strike_line_color = ColorProperty((0x32/255., 0xCD/255., 0x32/255., 1.0))  # lime green #32CD32
    strike_line_thickness_dp = NumericProperty(dp(2))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._g = InstructionGroup()
        self.canvas.add(self._g)
        self._border_tex = None
        self.bind(
            pos=self._redraw, size=self._redraw,
            keyboard_height_ratio=self._redraw, show_chevrons=self._redraw,
            strike_line_color=self._redraw, strike_line_thickness_dp=self._redraw
        )

    def _ensure_border_texture(self):
        if self._border_tex is not None:
            return
        h, w = 256, 8
        y = np.linspace(0.0, 1.0, h, dtype=np.float32)
        alpha_line = 1.0 - np.abs(2.0 * y - 1.0)
        arr = np.zeros((h, w, 4), dtype=np.uint8)
        arr[..., 0:3] = 255
        arr[..., 3] = (alpha_line[:, None] * 0.85 * 255).astype(np.uint8)
        arr = np.flipud(arr)  # Flip for Kivy's buffer orientation

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

        # Panel background -> #121C2A (semi-transparent)
        self._g.add(Color(0x12/255., 0x1C/255., 0x2A/255., 0.72))
        self._g.add(Rectangle(pos=(x0, y0), size=(W, H)))

        # Feathered vertical accents (centered to FULL column height)
        if H > 0:
            accent_len = max(dp(30), H / 3.0)
            yc = y0 + H / 2.0                         # <-- full column center
            accent_y = yc - accent_len / 2.0
            razor = max(1.0, dp(1.2))

            self._g.add(Color(1, 1, 1, 1))
            self._g.add(Rectangle(pos=(x0, accent_y), size=(razor, accent_len), texture=self._border_tex))
            self._g.add(Rectangle(pos=(x0 + W - razor, accent_y), size=(razor, accent_len), texture=self._border_tex))

        # Bottom mini keyboard -> #121C2A (semi-transparent)
        self._g.add(Color(0x12/255., 0x1C/255., 0x2A/255., 0.72))
        self._g.add(Rectangle(pos=(x0, y0), size=(W, kb_h)))

        # === NEW: strike zone line (lime green) ===
        line_th = float(self.strike_line_thickness_dp)
        self._g.add(Color(*self.strike_line_color))
        # Place at the top edge of the keyboard; subtract half thickness for crisp alignment
        self._g.add(Rectangle(pos=(x0, y0 + kb_h - line_th / 2.0), size=(W, line_th)))

        # === White keys (7) — single white base + pixel-snapped separators ===
        key_w = W / 7.0
        kb_color = (0.90, 0.92, 0.96, 1)

        # 1) Fill the whole keyboard area in white once
        self._g.add(Color(*kb_color))
        self._g.add(Rectangle(pos=(x0, y0), size=(W, kb_h)))

        # 2) Draw uniform separators on pixel boundaries
        sep_color = (0.07, 0.08, 0.10, 0.90)  # slightly darker; tweak to taste
        sep_w = max(1, int(round(dp(1))))     # ensures >= 1 pixel on all DPIs
        self._g.add(Color(*sep_color))
        for i in range(1, 7):  # 6 separators between 7 keys
            sep_x = x0 + i * key_w
            sep_x_int = int(round(sep_x))  # snap to pixel to avoid sub-pixel blur
            self._g.add(Rectangle(pos=(sep_x_int - sep_w // 2, y0), size=(sep_w, kb_h)))

        # Black keys centered on C-D, D-E, F-G, G-A, A-B
        black_boundaries = [0, 1, 3, 4, 5]
        bw = key_w * 0.56
        bh = kb_h * 0.62
        self._g.add(Color(0.06, 0.06, 0.09, 1))
        for j in black_boundaries:
            cx = x0 + (j + 1) * key_w
            bx = int(round(cx - bw / 2.0))  # optional pixel snapping for crispness
            by = y0 + kb_h - bh
            self._g.add(Rectangle(pos=(bx, by), size=(int(round(bw)), bh)))

        # NOTE: chevrons removed entirely.

    def set_chevrons_visible(self, show: bool):
        self.show_chevrons = bool(show)


# -----------------------------------------------------------------------------
# AirOverlayPanel: middle column horizontal 10-dots with manual Y overrides
# -----------------------------------------------------------------------------

from kivy.properties import DictProperty, BooleanProperty, NumericProperty, ListProperty, ColorProperty

from kivy.uix.widget import Widget
from kivy.properties import DictProperty, BooleanProperty, NumericProperty, ListProperty, ColorProperty
from kivy.graphics import Color, InstructionGroup, Ellipse, Line
from kivy.uix.label import Label
from kivy.metrics import dp

FINGERS_ORDER = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

class AirOverlayPanel(Widget):
    """
    Middle column air keyboard: 10 dots laid out horizontally.

    - Transparent background
    - Two smooth mini-arcs (one per hand) using quadratic Bézier curves:
        * Highest at Middle for each hand
        * Index and Thumb taper down; Thumb ≈ Ring height
    - Left-hand visual order inverted (Pinky -> Thumb), Right hand unchanged (Thumb -> Pinky)
    - Weighted intra-hand spacing + a slightly wider mid-gap (between the two thumbs)
    - Right-hand horizontal spacing can mirror the left exactly across the midline
    - Base color #4E6494, pressed color #F08CFF
    - Permanent note labels ABOVE the dots
    - Dot diameter ~ 1/12 of panel height (clamped)
    - Neon presence dots under the active hand's fingers
    - NEW: vertical centering pass – the whole layout is translated vertically so its
      average Y lands at panel center (0.5), preserving your arc shape.
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

    # Hand presence (for neon dots)
    left_present = BooleanProperty(False)
    right_present = BooleanProperty(False)

    # --- Horizontal spacing controls ---
    dot_spacing = NumericProperty(None)          # None → auto spacing by width
    horizontal_margin_dp = NumericProperty(dp(24))
    spacing_factor = NumericProperty(0.90)       # compress the whole row slightly
    mid_gap_factor = NumericProperty(1.4)        # thumbs gap multiple vs. finger gaps

    # Weighted intra-hand gaps (sum is used to solve exact total width)
    # Left hand gaps (visual order): Pinky→Ring, Ring→Middle, Middle→Index, Index→Thumb
    left_gap_weights  = ListProperty([0.78, 0.92, 1.04, 0.95])

    # Right hand gaps (visual order): Thumb→Index, Index→Middle, Middle→Ring, Ring→Pinky
    # If mirror_right_from_left=True, these are computed as reversed left gaps.
    right_gap_weights = ListProperty([0.78, 0.92, 1.04, 0.95])

    # Mirror the right hand spacing from the left (horizontal mirroring)
    mirror_right_from_left = BooleanProperty(True)

    # --- Colors / behavior ---
    base_color = ColorProperty((0x4E/255., 0x64/255., 0x94/255., 1.0))             # #4E6494
    pressed_color = ColorProperty((0xF0/255., 0x8C/255., 0xFF/255., 1.0))          # #F08CFF
    sustain_ring = BooleanProperty(True)

    # --- Sizes ---
    min_dot_diam_dp = NumericProperty(dp(14))
    max_dot_diam_dp = NumericProperty(dp(96))
    label_offset_dp = NumericProperty(dp(8))

    # --- Per-hand arc controls (normalized Y in [0..1]) ---
    # LEFT hand (visual order: Pinky -> Ring -> Middle -> Index -> Thumb)
    left_edge_start_y = NumericProperty(0.55)   # Pinky
    left_center_y     = NumericProperty(0.80)   # Middle (highest)
    left_edge_end_y   = NumericProperty(0.67)   # Thumb (≈ Ring)

    # RIGHT hand: lock to left for equal visual height by default
    lock_right_arc_to_left = BooleanProperty(True)
    right_edge_start_y = NumericProperty(0.67)  # Thumb
    right_center_y     = NumericProperty(0.80)  # Middle
    right_edge_end_y   = NumericProperty(0.55)  # Pinky

    # Presence dots (brighter neon + lower)
    presence_dot_color = ColorProperty((0.20, 1.00, 0.75, 1.0))  # bright neon-lime/teal (~#33FFBF)
    presence_dot_radius_dp = NumericProperty(dp(4))
    presence_dot_offset_dp = NumericProperty(dp(12))              # lower under main dot

    # --- Vertical centering ---
    # When True, after computing all normalized y's, shift them so their mean is 0.5
    auto_center_vertical = BooleanProperty(True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._g = InstructionGroup()
        self.canvas.add(self._g)

        self.bind(
            pos=self._redraw, size=self._redraw,
            left_labels=self._redraw, right_labels=self._redraw,
            left_pressed=self._redraw, right_pressed=self._redraw,
            left_fist=self._redraw, left_thumbup=self._redraw,
            right_fist=self._redraw, right_thumbup=self._redraw,

            # spacing
            dot_spacing=self._redraw, horizontal_margin_dp=self._redraw,
            spacing_factor=self._redraw, mid_gap_factor=self._redraw,
            left_gap_weights=self._redraw, right_gap_weights=self._redraw,
            mirror_right_from_left=self._redraw,

            # visuals
            base_color=self._redraw, pressed_color=self._redraw, sustain_ring=self._redraw,
            min_dot_diam_dp=self._redraw, max_dot_diam_dp=self._redraw,
            label_offset_dp=self._redraw,

            # arcs
            left_edge_start_y=self._redraw, left_center_y=self._redraw, left_edge_end_y=self._redraw,
            right_edge_start_y=self._redraw, right_center_y=self._redraw, right_edge_end_y=self._redraw,
            lock_right_arc_to_left=self._redraw,

            # presence
            left_present=self._redraw, right_present=self._redraw,
            presence_dot_color=self._redraw, presence_dot_radius_dp=self._redraw, presence_dot_offset_dp=self._redraw,

            # vertical centering
            auto_center_vertical=self._redraw,
        )

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
        self.left_fist, self.left_thumbup = bool(left_fist), bool(left_thumbup)
        self.right_fist, self.right_thumbup = bool(right_fist), bool(right_thumbup)
        self.left_present = bool(left_present)
        self.right_present = bool(right_present)

    @staticmethod
    def _quad_bezier_y(u: float, y0: float, yc: float, y1: float) -> float:
        u = 0.0 if u < 0.0 else (1.0 if u > 1.0 else float(u))
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
        names_left_vis  = ["Pinky", "Ring", "Middle", "Index", "Thumb"]
        names_right_vis = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

        all_labels  = [self.left_labels.get(nm, "—")  for nm in names_left_vis]  + \
                      [self.right_labels.get(nm, "—") for nm in names_right_vis]
        all_pressed = [bool(self.left_pressed.get(nm, False))  for nm in names_left_vis]  + \
                      [bool(self.right_pressed.get(nm, False)) for nm in names_right_vis]

        # ---------------- Horizontal positions (weighted gaps + mirrored right) ----------------
        margin = float(self.horizontal_margin_dp)
        usable_w = max(0.0, W - 2.0 * margin)
        N = 10

        if N <= 1 or usable_w <= 0:
            xs = [x0 + W / 2.0] * N
        else:
            # Determine left and right gap weight vectors
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

            # Left hand (5 dots, 4 weighted gaps)
            for i in range(5):
                xs.append(acc)
                if i < 4:
                    acc += s * float(Lw[i])

            # Middle gap (between left Thumb and right Thumb)
            acc += s * float(self.mid_gap_factor)

            # Right hand (5 dots, 4 weighted gaps)
            for i in range(5):
                xs.append(acc)
                if i < 4:
                    acc += s * float(Rw[i])

        # ---------------- Dot size ----------------
        diam = max(float(self.min_dot_diam_dp),
                   min(float(self.max_dot_diam_dp), H / 12.0 if H > 0 else float(self.min_dot_diam_dp)))
        r = diam / 2.0

        # ---------------- Vertical positions via per-hand Bézier arcs ----------------
        if self.lock_right_arc_to_left:
            right_edge_start_y = self.left_edge_end_y
            right_center_y     = self.left_center_y
            right_edge_end_y   = self.left_edge_start_y
        else:
            right_edge_start_y = self.right_edge_start_y
            right_center_y     = self.right_center_y
            right_edge_end_y   = self.right_edge_end_y

        # First compute normalized y's from arcs
        y_norms = []
        # Left group (Pinky..Thumb): u = i/4
        for i in range(5):
            u = i / 4.0
            y_norm = self._quad_bezier_y(u, self.left_edge_start_y, self.left_center_y, self.left_edge_end_y)
            y_norms.append(max(0.0, min(1.0, y_norm)))
        # Right group (Thumb..Pinky): u = j/4
        for j in range(5):
            u = j / 4.0
            y_norm = self._quad_bezier_y(u, right_edge_start_y, right_center_y, right_edge_end_y)
            y_norms.append(max(0.0, min(1.0, y_norm)))

        # ---- Vertical centering pass: shift mean to 0.5 ----
        if self.auto_center_vertical and len(y_norms) == 10:
            mean_y = sum(y_norms) / 10.0
            delta = 0.5 - mean_y
            for k in range(10):
                y_norms[k] = max(0.0, min(1.0, y_norms[k] + delta))

        # Convert to pixel y's
        ys = [y0 + yn * H for yn in y_norms]

        # ---------------- Draw dots + labels + presence dots ----------------
        pr = float(self.presence_dot_radius_dp)
        poff = float(self.presence_dot_offset_dp)
        for i in range(N):
            cx, cy = xs[i], ys[i]
            on = all_pressed[i]
            color = self.pressed_color if on else self.base_color

            # Main dot
            self._g.add(Color(*color))
            self._g.add(Ellipse(pos=(cx - r, cy - r), size=(2 * r, 2 * r)))

            # Presence dot under detected hand
            is_left_hand = (i < 5)
            if ((self.left_present and is_left_hand) or (self.right_present and not is_left_hand)) and pr > 0.0:
                extra = r * 0.15  # proportional drop on large dots
                py = max(y0 + pr, cy - r - (poff + extra))
                self._g.add(Color(*self.presence_dot_color))
                self._g.add(Ellipse(pos=(cx - pr, py - pr), size=(2 * pr, 2 * pr)))

            # Optional sustain ring on press
            if self.sustain_ring and on:
                self._g.add(Color(color[0], color[1], color[2], 0.25))
                self._g.add(Line(circle=(cx, cy, r + dp(4)), width=dp(1.6)))

            # Label ABOVE the dot (permanent)
            label_text = str(all_labels[i]) if all_labels[i] else "—"
            lbl = Label(
                text=label_text,
                font_size=dp(14),
                color=(0.92, 0.95, 1, 1),
                size_hint=(None, None),
                size=(dp(90), dp(22)),
                pos=(cx - dp(45), cy + r + float(self.label_offset_dp)),
                halign="center",
                valign="middle",
            )
            lbl.text_size = lbl.size
            lbl._air_label = True
            self.add_widget(lbl)


class GradientBackground(RelativeLayout):
    """
    Draws a vertical linear gradient background using a tiny 1xN texture
    stretched over the widget's size. Put any children inside—this provides
    the application-wide background behind everything.
    """
    # Defaults: Top #1A2B43, Bottom #182436
    color_top = ListProperty([0x1A/255., 0x2B/255., 0x43/255., 1.0])
    color_bottom = ListProperty([0x18/255., 0x24/255., 0x36/255., 1.0])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tex = None
        with self.canvas.before:
            # Keep color white so we don’t tint the gradient texture.
            self._color_instr = Color(1, 1, 1, 1)
            self._rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._update_rect, size=self._update_rect,
                  color_top=self._rebuild_gradient, color_bottom=self._rebuild_gradient)
        self._rebuild_gradient()

    def _update_rect(self, *args):
        self._rect.pos = self.pos
        self._rect.size = self.size

    def _rebuild_gradient(self, *args):
        """Build a small vertical RGBA gradient strip and stretch it to fill."""
        height = 64
        tex = Texture.create(size=(1, height), colorfmt='rgba')
        tex.wrap = 'clamp_to_edge'

        r1, g1, b1, a1 = self.color_top
        r2, g2, b2, a2 = self.color_bottom

        buf = bytearray()
        for i in range(height):
            t = i / float(height - 1)
            r = int((r1 * (1 - t) + r2 * t) * 255)
            g = int((g1 * (1 - t) + g2 * t) * 255)
            b = int((b1 * (1 - t) + b2 * t) * 255)
            a = int((a1 * (1 - t) + a2 * t) * 255)
            buf.extend([r, g, b, a])

        tex.blit_buffer(bytes(buf), colorfmt='rgba', bufferfmt='ubyte')
        self._tex = tex
        self._rect.texture = self._tex
        # Vertical stretch
        self._rect.tex_coords = (0, 0,  1, 0,  1, 1,  0, 1)

from kivy.uix.widget import Widget
from kivy.properties import ColorProperty
from kivy.graphics import Color, Rectangle, InstructionGroup

class LeftOptionsPanel(Widget):
    """
    Full-rect background for the options area under the video.
    Add as the FIRST child in a RelativeLayout to render behind other content.
    """
    # Same color as the right piano roll panel (#121C2A). Opaque by default.
    color = ColorProperty((0x12/255., 0x1C/255., 0x2A/255., 1.0))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._g = InstructionGroup()
        self.canvas.before.add(self._g)
        self.bind(pos=self._redraw, size=self._redraw, color=self._redraw)

    def _redraw(self, *args):
        self._g.clear()
        self._g.add(Color(*self.color))
        self._g.add(Rectangle(pos=self.pos, size=self.size))



from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import (
    StringProperty, BooleanProperty, NumericProperty, ColorProperty, ListProperty
)
from kivy.metrics import dp
from kivy.graphics import Color, RoundedRectangle, InstructionGroup, Line
from kivy.clock import Clock


from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import (
    StringProperty, BooleanProperty, NumericProperty, ColorProperty, ListProperty
)
from kivy.graphics import Color, RoundedRectangle, InstructionGroup, Ellipse
from kivy.metrics import dp
from kivy.clock import Clock


class PillButton(Button):

    text_label = StringProperty("")
    toggled = BooleanProperty(False)
    toggle_mode = BooleanProperty(False)

    # Colors
    base_color = ColorProperty((0x12/255., 0x1C/255., 0x2A/255., 0.98))
    base_color_down = ColorProperty((0x15/255., 0x22/255., 0x33/255., 1))
    text_color = ColorProperty((0.92, 0.95, 1, 1))
    active_color = ColorProperty((0.80, 0.20, 0.25, 1))
    active_tint = ColorProperty((0.90, 0.30, 0.35, 1))

    radius_dp = NumericProperty(dp(18))
    pad_x = NumericProperty(dp(14))
    pad_y = NumericProperty(dp(8))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_normal = ""
        self.background_down = ""
        self.background_color = (0, 0, 0, 0)
        self.color = tuple(self.text_color)
        self.bold = True
        self.font_size = dp(14)
        self.halign = "center"
        self.valign = "middle"
        self.text = self.text_label or self.text

        # Custom canvas
        self._g = InstructionGroup()
        self.canvas.before.add(self._g)

        self.bind(pos=self._redraw, size=self._redraw,
                  text_label=self._sync_text, toggled=self._redraw,
                  text=self._redraw, text_color=self._redraw)

        # Make sure label aligns
        Clock.schedule_once(lambda *_: self._sync_text(), 0)

        # Toggle behavior
        self.bind(on_release=self._on_released)

    def _on_released(self, *_):
        if self.toggle_mode:
            self.toggled = not self.toggled

    def _sync_text(self, *_):

        if self.text_label:
            self.text = self.text_label
        self.text_size = (self.width - 2 * self.pad_x, self.height - 2 * self.pad_y)

    def _redraw(self, *_):
        self._g.clear()


        if self.toggle_mode and self.toggled:
            fill = self.active_color
            down = self.active_tint
        else:
            fill = self.base_color
            down = self.base_color_down

        current = down if self.state == "down" else fill

        # Subtle shadow
        self._g.add(Color(0, 0, 0, 0.22))
        self._g.add(RoundedRectangle(
            pos=(self.x, self.y - dp(1.5)),
            size=(self.width, self.height),
            radius=[self.radius_dp] * 4
        ))

        # Fill only (no outline)
        self._g.add(Color(*current))
        self._g.add(RoundedRectangle(
            pos=self.pos, size=self.size, radius=[self.radius_dp] * 4
        ))


class CircleIconButton(Button):
    """
    Circular pill button with a centered white dot for 'Record'.
    Matches dusk-blue aesthetic. Toggle-capable.
    """
    toggled = BooleanProperty(False)          # active state (e.g., recording)
    toggle_mode = BooleanProperty(True)       # default: toggling behavior

    # Colors (harmonized with your panels)
    base_color = ColorProperty((0x12/255., 0x1C/255., 0x2A/255., 0.98))
    base_color_down = ColorProperty((0x15/255., 0x22/255., 0x33/255., 1))
    active_color = ColorProperty((0.80, 0.20, 0.25, 1))   # active fill
    active_tint = ColorProperty((0.90, 0.30, 0.35, 1))    # pressed while active

    # Geometry
    radius_dp = NumericProperty(dp(24))       # keep larger than other pills
    dot_scale = NumericProperty(0.38)         # inner white dot diameter vs. outer (0..1)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_normal = ""
        self.background_down = ""
        self.background_color = (0, 0, 0, 0)  # custom canvas
        self.text = ""  # icon-only
        self._g = InstructionGroup()
        self.canvas.before.add(self._g)

        self.bind(pos=self._redraw, size=self._redraw,
                  toggled=self._redraw, state=self._redraw,
                  base_color=self._redraw, active_color=self._redraw,
                  dot_scale=self._redraw)

        self.bind(on_release=self._maybe_toggle)

    def _maybe_toggle(self, *_):
        if self.toggle_mode:
            self.toggled = not self.toggled

    def _redraw(self, *_):
        self._g.clear()

        # Choose outer fill based on state
        if self.toggle_mode and self.toggled:
            fill = self.active_color
            down = self.active_tint
        else:
            fill = self.base_color
            down = self.base_color_down
        current = down if self.state == "down" else fill

        # Outer circle geometry
        outer_r = min(self.width, self.height) / 2.0
        rad = max(self.radius_dp, outer_r)

        # Subtle shadow (slightly below)
        self._g.add(Color(0, 0, 0, 0.22))
        self._g.add(RoundedRectangle(
            pos=(self.center_x - outer_r, self.center_y - outer_r - dp(1.2)),
            size=(2 * outer_r, 2 * outer_r),
            radius=[rad] * 4
        ))

        # Outer fill (no outline)
        self._g.add(Color(*current))
        self._g.add(RoundedRectangle(
            pos=(self.center_x - outer_r, self.center_y - outer_r),
            size=(2 * outer_r, 2 * outer_r),
            radius=[rad] * 4
        ))

        # Inner white dot
        inner_d = max(0.0, min(1.0, float(self.dot_scale))) * (2 * outer_r)
        inner_r = inner_d / 2.0
        if inner_r > 0:
            self._g.add(Color(1, 1, 1, 1))
            self._g.add(Ellipse(
                pos=(self.center_x - inner_r, self.center_y - inner_r),
                size=(inner_d, inner_d)
            ))


from kivy.uix.anchorlayout import AnchorLayout

from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import NumericProperty, ListProperty
from kivy.metrics import dp



class UpperDock(Widget):
    """
    Floating dock anchored to the TOP.
    Left-aligned; contains: [● Record (largest)] [Instrument] [Layer]
    Instrument & Layer are vertically centered to the Record button.
    """
    margin_dp = NumericProperty(dp(10))
    spacing_dp = NumericProperty(dp(8))
    pad_dp = ListProperty([dp(8), dp(8), dp(8), dp(8)])  # [l, t, r, b]

    # Uniform pill height (Instrument/Layer)
    pill_height_dp = NumericProperty(dp(44))
    # Record is visually largest
    record_diameter_dp = NumericProperty(dp(56))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Horizontal row that hosts all three controls
        self._row = BoxLayout(
            orientation="horizontal",
            spacing=self.spacing_dp,
            padding=self.pad_dp,
            size_hint=(None, None),
            height=self.pill_height_dp,   # row height = pill height
            width=0
        )
        self.add_widget(self._row)

        # --- Record (largest), vertically centered in row
        self._rec_slot = AnchorLayout(
            anchor_x="left",
            anchor_y="center",
            size_hint=(None, 1),
            width=self.record_diameter_dp
        )
        self.btn_record = CircleIconButton(size_hint=(None, None))
        self.btn_record.size = (self.record_diameter_dp, self.record_diameter_dp)
        self._rec_slot.add_widget(self.btn_record)

        # --- Instrument: centered vertically w.r.t. row (and thus record)
        self._inst_slot = AnchorLayout(
            anchor_x="left",
            anchor_y="center",
            size_hint=(None, 1),
            width=dp(180)
        )
        self.btn_instrument = PillButton(text_label="Instrument: Piano",
                                         size_hint=(None, None))
        self.btn_instrument.height = self.pill_height_dp
        self.btn_instrument.width = self._inst_slot.width
        self._inst_slot.add_widget(self.btn_instrument)

        # --- Layer: centered vertically w.r.t. row (and thus record)
        self._layer_slot = AnchorLayout(
            anchor_x="left",
            anchor_y="center",
            size_hint=(None, 1),
            width=dp(130)
        )
        self.btn_layer = PillButton(text_label="Layer: 1",
                                    size_hint=(None, None))
        self.btn_layer.height = self.pill_height_dp
        self.btn_layer.width = self._layer_slot.width
        self._layer_slot.add_widget(self.btn_layer)

        # Add children once, in order
        self._row.add_widget(self._rec_slot)
        self._row.add_widget(self._inst_slot)
        self._row.add_widget(self._layer_slot)
        self._recompute_row_width()

        # Reactive bindings
        self.bind(pos=self._sync, size=self._sync,
                  spacing_dp=self._rebuild_layout, pad_dp=self._rebuild_layout,
                  pill_height_dp=self._sync_sizes, record_diameter_dp=self._sync_sizes)

        # Events
        self.register_event_type("on_record_toggled")
        self.register_event_type("on_instrument_pressed")
        self.register_event_type("on_layer_pressed")

        # Wire actions
        self.btn_record.bind(on_release=lambda *_: self.dispatch("on_record_toggled", self.btn_record.toggled))
        self.btn_instrument.bind(on_release=lambda *_: self.dispatch("on_instrument_pressed"))
        self.btn_layer.bind(on_release=lambda *_: self.dispatch("on_layer_pressed"))

    def _sync_sizes(self, *_):
        # Keep row/pills height uniform
        self._row.height = self.pill_height_dp

        self.btn_instrument.height = self.pill_height_dp
        self.btn_layer.height = self.pill_height_dp

        # Record circle remains largest and centered
        self._rec_slot.width = self.record_diameter_dp
        self.btn_record.size = (self.record_diameter_dp, self.record_diameter_dp)

        # Ensure slot widths match their button widths
        self._inst_slot.width = self.btn_instrument.width
        self._layer_slot.width = self.btn_layer.width

        self._recompute_row_width()

    def _rebuild_layout(self, *_):
        self._row.spacing = self.spacing_dp
        self._row.padding = self.pad_dp
        self._recompute_row_width()

    def _recompute_row_width(self):
        total = 0
        for ch in self._row.children:
            total += ch.width
        if len(self._row.children) > 1:
            total += self._row.spacing * (len(self._row.children) - 1)
        self._row.width = total

    def _sync(self, *_):
        # Position the row at the very top within this widget's rect
        self._row.pos = (self.x + self.margin_dp, self.top - self._row.height - self.margin_dp)

    # Event stubs
    def on_record_toggled(self, is_recording: bool): pass
    def on_instrument_pressed(self): pass
    def on_layer_pressed(self): pass



class LowerDock(Widget):

    margin_dp = NumericProperty(dp(10))
    spacing_dp = NumericProperty(dp(8))
    pad_dp = ListProperty([dp(8), dp(8), dp(8), dp(8)])  # [l, t, r, b]

    pill_height_dp = NumericProperty(dp(44))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._container = BoxLayout(
            orientation="horizontal",
            padding=self.pad_dp,
            size_hint=(1, None),
            height=self.pill_height_dp
        )
        self._row = BoxLayout(
            orientation="horizontal",
            spacing=self.spacing_dp,
            size_hint=(None, 1),
            width=0
        )
        self._container.add_widget(self._row)
        self.add_widget(self._container)


        self.btn_mute = PillButton(text_label="Mute", toggle_mode=True, size_hint=(None, None))
        self.btn_mute.height = self.pill_height_dp
        self.btn_mute.width = dp(110)

        self.btn_controls = PillButton(text_label="Controls: 10-key mapping", size_hint=(None, None))
        self.btn_controls.height = self.pill_height_dp
        self.btn_controls.width = dp(230)

        self._row.add_widget(self.btn_mute)
        self._row.add_widget(self.btn_controls)
        self._recompute_row_width()

        self.bind(pos=self._sync, size=self._sync,
                  spacing_dp=self._rebuild_layout, pad_dp=self._rebuild_layout,
                  pill_height_dp=self._sync_sizes)

        # Events
        self.register_event_type("on_mute_toggled")
        self.register_event_type("on_controls_pressed")

        self.btn_mute.bind(on_release=self._emit_mute)
        self.btn_controls.bind(on_release=lambda *_: self.dispatch("on_controls_pressed"))

    def _emit_mute(self, *_):

        self.dispatch("on_mute_toggled", self.btn_mute.toggled)

    def _sync_sizes(self, *_):
        self._container.height = self.pill_height_dp
        self.btn_mute.height = self.pill_height_dp
        self.btn_controls.height = self.pill_height_dp
        self._recompute_row_width()

    def _rebuild_layout(self, *_):
        self._container.padding = self.pad_dp
        self._row.spacing = self.spacing_dp
        self._recompute_row_width()

    def _recompute_row_width(self):
        total = 0
        for ch in self._row.children:
            total += ch.width
        if len(self._row.children) > 1:
            total += self._row.spacing * (len(self._row.children) - 1)
        self._row.width = total

    def _sync(self, *_):
        self._container.pos = (self.x + self.margin_dp, self.y + self.margin_dp)
        self._container.width = self.width - 2 * self.margin_dp
        self._container.height = self.pill_height_dp

    # Event stubs
    def on_mute_toggled(self, is_muted: bool): pass
    def on_controls_pressed(self): pass