# src/ui/widgets.py
from __future__ import annotations

from typing import Dict

import numpy as np

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


# -----------------------------------------------------------------------------
# VideoFeed: rounded video preview with stencil mask + optional subtle border
# -----------------------------------------------------------------------------
class VideoFeed(Image):
    """Image widget with rounded corners using a stencil mask."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.corner_radius_dp = dp(14)
        self.draw_border = True

        self._pre = InstructionGroup()
        self.canvas.before.add(self._pre)

        self._post = InstructionGroup()
        self.canvas.after.add(self._post)

        self._texture = None
        self.bind(pos=self._rebuild_stencil, size=self._rebuild_stencil)
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

        # Optional border line atop
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
class PianoRollPanel(Widget):
    """Right-side piano roll panel (Synthesia-inspired)."""
    keyboard_height_ratio = NumericProperty(0.18)
    show_chevrons = BooleanProperty(True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._g = InstructionGroup()
        self.canvas.add(self._g)
        self._border_tex = None
        self.bind(pos=self._redraw, size=self._redraw,
                  keyboard_height_ratio=self._redraw, show_chevrons=self._redraw)

    def _ensure_border_texture(self):
        if self._border_tex is not None:
            return

        h, w = 256, 8
        y = np.linspace(0.0, 1.0, h, dtype=np.float32)
        alpha_line = 1.0 - np.abs(2.0 * y - 1.0)
        arr = np.zeros((h, w, 4), dtype=np.uint8)
        arr[..., 0:3] = 255
        arr[..., 3] = (alpha_line[:, None] * 0.85 * 255).astype(np.uint8)
        # Flip for Kivy's buffer orientation
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

        # Panel background
        self._g.add(Color(0.11, 0.12, 0.14, 1))
        self._g.add(Rectangle(pos=(x0, y0), size=(W, H)))

        # Feathered vertical accents
        if track_h > 0:
            accent_len = max(dp(30), track_h / 3.0)
            yc = track_y + track_h / 2.0
            accent_y = yc - accent_len / 2.0
            razor = max(1.0, dp(1.2))

            self._g.add(Color(1, 1, 1, 1))
            self._g.add(Rectangle(pos=(x0, accent_y), size=(razor, accent_len), texture=self._border_tex))
            self._g.add(Rectangle(pos=(x0 + W - razor, accent_y), size=(razor, accent_len), texture=self._border_tex))

        # Bottom mini keyboard
        self._g.add(Color(0.07, 0.08, 0.10, 1))
        self._g.add(Rectangle(pos=(x0, y0), size=(W, kb_h)))

        # White keys (7)
        key_w = W / 7.0
        pad = dp(1.0)
        self._g.add(Color(0.90, 0.92, 0.96, 1))
        for i in range(7):
            wx = x0 + i * key_w + pad
            wy = y0 + pad
            ww = max(0.0, key_w - 2 * pad)
            wh = max(0.0, kb_h - 2 * pad)
            if ww > 0 and wh > 0:
                self._g.add(Rectangle(pos=(wx, wy), size=(ww, wh)))

        # Black keys centered on C-D, D-E, F-G, G-A, A-B
        black_boundaries = [0, 1, 3, 4, 5]
        bw = key_w * 0.56
        bh = kb_h * 0.62
        self._g.add(Color(0.06, 0.06, 0.09, 1))
        for j in black_boundaries:
            cx = x0 + (j + 1) * key_w
            bx = cx - bw / 2.0
            by = y0 + kb_h - bh
            if bw > 0 and bh > 0:
                self._g.add(Rectangle(pos=(bx, by), size=(bw, bh)))

        # Optional chevrons
        if self.show_chevrons and track_h > dp(20):
            self._g.add(Color(0.2, 1.0, 0.4, 0.45))
            yc = track_y + track_h / 2.0
            margin = dp(12)
            sz = dp(18)

            # Left '<'
            xL = x0 + margin
            self._g.add(Line(points=[xL + sz, yc + sz, xL, yc], width=dp(2)))
            self._g.add(Line(points=[xL, yc, xL + sz, yc - sz], width=dp(2)))

            # Right '>'
            xR = x0 + W - margin
            self._g.add(Line(points=[xR - sz, yc + sz, xR, yc], width=dp(2)))
            self._g.add(Line(points=[xR, yc, xR - sz, yc - sz], width=dp(2)))

    def set_chevrons_visible(self, show: bool):
        self.show_chevrons = bool(show)


# -----------------------------------------------------------------------------
# AirOverlayPanel: middle column air keyboard (two 5-dot stacks)
# -----------------------------------------------------------------------------
FINGERS_ORDER = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


class AirOverlayPanel(Widget):
    """Middle-column air keyboard: Left & Right stacks of 5 dots, with labels & press glow."""
    left_labels = DictProperty({})
    right_labels = DictProperty({})
    left_pressed = DictProperty({})
    right_pressed = DictProperty({})
    left_fist = BooleanProperty(False)
    left_thumbup = BooleanProperty(False)
    right_fist = BooleanProperty(False)
    right_thumbup = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._g = InstructionGroup()
        self.canvas.add(self._g)

        self.bind(
            pos=self._redraw, size=self._redraw,
            left_labels=self._redraw, right_labels=self._redraw,
            left_pressed=self._redraw, right_pressed=self._redraw,
            left_fist=self._redraw, left_thumbup=self._redraw,
            right_fist=self._redraw, right_thumbup=self._redraw
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
        right_thumbup=False
    ):
        self.left_labels = left_labels or {}
        self.right_labels = right_labels or {}
        self.left_pressed = left_pressed or {}
        self.right_pressed = right_pressed or {}
        self.left_fist, self.left_thumbup = bool(left_fist), bool(left_thumbup)
        self.right_fist, self.right_thumbup = bool(right_fist), bool(right_thumbup)

    def _redraw(self, *args):
        self._g.clear()
        x0, y0 = self.x, self.y
        W, H = self.width, self.height

        # Background
        self._g.add(Color(0.07, 0.08, 0.10, 1))
        self._g.add(Rectangle(pos=(x0, y0), size=(W, H)))

        # Layout
        gap = dp(36)
        col_w = (W - gap) / 2.0
        left_x = x0 + dp(24)
        right_x = x0 + col_w + gap + dp(24)
        top_pad, bot_pad = dp(36), dp(36)
        usable_h = max(1.0, H - top_pad - bot_pad)
        row_gap = usable_h / (len(FINGERS_ORDER) + 1)

        # Clean old labels once (so they don't stack up between redraws)
        for c in list(self.children):
            if getattr(c, "_air_label", False):
                self.remove_widget(c)

        def draw_stack(xc: float, labels: Dict[str, str], pressed: Dict[str, bool], fist=False, thumbup=False):
            # Small icons near top for whole-hand gestures
            if fist:
                self._g.add(Color(1.0, 0.55, 0.20, 1.0))
                self._g.add(Line(circle=(xc, y0 + H - dp(22), dp(10)), width=dp(2)))
            if thumbup:
                self._g.add(Color(0.25, 0.85, 0.35, 1.0))
                self._g.add(Line(circle=(xc + dp(26), y0 + H - dp(22), dp(8)), width=dp(2)))

            # Dots + Labels for Thumb..Pinky
            for i, name in enumerate(FINGERS_ORDER):
                cy = y0 + H - top_pad - (i + 1) * row_gap
                on = bool(pressed.get(name, False))
                text = labels.get(name, "—")

                # Dot
                self._g.add(Color(*( (1.0, 0.55, 0.20, 1.0) if on else (0.24, 0.94, 0.78, 1.0) )))
                r = dp(12 if on else 10)
                self._g.add(Ellipse(pos=(xc - r, cy - r), size=(2 * r, 2 * r)))

                # Label (emoji-capable via Kivy Label)
                lbl = Label(
                    text=text,
                    font_size=dp(16),
                    color=(0.92, 0.95, 1, 1),
                    size_hint=(None, None),
                    size=(col_w - dp(48), dp(24)),
                    pos=(xc + dp(16), cy - dp(12)),
                    halign="left",
                    valign="middle",
                )
                lbl.text_size = lbl.size
                lbl._air_label = True
                self.add_widget(lbl)

        # Left stack then right stack
        draw_stack(left_x, self.left_labels, self.left_pressed, fist=self.left_fist, thumbup=self.left_thumbup)
        draw_stack(right_x, self.right_labels, self.right_pressed, fist=self.right_fist, thumbup=self.right_thumbup)