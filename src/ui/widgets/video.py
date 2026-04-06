# src/ui/widgets/video.py
from __future__ import annotations

import numpy as np
from typing import Dict

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

from kivy.properties import (
    BooleanProperty,
    NumericProperty,
    DictProperty,
)

from kivy.metrics import dp
from kivy.clock import Clock
from kivy.animation import Animation
from kivy.graphics import (
    Color, Rectangle, InstructionGroup, Line, Ellipse, RoundedRectangle,
    StencilPush, StencilUse, StencilUnUse, StencilPop
)
from kivy.graphics.texture import Texture


# ---------------------------------------------------------------------
# Root container
# ---------------------------------------------------------------------
class RootView(BoxLayout):
    """Top-level Kivy container used by KV."""
    pass


# ---------------------------------------------------------------------
# VideoFeed
# ---------------------------------------------------------------------
class VideoFeed(Image):
    """Image widget with rounded corners using a stencil mask."""
    draw_border = BooleanProperty(True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.corner_radius_dp = dp(14)

        self._pre = InstructionGroup()
        self.canvas.before.add(self._pre)

        self._post = InstructionGroup()
        self.canvas.after.add(self._post)

        self._texture = None
        self.bind(pos=self._rebuild_stencil, size=self._rebuild_stencil, draw_border=self._rebuild_stencil)
        self._rebuild_stencil()

    def _rebuild_stencil(self, *args):
        self._pre.clear()
        self._post.clear()

        # Stencil mask
        self._pre.add(StencilPush())
        self._pre.add(Color(1, 1, 1, 1))
        self._mask = RoundedRectangle(
            pos=self.pos,
            size=self.size,
            radius=[self.corner_radius_dp] * 4
        )
        self._pre.add(self._mask)
        self._pre.add(StencilUse())

        # End stencil
        self._post.add(StencilUnUse())
        self._post.add(Color(1, 1, 1, 1))
        self._post.add(StencilPop())

        # Optional border
        if self.draw_border:
            self._post.add(Color(1, 1, 1, 0.18))
            self._post.add(Line(
                rounded_rectangle=(self.x, self.y, self.width, self.height, self.corner_radius_dp),
                width=dp(1.2))
            )

    def set_frame(self, rgb_frame: np.ndarray):
        if rgb_frame is None or not hasattr(rgb_frame, "shape") or len(rgb_frame.shape) != 3:
            return

        h, w = rgb_frame.shape[:2]
        if h <= 1 or w <= 1:
            return

        if (self._texture is None) or (self._texture.width != w) or (self._texture.height != h):
            tex = Texture.create(size=(w, h))
            tex.flip_vertical()
            self._texture = tex

        self.texture = self._texture
        self.texture.blit_buffer(rgb_frame.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
        self.canvas.ask_update()


# ---------------------------------------------------------------------
# CircleButton
# ---------------------------------------------------------------------
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

        self._bg_instr.add(Color(0.12, 0.14, 0.18, 1))
        self._bg_instr.add(Ellipse(pos=(self.x, self.y), size=(2*r, 2*r)))

        self._bg_instr.add(Color(1, 1, 1, 0.18))
        self._bg_instr.add(Line(circle=(self.center_x, self.center_y, r), width=dp(1.2)))


# ---------------------------------------------------------------------
# QuickMenu
# ---------------------------------------------------------------------
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
            padding=[dp(10)] * 4
        )
        self.add_widget(self._items)

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
        self._bg_instr.add(Color(0.13, 0.14, 0.18, 0.98))
        self._bg_instr.add(RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(12)] * 4))

        self._items.pos = self.pos
        self._items.size = self.size


# ---------------------------------------------------------------------
# GestureHUD
# ---------------------------------------------------------------------
GESTURE_META: Dict[str, Dict[str, object]] = {
    "Open Hand": {"icon": "🖐", "rgb_f": (0.24, 0.82, 0.24)},
    "Fist": {"icon": "✊", "rgb_f": (0.98, 0.55, 0.20)},
    "Point": {"icon": "☝️", "rgb_f": (0.16, 0.67, 0.90)},
    "Thumbs Up": {"icon": "👍", "rgb_f": (0.16, 0.82, 0.82)},
    "": {"icon": "❓", "rgb_f": (0.55, 0.55, 0.55)},
}


class GestureHUD(Widget):
    FADE_DELAY = 1.8
    BADGE_H = 38
    BADGE_PAD_X = 12
    BADGE_PAD_Y = 6
    MAX_HANDS = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._gestures = []
        self._fade_event = None

        self._g = InstructionGroup()
        self.canvas.add(self._g)

        self.opacity = 0.0
        self.bind(pos=self._redraw, size=self._redraw)

    def update_gestures(self, gestures):
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

        self._g.add(Color(0.09, 0.10, 0.13, 0.92))
        self._g.add(RoundedRectangle(
            pos=(x0, y0 + self.height - total_h),
            size=(W, total_h),
            radius=[dp(10)] * 4
        ))

        for child in list(self.children):
            self.remove_widget(child)

        for i, gesture in enumerate(self._gestures):
            meta = GESTURE_META.get(gesture, GESTURE_META[""])
            r, g_c, b = meta["rgb_f"]

            row_y = y0 + self.height - dp(self.BADGE_PAD_Y) * (i + 1) - row_h * (i + 1)

            self._g.add(Color(r, g_c, b, 0.18))
            self._g.add(RoundedRectangle(
                pos=(x0 + dp(self.BADGE_PAD_X), row_y),
                size=(W - dp(self.BADGE_PAD_X) * 2, row_h),
                radius=[dp(7)] * 4
            ))

            self._g.add(Color(r, g_c, b, 0.90))
            self._g.add(RoundedRectangle(
                pos=(x0 + dp(self.BADGE_PAD_X), row_y),
                size=(dp(4), row_h),
                radius=[dp(3)] * 4
            ))

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
