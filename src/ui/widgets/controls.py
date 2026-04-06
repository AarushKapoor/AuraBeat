# src/ui/widgets/controls.py
from __future__ import annotations

from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.properties import BooleanProperty, NumericProperty
from kivy.graphics import Color, RoundedRectangle, InstructionGroup
from kivy.metrics import dp


# ---------------------------------------------------------------------
# PillButton
# ---------------------------------------------------------------------
class PillButton(Button):
    """Rounded pill-shaped button used in left options panel."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_normal = ""
        self.background_down = ""
        self.background_color = (0, 0, 0, 0)
        self.color = (0.92, 0.95, 1, 1)
        self.font_size = dp(16)

        self._bg = InstructionGroup()
        self.canvas.before.add(self._bg)

        self.bind(pos=self._redraw, size=self._redraw)
        self._redraw()

    def _redraw(self, *args):
        self._bg.clear()
        self._bg.add(Color(0.0706, 0.1098, 0.1647, 1))
        self._bg.add(RoundedRectangle(
            pos=self.pos,
            size=self.size,
            radius=[dp(18)] * 4
        ))


# ---------------------------------------------------------------------
# CircleIconButton
# ---------------------------------------------------------------------
class CircleIconButton(Button):
    """Circular icon button (e.g., for toggles)."""
    active = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_normal = ""
        self.background_down = ""
        self.background_color = (0, 0, 0, 0)
        self.color = (0.92, 0.95, 1, 1)
        self.font_size = dp(18)

        self._g = InstructionGroup()
        self.canvas.before.add(self._g)

        self.bind(pos=self._redraw, size=self._redraw, active=self._redraw)
        self._redraw()

    def _redraw(self, *args):
        self._g.clear()
        r = min(self.width, self.height) / 2.0

        # Outer circle
        if self.active:
            self._g.add(Color(1, 1, 1, 1))  # white
        else:
            self._g.add(Color(0.0706, 0.1098, 0.1647, 1))  # theme navy

        self._g.add(RoundedRectangle(
            pos=(self.x, self.y),
            size=(2 * r, 2 * r),
            radius=[r]
        ))

        # Inner circle
        inner_r = r * 0.40
        if self.active:
            self._g.add(Color(0, 0, 0, 1))  # black
        else:
            self._g.add(Color(1, 0.15, 0.15, 1))  # red

        self._g.add(RoundedRectangle(
            pos=(self.center_x - inner_r, self.center_y - inner_r),
            size=(2 * inner_r, 2 * inner_r),
            radius=[inner_r]
        ))




