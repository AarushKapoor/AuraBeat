# src/ui/widgets/layout.py
from __future__ import annotations

from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.widget import Widget
from kivy.properties import ListProperty
from kivy.graphics import Color, Rectangle
from kivy.graphics.texture import Texture
from kivy.metrics import dp


# ---------------------------------------------------------------------
# GradientBackground
# ---------------------------------------------------------------------
class GradientBackground(RelativeLayout):
    """
    Vertical linear gradient background using a 1xN texture.
    """
    color_top = ListProperty([0x1A/255., 0x2B/255., 0x43/255., 1.0])
    color_bottom = ListProperty([0x18/255., 0x24/255., 0x36/255., 1.0])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tex = None

        with self.canvas.before:
            Color(1, 1, 1, 1)
            self._rect = Rectangle()

        self.bind(pos=self._update_rect, size=self._update_rect,
                  color_top=self._rebuild_texture, color_bottom=self._rebuild_texture)

        self._rebuild_texture()

    def _rebuild_texture(self, *args):
        h = 256
        arr = []

        r1, g1, b1, a1 = self.color_top
        r2, g2, b2, a2 = self.color_bottom

        import numpy as np
        y = np.linspace(0.0, 1.0, h)
        grad = np.zeros((h, 1, 4), dtype=np.float32)
        grad[:, 0, 0] = r1 * (1 - y) + r2 * y
        grad[:, 0, 1] = g1 * (1 - y) + g2 * y
        grad[:, 0, 2] = b1 * (1 - y) + b2 * y
        grad[:, 0, 3] = a1 * (1 - y) + a2 * y

        grad = (grad * 255).astype("uint8")
        grad = np.flipud(grad)

        tex = Texture.create(size=(1, h))
        tex.blit_buffer(grad.tobytes(), colorfmt="rgba", bufferfmt="ubyte")
        tex.wrap = "repeat"
        self._tex = tex

        self._update_rect()

    def _update_rect(self, *args):
        if self._tex is None:
            return
        self._rect.texture = self._tex
        self._rect.pos = self.pos
        self._rect.size = self.size


# ---------------------------------------------------------------------
# LeftOptionsPanel
# ---------------------------------------------------------------------
class LeftOptionsPanel(Widget):
    """Left column container background."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        with self.canvas.before:
            Color(0.10, 0.12, 0.16, 0.85)  # same tone as docks
            self._rect = Rectangle()

        self.bind(pos=self._update_rect, size=self._update_rect)

    def _update_rect(self, *args):
        self._rect.pos = self.pos
        self._rect.size = self.size

