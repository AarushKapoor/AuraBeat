from kivy.metrics import dp
from kivy.graphics import Color, RoundedRectangle
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.properties import ObjectProperty


from kivy.animation import Animation
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.image import Image
from kivy.graphics import Color, Rectangle


class ImageButton(ButtonBehavior, Image):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Overlay color for tap animation
        with self.canvas.after:
            self._tap_color = Color(1, 1, 1, 0)  # transparent
            self._tap_rect = Rectangle(pos=self.pos, size=self.size)

        self.bind(pos=self._update_rect, size=self._update_rect)

    def _update_rect(self, *_):
        self._tap_rect.pos = self.pos
        self._tap_rect.size = self.size

    def on_press(self):
        # Flash brighter
        Animation.cancel_all(self._tap_color)
        anim = Animation(a=0.35, duration=0.08)
        anim.start(self._tap_color)

    def on_release(self):
        # Fade back to transparent
        Animation.cancel_all(self._tap_color)
        anim = Animation(a=0.0, duration=0.12)
        anim.start(self._tap_color)



class TransportControls(BoxLayout):
    # Callbacks that UpperDock will assign
    on_play = ObjectProperty(None, allownone=True)
    on_pause = ObjectProperty(None, allownone=True)
    on_stop = ObjectProperty(None, allownone=True)
    on_back = ObjectProperty(None, allownone=True)
    on_forward = ObjectProperty(None, allownone=True)
    on_loop = ObjectProperty(None, allownone=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.orientation = "horizontal"
        self.spacing = dp(6)
        self.padding = dp(6)
        self.size_hint = (None, None)

        # Background
        with self.canvas.before:
            Color(0.0706, 0.1098, 0.1647, 1)  # #121C2A
            self.bg = RoundedRectangle(radius=[dp(12)])

        self.bind(pos=self._update_bg, size=self._update_bg)

        # --- Buttons ---
        self.btn_back = ImageButton(source="img/skip_back.png")
        self.btn_play = ImageButton(source="img/play.png")
        self.btn_pause = ImageButton(source="img/pause.png")
        self.btn_stop = ImageButton(source="img/stop.png") if False else None  # optional
        self.btn_forward = ImageButton(source="img/skip_forward.png")
        self.btn_loop = ImageButton(source="img/loop.png")

        buttons = [
            self.btn_back,
            self.btn_play,
            self.btn_pause,
            self.btn_forward,
            self.btn_loop,
        ]

        for b in buttons:
            b.size_hint = (None, None)
            b.size = (dp(32), dp(32))
            self.add_widget(b)

        # Bind button actions
        self.btn_play.bind(on_release=lambda *_: self._trigger("play"))
        self.btn_pause.bind(on_release=lambda *_: self._trigger("pause"))
        self.btn_back.bind(on_release=lambda *_: self._trigger("back"))
        self.btn_forward.bind(on_release=lambda *_: self._trigger("forward"))
        self.btn_loop.bind(on_release=lambda *_: self._trigger("loop"))

    # ---------------------------------------------------------
    # Trigger callbacks
    # ---------------------------------------------------------
    def _trigger(self, action):
        if action == "play" and self.on_play:
            self.on_play()
        elif action == "pause" and self.on_pause:
            self.on_pause()
        elif action == "back" and self.on_back:
            self.on_back()
        elif action == "forward" and self.on_forward:
            self.on_forward()
        elif action == "loop" and self.on_loop:
            self.on_loop()

    # ---------------------------------------------------------
    # Background sync
    # ---------------------------------------------------------
    def _update_bg(self, *args):
        self.bg.pos = self.pos
        self.bg.size = self.size
