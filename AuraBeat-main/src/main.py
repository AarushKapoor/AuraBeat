"""
AuraBeat — Video (top-left) + Circular Gear Menu + Right Piano Roll (¼ width)
- Fullscreen by default (F11 toggle, ESC exit). Preference is remembered.
- Right panel uses dusk gray background (no gradient), with razor-thin feathered boundary accents.
- Bottom mini keyboard with centered black keys (C#, D#, F#, G#, A#).
- Minimal circular gear button (⚙) to the left of the video feed opens a rounded quick menu.

Run:
    python aura_video_with_roll.py
"""

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import (Color, Rectangle, InstructionGroup, Line,
                           Ellipse, RoundedRectangle)
from kivy.graphics.texture import Texture
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.properties import NumericProperty, BooleanProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.label import Label


# --- Try to import your existing tracker ---
try:
    from hand_tracking.hands import HandTracker
    _HAS_TRACKER = True
except Exception as e:
    print("Could not import HandTracker:", e)
    _HAS_TRACKER = False


KV = r"""
#:kivy 2.3.0

<RootView>:
    orientation: "horizontal"
    spacing: dp(6)
    padding: dp(6)

    # Center stage area with minimized video pinned top-left and gear button to its left
    FloatLayout:
        id: stage
        canvas.before:
            Color:
                rgba: 0.05, 0.06, 0.08, 1
            Rectangle:
                pos: self.pos
                size: self.size

        # Circular gear button (⚙), anchored top-left with padding
        CircleButton:
            id: gear
            text: "⚙"
            size_hint: None, None
            size: dp(42), dp(42)
            # Use 'root' (the RootView of this rule) to compute positions
            pos: root.x + dp(8), root.top - dp(8) - self.height
            on_release: app.toggle_quick_menu()

        # Small video preview pinned to top-left, offset to the right of the gear button
        VideoFeed:
            id: video
            size_hint: None, None
            width: dp(420)
            height: self.width * 9/16
            # Top aligned; x starts just to the right of the gear button
            pos_hint: {"top": 1}
            x: root.ids.gear.right + dp(8)
            fit_mode: "contain"

        # Rounded quick menu (hidden by default), positioned to the right of the gear
        QuickMenu:
            id: quickmenu
            size_hint: None, None
            size: dp(260), dp(260)
            pos: root.ids.gear.right + dp(10), root.ids.gear.top - self.height - dp(4)
            visible: False  # toggled by app.toggle_quick_menu()

    # Right piano roll column — ~¼ width of the window
    PianoRollPanel:
        id: roll
        size_hint_x: 0.25
        keyboard_height_ratio: 0.18
        show_chevrons: True
"""


class RootView(BoxLayout):
    pass


class VideoFeed(Image):
    """
    Image widget with rounded corners using a stencil mask.
    - Set `corner_radius_dp` to control how round the corners are.
    - Optionally draw a subtle border (toggle `draw_border`).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.corner_radius_dp = dp(14)  # tweak to taste
        self.draw_border = True         # set False to hide the border

        # --- Build stencil mask in canvas.before so the Image's own
        # rectangle (drawn by Kivy) is clipped inside it.
        self._pre = InstructionGroup()
        self.canvas.before.add(self._pre)

        # --- Matching pop/unuse and border in canvas.after
        self._post = InstructionGroup()
        self.canvas.after.add(self._post)

        # Rebuild when size/pos changes
        self.bind(pos=self._rebuild_stencil, size=self._rebuild_stencil)
        self._rebuild_stencil()

    def _rebuild_stencil(self, *args):
        self._pre.clear()
        self._post.clear()

        # --- STENCIL: push & define rounded rect mask
        from kivy.graphics import StencilPush, StencilUse, StencilUnUse, StencilPop, RoundedRectangle, Color, Line

        # Push stencil
        self._pre.add(StencilPush())
        # Draw the rounded rect into stencil buffer
        self._pre.add(Color(1, 1, 1, 1))  # color doesn't matter for stencil
        self._mask = RoundedRectangle(
            pos=self.pos,
            size=self.size,
            radius=[self.corner_radius_dp] * 4
        )
        self._pre.add(self._mask)
        # Use the stencil for subsequent draws (the Image's texture)
        self._pre.add(StencilUse())

        # --- After the Image draws, remove stencil and optionally draw a border
        self._post.add(StencilUnUse())
        # Clear stencil buffer
        self._post.add(Color(1, 1, 1, 1))
        self._post.add(StencilPop())

        # Optional subtle border on top
        if self.draw_border:
            self._post.add(Color(1, 1, 1, 0.18))  # very subtle
            self._post.add(Line(rounded_rectangle=(
                self.x, self.y, self.width, self.height, self.corner_radius_dp
            ), width=dp(1.2)))

    def set_frame(self, rgb_frame: np.ndarray):
        h, w = rgb_frame.shape[:2]
        if not hasattr(self, "_texture") or self._texture is None or \
           self._texture.width != w or self._texture.height != h:
            texture = Texture.create(size=(w, h))
            texture.flip_vertical()
            self._texture = texture
        self.texture = self._texture
        self.texture.blit_buffer(rgb_frame.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
        self.canvas.ask_update()


class CircleButton(Button):
    """
    Minimal circular button. Uses a Unicode gear (⚙) for the icon.
    No external assets needed.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_normal = ""   # remove default background
        self.background_down = ""
        self.background_color = (0, 0, 0, 0)  # fully transparent, we draw our own
        self.color = (0.9, 0.95, 1, 1)
        self.font_size = dp(20)
        self.bold = True
        self._bg_instr = InstructionGroup()
        self.canvas.before.add(self._bg_instr)
        self.bind(pos=self._redraw_bg, size=self._redraw_bg)

    def _redraw_bg(self, *args):
        self._bg_instr.clear()
        r = min(self.width, self.height) / 2.0
        # Base circle
        self._bg_instr.add(Color(0.12, 0.14, 0.18, 1))
        self._bg_instr.add(Ellipse(pos=(self.x, self.y), size=(2*r, 2*r)))
        # Subtle border ring
        self._bg_instr.add(Color(1, 1, 1, 0.18))
        self._bg_instr.add(Line(circle=(self.center_x, self.center_y, r), width=dp(1.2)))


class QuickMenu(Widget):
    """
    Rounded rectangular quick menu that appears next to the gear button.
    Contains placeholder items. Toggle with `visible`.
    """
    visible = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._bg_instr = InstructionGroup()
        self.canvas.before.add(self._bg_instr)

        # internal layout: vertical column of placeholder items
        self._items = BoxLayout(orientation="vertical", spacing=dp(6),
                                padding=[dp(10), dp(10), dp(10), dp(10)])
        self.add_widget(self._items)
        # Placeholders (swap later)
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
        self._bg_instr.add(RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(12)]*4))
        # Place inner items snugly inside with padding
        self._items.pos = (self.x, self.y)
        self._items.size = (self.width, self.height)


class PianoRollPanel(Widget):
    """
    Right-side piano roll panel (Synthesia-inspired).
    - Background: dusk gray (very dark gray), no gradient
    - Razor-thin FEATHERED boundary accents (~1/3 track height, centered)
    - Bottom mini keyboard (7 white + 5 black), black keys centered
    - Optional green chevrons indicating scale shift
    """
    keyboard_height_ratio = NumericProperty(0.18)  # portion of widget height for the bottom keyboard
    show_chevrons = BooleanProperty(True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._g = InstructionGroup()
        self.canvas.add(self._g)
        self._border_tex = None  # alpha-feather texture for boundary accents
        # Redraw on size/pos changes
        self.bind(pos=self._redraw, size=self._redraw,
                  keyboard_height_ratio=self._redraw, show_chevrons=self._redraw)

    # Feather alpha texture: 0→1→0 vertically (triangular)
    def _ensure_border_texture(self):
        if self._border_tex is not None:
            return
        h, w = 256, 8
        y = np.linspace(0.0, 1.0, h, dtype=np.float32)
        alpha_line = 1.0 - np.abs(2.0 * y - 1.0)  # 0 at ends, 1 in middle
        color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        max_alpha = 0.85
        arr = np.zeros((h, w, 4), dtype=np.uint8)
        rgb = (color * 255).astype(np.uint8)
        arr[..., 0] = rgb[0]
        arr[..., 1] = rgb[1]
        arr[..., 2] = rgb[2]
        arr[..., 3] = (alpha_line[:, None] * max_alpha * 255).astype(np.uint8)
        # Flip for Kivy buffer orientation
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
        kb_h = max(dp(40), H * float(self.keyboard_height_ratio))  # ensure a sensible minimum keyboard height

        track_y = y0 + kb_h
        track_h = max(0, H - kb_h)

        # --- Dusk gray background (no gradient) for the whole panel ---
        self._g.add(Color(0.11, 0.12, 0.14, 1))  # dusk gray
        self._g.add(Rectangle(pos=(x0, y0), size=(W, H)))

        # --- Razor-thin FEATHERED boundary accents (centered, ~1/3 of track height) ---
        if track_h > 0:
            accent_len = max(dp(30), track_h / 3.0)
            yc = track_y + track_h / 2.0
            accent_y = yc - accent_len / 2.0
            razor = max(1.0, dp(1.2))  # razor-thin look

            self._g.add(Color(1, 1, 1, 1))  # color multiplier; texture alpha feathers it
            # Left accent
            self._g.add(Rectangle(pos=(x0, accent_y), size=(razor, accent_len), texture=self._border_tex))
            # Right accent
            self._g.add(Rectangle(pos=(x0 + W - razor, accent_y), size=(razor, accent_len), texture=self._border_tex))

        # --- Bottom keyboard base ---
        self._g.add(Color(0.07, 0.08, 0.10, 1))
        self._g.add(Rectangle(pos=(x0, y0), size=(W, kb_h)))

        # --- Bottom white keys (7) ---
        key_w = W / 7.0
        pad = dp(1.0)
        self._g.add(Color(0.90, 0.92, 0.96, 1))  # dim white
        for i in range(7):
            wx = x0 + i * key_w + pad
            wy = y0 + pad
            ww = max(0.0, key_w - 2 * pad)
            wh = max(0.0, kb_h - 2 * pad)
            if ww > 0 and wh > 0:
                self._g.add(Rectangle(pos=(wx, wy), size=(ww, wh)))

        # --- Black keys: centered between white keys (skip E–F and B–C) ---
        black_boundaries = [0, 1, 3, 4, 5]  # C-D, D-E, F-G, G-A, A-B
        bw = key_w * 0.56
        bh = kb_h * 0.62
        self._g.add(Color(0.06, 0.06, 0.09, 1))
        for j in black_boundaries:
            cx = x0 + (j + 1) * key_w
            bx = cx - bw / 2.0
            by = y0 + kb_h - bh
            if bw > 0 and bh > 0:
                self._g.add(Rectangle(pos=(bx, by), size=(bw, bh)))

        # --- Semi-transparent green chevrons (left/right) centered in track ---
        if self.show_chevrons and track_h > dp(20):
            self._g.add(Color(0.2, 1.0, 0.4, 0.45))  # semi-transparent green
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


class VideoController:
    """
    Background capture + (optional) tracking.
    Feeds annotated (or raw) frames into the VideoFeed.
    """
    def __init__(self, video_widget: VideoFeed, cam_index=0, use_tracker=True):
        self.video_widget = video_widget
        self.cam = cv2.VideoCapture(cam_index)
        if not self.cam.isOpened():
            raise RuntimeError("Could not open webcam (index 0). Try index 1 or close other apps using the camera.")
        self.running = False
        self.tracker = None

        if use_tracker and _HAS_TRACKER:
            try:
                self.tracker = HandTracker(
                    max_hands=2,
                    detection_confidence=0.7,
                    tracking_confidence=0.7,
                    # If your tracker uses the Tasks API and needs a local model path, uncomment:
                    # model_asset_path="models/hand_landmarker.task",
                )
                print("HandTracker initialized.")
            except Exception as e:
                print("Failed to create HandTracker:", e)
                self.tracker = None
        else:
            print("Running without HandTracker.")

    def start(self):
        self.running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self):
        self.running = False
        try:
            self.cam.release()
        except Exception:
            pass
        try:
            if self.tracker and hasattr(self.tracker, "close"):
                self.tracker.close()
        except Exception:
            pass

    def _loop(self):
        while self.running:
            ok, frame = self.cam.read()
            if not ok:
                time.sleep(0.02)
                continue

            annotated = frame
            if self.tracker:
                try:
                    _results, annotated = self.tracker.process(frame)
                except Exception as e:
                    print("Tracker error:", e)
                    annotated = frame

            # Convert BGR -> RGB and mirror for natural feel
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            rgb = cv2.flip(rgb, 1)

            # Push into UI thread
            Clock.schedule_once(lambda dt, im=rgb: self.video_widget.set_frame(im))

            time.sleep(0.001)


class AuraBeatVideoRollApp(App):
    """
    Remembers fullscreen preference and supports F11 toggle / ESC leave fullscreen.
    """
    CONFIG_FILE = "settings.json"

    def _config_path(self) -> Path:
        return Path(self.user_data_dir) / self.CONFIG_FILE

    def _load_config(self):
        p = self._config_path()
        try:
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            else:
                cfg = {}
        except Exception:
            cfg = {}
        self.fullscreen_pref = bool(cfg.get("fullscreen", True))

    def _save_config(self):
        p = self._config_path()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"fullscreen": bool(Window.fullscreen)}, f, indent=2)
        except Exception as e:
            print("Config save failed:", e)

    def build(self):
        # Load config and apply fullscreen BEFORE showing UI
        self._load_config()
        Window.fullscreen = 'auto' if self.fullscreen_pref else False

        Builder.load_string(KV)
        root = RootView()

        # ⬇️ FIX: 'video' id lives in RootView.ids, not stage.ids
        self.controller = VideoController(
            video_widget=root.ids.video,
            cam_index=0,
            use_tracker=True
        )

        # Start background capture
        Clock.schedule_once(lambda dt: self.controller.start(), 0)

        # Bind keys for fullscreen toggle / esc
        Window.bind(on_key_down=self._on_key_down)

        return root

    def _on_key_down(self, window, key, scancode, codepoint, modifiers):
        # F11 toggles fullscreen
        if key == 293:  # F11
            Window.fullscreen = False if Window.fullscreen else 'auto'
            self._save_config()
            return True
        # ESC exits fullscreen (if in fullscreen)
        if key == 27:   # ESC
            if Window.fullscreen:
                Window.fullscreen = False
                self._save_config()
                return True
        return False

    def toggle_quick_menu(self):
        # ⬇️ FIX: 'quickmenu' id also lives in RootView.ids
        qm = self.root.ids.quickmenu
        qm.visible = not qm.visible

    def on_stop(self):
        if hasattr(self, "controller"):
            self.controller.stop()


if __name__ == "__main__":
    AuraBeatVideoRollApp().run()