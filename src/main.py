# src/main.py
from kivy.app import App
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.factory import Factory as F
from kivy.clock import Clock
from kivy.logger import Logger

import os

# --- Project imports ---
try:
    from ui.kv import KV
except Exception as e:
    raise ImportError(
        "Failed to import KV from ui.kv. Ensure ui/kv.py defines a variable named KV (str)."
    ) from e

try:
    from ui.widgets import (
        RootView, VideoFeed, CircleButton, QuickMenu, GestureHUD,
        PianoRollPanel, AirOverlayPanel
    )
except Exception as e:
    raise ImportError(
        "Failed to import one or more widgets from ui.widgets. "
        "Please ensure all classes exist and import side-effects don't fail."
    ) from e

# These may fail; we’ll log and keep the app window opening.
try:
    from hand_tracking.hands import HandTracker
except Exception as e:
    HandTracker = None
    Logger.warning(f"HandTracker import failed; continuing without tracker. Error: {e}")

try:
    from hand_tracking.camera import VideoController
except Exception as e:
    VideoController = None
    Logger.warning(f"VideoController import failed; continuing without camera. Error: {e}")

try:
    from mapping.scale_window import ScaleWindow
except Exception as e:
    ScaleWindow = None
    Logger.warning(f"ScaleWindow import failed; continuing without scale window. Error: {e}")


def _safe_register(name, cls):
    try:
        F.register(name, cls=cls)
    except Exception as e:
        Logger.debug(f"Factory.register('{name}') skipped (likely already registered): {e}")


class AuraBeatApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.controller = None

    def build(self):
        # Register widgets for KV lookups
        _safe_register("RootView", RootView)
        _safe_register("VideoFeed", VideoFeed)
        _safe_register("CircleButton", CircleButton)
        _safe_register("QuickMenu", QuickMenu)
        _safe_register("GestureHUD", GestureHUD)
        _safe_register("PianoRollPanel", PianoRollPanel)
        _safe_register("AirOverlayPanel", AirOverlayPanel)

        # Load KV templates/rules
        try:
            Builder.load_string(KV)
        except Exception as e:
            Logger.exception("Failed to load KV — check for syntax errors or missing properties.")
            raise

        # IMPORTANT: instantiate the root widget explicitly
        root = RootView()

        # Fullscreen pref
        try:
            Window.fullscreen = 'auto'
        except Exception as e:
            Logger.warning(f"Could not set fullscreen mode: {e}")

        # Validate required ids exist on the instantiated root
        required_ids = ["video", "overlay", "quickmenu"]
        missing = [w for w in required_ids if w not in root.ids]
        if missing:
            msg = (
                "The following required widget ids are missing in your KV: "
                f"{', '.join(missing)}.\n"
                "Ensure your <RootView> rule defines:\n"
                "    id: video  (VideoFeed)\n"
                "    id: overlay (AirOverlayPanel)\n"
                "    id: quickmenu (QuickMenu)\n"
            )
            Logger.critical(msg)
            raise RuntimeError(msg)

        # Build optional subsystems
        tracker = None
        model_path = None
        if HandTracker is not None:
            try:
                # Model path relative to this file's folder (src/)
                base_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(base_dir, "models", "hand_landmarker.task")
                if not os.path.exists(model_path):
                    Logger.warning(
                        f"Hand model not found at '{model_path}'. "
                        "Place 'hand_landmarker.task' there to enable hand tracking."
                    )
                else:
                    tracker = HandTracker(
                        max_hands=2,
                        detection_confidence=0.7,
                        tracking_confidence=0.7,
                        use_flip=False,  # we mirror in UI instead
                        model_asset_path=model_path,
                        running_mode="VIDEO",
                    )
            except Exception as e:
                Logger.exception(f"Failed to initialize HandTracker: {e}")
                tracker = None

        scale = None
        if ScaleWindow is not None:
            try:
                scale = ScaleWindow.create_c_major()
            except Exception as e:
                Logger.exception(f"Failed to construct ScaleWindow C major: {e}")
                scale = None

        # Controller: video + overlay updates (don’t block app if it fails)
        if VideoController is not None:
            try:
                self.controller = VideoController(
                    video_widget=root.ids.get("video"),
                    overlay_widget=root.ids.get("overlay"),
                    hand_tracker=tracker,
                    scale=scale,
                    audio_engine=None,  # plug your audio engine here
                    cam_index=0,
                )
                Clock.schedule_once(lambda dt: self._start_controller_safe(), 0)
            except Exception as e:
                Logger.exception(f"Failed to initialize VideoController: {e}")
                self.controller = None
        else:
            Logger.warning("VideoController not available; skipping camera startup.")

        # Key bindings
        Window.bind(on_key_down=self._on_key_down)
        return root

    def _start_controller_safe(self):
        if self.controller is None:
            Logger.warning("Controller is None; skipping start.")
            return
        try:
            self.controller.start()
            Logger.info("VideoController started.")
        except Exception as e:
            Logger.exception(f"VideoController.start() failed: {e}")

    def _on_key_down(self, window, key, scancode, codepoint, modifiers):
        try:
            # F11 toggles fullscreen
            if key == 293:
                Window.fullscreen = False if Window.fullscreen else 'auto'
                return True
            # ESC exits fullscreen
            if key == 27 and Window.fullscreen:
                Window.fullscreen = False
                return True
        except Exception as e:
            Logger.debug(f"Key handling error: {e}")
        return False

    def toggle_quick_menu(self):
        try:
            qm = self.root.ids.get("quickmenu") if self.root else None
            if qm is None:
                Logger.warning("QuickMenu id not found (quickmenu).")
                return
            qm.visible = not getattr(qm, "visible", False)
        except Exception as e:
            Logger.exception(f"toggle_quick_menu failed: {e}")

    def on_stop(self):
        try:
            if getattr(self, "controller", None):
                self.controller.stop()
        except Exception as e:
            Logger.debug(f"Issue stopping controller on app shutdown: {e}")


if __name__ == "__main__":
    AuraBeatApp().run()