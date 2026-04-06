# src/main.py
from kivy.app import App
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.factory import Factory as F
from kivy.clock import Clock
from kivy.logger import Logger

import os

print("DEBUG: main.py loaded")

# --- Project imports ---
try:
    from ui.kv import KV
    print("DEBUG: KV imported successfully")
except Exception as e:
    print("ERROR: Failed to import KV:", e)
    raise

try:
    from ui.widgets import (
        RootView, VideoFeed, CircleButton, QuickMenu, GestureHUD,
        PianoRollPanel, AirOverlayPanel, GradientBackground,
        LeftOptionsPanel, PillButton, UpperDock, LowerDock
    )
    print("DEBUG: ui.widgets imported successfully")
except Exception as e:
    print("ERROR: Failed to import widgets:", e)
    raise

# NEW: expanded piano roll popup
ExpandedPianoRollPopup = None
try:
    ExpandedPianoRollPopup = __import__(
        "ui.widgets.expanded_piano_roll",
        fromlist=["ExpandedPianoRollPopup"]
    ).ExpandedPianoRollPopup
    print("DEBUG: ExpandedPianoRollPopup imported:", ExpandedPianoRollPopup)
except Exception as e:
    print("ERROR: ExpandedPianoRollPopup import failed:", e)

# --- Optional subsystems ---
def safe_import(name, import_fn):
    try:
        obj = import_fn()
        print(f"DEBUG: {name} imported successfully")
        return obj
    except Exception as e:
        print(f"WARNING: {name} import failed:", e)
        return None

HandTracker = safe_import("HandTracker", lambda: __import__("hand_tracking.hands", fromlist=["HandTracker"]).HandTracker)
VideoController = safe_import("VideoController", lambda: __import__("hand_tracking.camera", fromlist=["VideoController"]).VideoController)
ScaleWindow = safe_import("ScaleWindow", lambda: __import__("mapping.scale_window", fromlist=["ScaleWindow"]).ScaleWindow)
AudioEngine = safe_import("AudioEngine", lambda: __import__("audio.engine", fromlist=["AudioEngine"]).AudioEngine)

Recorder = safe_import("Recorder", lambda: __import__("recording.recorder", fromlist=["Recorder"]).Recorder)
RecorderToPianoRollBridge = safe_import("RecorderToPianoRollBridge", lambda: __import__("recording.recorder_integration", fromlist=["RecorderToPianoRollBridge"]).RecorderToPianoRollBridge)
TimeGrid = safe_import("TimeGrid", lambda: __import__("playback.time_grid", fromlist=["TimeGrid"]).TimeGrid)
PlaybackEngine = safe_import("PlaybackEngine", lambda: __import__("playback.playback_engine", fromlist=["PlaybackEngine"]).PlaybackEngine)
PitchMapper = safe_import("PitchMapper", lambda: __import__("mapping.pitch_mapper", fromlist=["PitchMapper"]).PitchMapper)

KeySelectDialog = safe_import("KeySelectDialog", lambda: __import__("ui.widgets.key_select_dialog", fromlist=["KeySelectDialog"]).KeySelectDialog)


def _safe_register(name, cls):
    try:
        F.register(name, cls=cls)
        print(f"DEBUG: Registered widget {name}")
    except Exception as e:
        print(f"WARNING: Failed to register {name}:", e)


class AuraBeatApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("DEBUG: AuraBeatApp __init__")

        self.controller = None
        self.audio_engine = None

        self.recorder = None
        self.pitch_mapper = None
        self.time_grid = None
        self.piano_roll = None
        self.playback = None
        self.bridge = None

        self.expanded_popup = None

    def build(self):
        print("DEBUG: build() started")

        _safe_register("RootView", RootView)
        _safe_register("VideoFeed", VideoFeed)
        _safe_register("CircleButton", CircleButton)
        _safe_register("QuickMenu", QuickMenu)
        _safe_register("GestureHUD", GestureHUD)
        _safe_register("PianoRollPanel", PianoRollPanel)
        _safe_register("AirOverlayPanel", AirOverlayPanel)

        try:
            Builder.load_string(KV)
            print("DEBUG: KV loaded successfully")
        except Exception as e:
            print("ERROR: KV load failed:", e)
            raise

        root = RootView()
        print("DEBUG: RootView created")

        required_ids = ["video", "overlay", "pianoroll"]
        for rid in required_ids:
            print("DEBUG: Checking id:", rid)
        missing = [w for w in required_ids if w not in root.ids]
        if missing:
            print("ERROR: Missing required ids:", missing)
            raise RuntimeError("Missing required ids")

        # Audio engine
        if AudioEngine:
            try:
                self.audio_engine = AudioEngine(sr=48000, buffersize=256)
                self.audio_engine.start()
                print("DEBUG: AudioEngine started")
            except Exception as e:
                print("ERROR: AudioEngine failed:", e)

        # Hand tracker
        tracker = None
        if HandTracker:
            try:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(base_dir, "models", "hand_landmarker.task")
                print("DEBUG: Hand model path:", model_path)
                if os.path.exists(model_path):
                    tracker = HandTracker(
                        max_hands=2,
                        detection_confidence=0.7,
                        tracking_confidence=0.7,
                        use_flip=False,
                        model_asset_path=model_path,
                        running_mode="VIDEO",
                    )
                    print("DEBUG: HandTracker initialized")
            except Exception as e:
                print("ERROR: HandTracker failed:", e)

        # Scale window
        scale = None
        if ScaleWindow:
            try:
                scale = ScaleWindow.create_c_major()
                print("DEBUG: ScaleWindow created")
            except Exception as e:
                print("ERROR: ScaleWindow failed:", e)

        # DAW subsystem
        if PitchMapper and Recorder and TimeGrid and PlaybackEngine and RecorderToPianoRollBridge:
            try:
                self.pitch_mapper = PitchMapper(scale)
                self.recorder = Recorder(self.pitch_mapper)
                self.time_grid = TimeGrid(pixels_per_second=120)
                self.piano_roll = root.ids["pianoroll"]
                self.piano_roll.time_grid = self.time_grid

                print("DEBUG: PianoRollPanel instance:", self.piano_roll)

                self.playback = PlaybackEngine(
                    time_grid=self.time_grid,
                    note_canvas=self.piano_roll.note_canvas,
                    scroll_view=self.piano_roll.scroll,
                    audio_interface=self.audio_engine,
                )

                print("DEBUG: PlaybackEngine initialized")

                self.bridge = RecorderToPianoRollBridge(
                    recorder=self.recorder,
                    piano_roll=self.piano_roll,
                    time_grid=self.time_grid,
                    pitch_mapper=self.pitch_mapper
                )
                print("DEBUG: RecorderToPianoRollBridge initialized")
            except Exception as e:
                print("ERROR: DAW subsystem failed:", e)

        # Video controller
        if VideoController:
            try:
                self.controller = VideoController(
                    video_widget=root.ids["video"],
                    overlay_widget=root.ids["overlay"],
                    hand_tracker=tracker,
                    scale=scale,
                    audio_engine=self.audio_engine,
                    cam_index=0,
                )
                print("DEBUG: VideoController created")

                if self.controller:
                    self.controller.recorder = self.recorder
                    self.controller.pitch_mapper = self.pitch_mapper
                    Clock.schedule_once(lambda dt: self._start_controller_safe(), 0)
                    print("DEBUG: VideoController scheduled to start")
            except Exception as e:
                print("ERROR: VideoController failed:", e)

        Window.bind(on_key_down=self._on_key_down)
        print("DEBUG: Window.bind(on_key_down) attached")

        print("DEBUG: build() completed")
        return root

    def _start_controller_safe(self):
        try:
            if self.controller:
                self.controller.start()
                print("DEBUG: VideoController started")
        except Exception as e:
            print("ERROR: VideoController.start() failed:", e)

    # ============================================================
    # KEY HANDLER (debug version)
    # ============================================================
    def _on_key_down(self, window, key, scancode, codepoint, modifiers):
        print(f"DEBUG: key_down fired: key={key}, codepoint={codepoint}, mods={modifiers}")

        try:
            # F11
            if key == 293:
                print("DEBUG: F11 detected")
                Window.fullscreen = False if Window.fullscreen else 'auto'
                return True

            # ESC exits fullscreen
            if key == 27 and Window.fullscreen:
                print("DEBUG: ESC fullscreen exit")
                Window.fullscreen = False
                return True

            # Panic
            if codepoint and codepoint.lower() == 'p':
                print("DEBUG: Panic key detected")
                if self.audio_engine:
                    self.audio_engine.panic()
                return True

            # F key detection (robust)
            if key in (70, 102) or (codepoint and codepoint.lower() == 'f'):
                print("DEBUG: F key detected")

                if ExpandedPianoRollPopup:
                    if self.expanded_popup is None:
                        print("DEBUG: Creating popup instance")
                        self.expanded_popup = ExpandedPianoRollPopup(
                            piano_roll=self.piano_roll,
                            time_grid=self.time_grid,
                            pitch_mapper=self.pitch_mapper
                        )
                        print("DEBUG: Calling popup.open()")
                        self.expanded_popup.open()
                    else:
                        print("DEBUG: Closing popup")
                        self.expanded_popup.dismiss()
                        self.expanded_popup = None
                else:
                    print("ERROR: ExpandedPianoRollPopup is None")

                return True

        except Exception as e:
            print("ERROR in _on_key_down:", e)

        return False

    # ============================================================
    # Record toggle
    # ============================================================
    def toggle_record(self):
        if not self.recorder or not self.bridge:
            print("Recorder or bridge not initialized")
            return

        # Start recording
        if not getattr(self.recorder, "is_recording", False):
            self.recorder.start()
            print("Recording started")
            return

        # Stop recording
        self.recorder.stop()
        print("Recorded events (first 4):")
        for ev in self.recorder.events[:4]:
            print(f"{ev.pitch}  start={ev.start:.3f}  end={ev.end:.3f}  {ev.hand}-{ev.finger}")

        print("Recording stopped")

        # Push events into the piano roll
        try:
            self.bridge.apply_recorded_events()
        except Exception as e:
            print("Failed to apply recorded events:", e)

    # ============================================================
    # Open finger → key mapping dialog
    # ============================================================
    def open_finger_mapping_dialog(self, hand, finger):
        print(f"DEBUG: Opening KeySelectDialog for {hand} {finger}")

        if not KeySelectDialog:
            print("ERROR: KeySelectDialog class not available")
            return

        try:
            dlg = KeySelectDialog(
                hand=hand,
                finger=finger,
                pitch_mapper=self.pitch_mapper,
                on_apply=lambda: self._after_mapping_change()
            )
            dlg.open()
        except Exception as e:
            print("ERROR: Failed to open KeySelectDialog:", e)

    def _after_mapping_change(self):
        """
        Called after the user selects a new key.
        Refresh overlay labels and redraw piano roll if needed.
        """
        print("DEBUG: Mapping changed → refreshing overlay labels")

        try:
            overlay = self.root.ids.get("overlay")
            if overlay:
                overlay.refresh_labels()
        except Exception as e:
            print("ERROR: Failed to refresh overlay labels:", e)



if __name__ == "__main__":
    print("DEBUG: Running AuraBeatApp")
    AuraBeatApp().run()
