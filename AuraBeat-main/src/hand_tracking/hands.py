"""
AuraBeat - Hand Tracking Module (solutions + tasks compatible, local model)

Prefers the classic mp.solutions.hands API. If unavailable, falls back to the
Tasks API and loads the model from your local workspace:
    models/hand_landmarker.task
"""

import os
from pathlib import Path
import cv2


class _LandmarkShim:
    class _Hand:
        def __init__(self, landmarks):
            class _LM: pass
            self.landmark = []
            for (x, y, z) in landmarks:
                lm = _LM()
                lm.x, lm.y, lm.z = x, y, z
                self.landmark.append(lm)

    def __init__(self, hands_xyz):
        self.multi_hand_landmarks = [self._Hand(hand) for hand in hands_xyz]


class HandTracker:
    def __init__(self,
                 max_hands=2,
                 detection_confidence=0.7,
                 tracking_confidence=0.7,
                 model_complexity=1,
                 use_flip=False,
                 # default to your workspace model path
                 model_asset_path="models/hand_landmarker.task"):
        """
        Args:
            max_hands: Maximum hands to detect.
            detection_confidence: Min detection conf.
            tracking_confidence: Min tracking/presence conf.
            model_complexity: Solutions Hands complexity (0/1/2).
            use_flip: Flip frames horizontally inside tracker.
            model_asset_path: Path to the .task model for the Tasks API fallback.
                              Defaults to 'models/hand_landmarker.task' (relative to CWD).
        """
        self.max_hands = int(max_hands)
        self.det_conf = float(detection_confidence)
        self.trk_conf = float(tracking_confidence)
        self.model_complexity = int(model_complexity)
        self.use_flip = bool(use_flip)
        self.model_asset_path = model_asset_path

        import mediapipe as mp
        self._mp = mp
        self._mode = None
        self._hands = None
        self._connections = None
        self._draw = None
        self._styles = None

        # --- Prefer classic solutions API ---
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self._mode = "solutions"
            self._hands = mp.solutions.hands.Hands(
                max_num_hands=self.max_hands,
                min_detection_confidence=self.det_conf,
                min_tracking_confidence=self.trk_conf,
                model_complexity=self.model_complexity,
                static_image_mode=False,
            )
            self._connections = mp.solutions.hands.HAND_CONNECTIONS
            if hasattr(mp.solutions, "drawing_utils"):
                self._draw = mp.solutions.drawing_utils
                self._styles = getattr(mp.solutions, "drawing_styles", None)

        # --- Fallback to Tasks API (explicit local model) ---
        elif hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
            self._mode = "tasks"
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

            # Resolve model path relative to the current working directory.
            model_p = Path(self.model_asset_path).expanduser().resolve()
            if not model_p.exists():
                cwd = Path.cwd()
                raise FileNotFoundError(
                    "HandTracker (Tasks API) could not find the model file.\n"
                    f"Expected at: {model_p}\n"
                    f"Current working directory: {cwd}\n\n"
                    "Make sure you downloaded 'hand_landmarker.task' into your project:\n"
                    "    <project_root>/models/hand_landmarker.task\n"
                    "Or pass an absolute path via HandTracker(model_asset_path=...)."
                )

            base_options = mp_python.BaseOptions(model_asset_path=str(model_p))
            options = HandLandmarkerOptions(
                base_options=base_options,
                num_hands=self.max_hands,
                min_hand_detection_confidence=self.det_conf,
                min_hand_presence_confidence=self.trk_conf,
                min_tracking_confidence=self.trk_conf,
                running_mode=mp_vision.RunningMode.IMAGE,  # per-frame webcam is fine
            )
            self._hands = HandLandmarker.create_from_options(options)
            self._mp_vision = mp_vision

        else:
            raise RuntimeError(
                "Neither mp.solutions.hands nor mp.tasks.vision.HandLandmarker is available. "
                "Check your mediapipe installation."
            )

    # ----------------- helpers -----------------
    def _results_from_solutions(self, results):
        hands_xyz = []
        if results and results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                coords = [(float(lm.x), float(lm.y), float(getattr(lm, "z", 0.0)))
                          for lm in hand.landmark]
                hands_xyz.append(coords)
        return results, hands_xyz

    def _results_from_tasks(self, tasks_result):
        hands_xyz = []
        if tasks_result and getattr(tasks_result, "hand_landmarks", None):
            for hand_landmarks in tasks_result.hand_landmarks:
                # normalized coords in [0..1]
                coords = [(float(lm.x), float(lm.y), 0.0) for lm in hand_landmarks]
                hands_xyz.append(coords)
        return _LandmarkShim(hands_xyz), hands_xyz

    def _draw_landmarks(self, image_bgr, hands_xyz):
        if not hands_xyz:
            return
        h, w = image_bgr.shape[:2]
        # Use solutions drawer if available
        if self._draw is not None and self._connections is not None:
            for coords in hands_xyz:
                shim = _LandmarkShim([coords])
                hand = shim.multi_hand_landmarks[0]
                if self._styles:
                    self._draw.draw_landmarks(
                        image_bgr, hand, self._connections,
                        self._styles.get_default_hand_landmarks_style(),
                        self._styles.get_default_hand_connections_style(),
                    )
                else:
                    self._draw.draw_landmarks(image_bgr, hand, self._connections)
            return

        # Manual draw fallback
        chain = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20),
            (0,17),
        ]
        for coords in hands_xyz:
            pts = [(int(x * w), int(y * h)) for (x, y, _z) in coords]
            for (x, y) in pts:
                cv2.circle(image_bgr, (x, y), 3, (255, 255, 255), -1)
            for a, b in chain:
                cv2.line(image_bgr, pts[a], pts[b], (60, 240, 200), 2)

    # ----------------- main entry -----------------
    def process(self, frame_bgr):
        """
        Returns: (results_like, annotated_bgr)
        - results_like has .multi_hand_landmarks (solutions-compatible)
        """
        import mediapipe as mp  # ensure we have access to mp.Image
        frame = frame_bgr.copy()
        if self.use_flip:
            frame = cv2.flip(frame, 1)

        if self._mode == "solutions":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._hands.process(rgb)
            results_like, hands_xyz = self._results_from_solutions(results)
        else:
            # Use top-level mediapipe.Image and mediapipe.ImageFormat (NOT from tasks.vision)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
            results = self._hands.detect(mp_image)
            results_like, hands_xyz = self._results_from_tasks(results)

        annotated = frame.copy()
        self._draw_landmarks(annotated, hands_xyz)
        return results_like, annotated

    def close(self):
        try:
            if self._hands and hasattr(self._hands, "close"):
                self._hands.close()
        except Exception:
            pass