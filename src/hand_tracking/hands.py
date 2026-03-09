"""
AuraBeat - Hand Tracking Module (Tasks-only, new API compatible)

- Uses MediaPipe Tasks HandLandmarker exclusively.
- Supports IMAGE / VIDEO / LIVE_STREAM with the correct entry points.
- Returns a Solutions-like shim (.multi_hand_landmarks) for compatibility.
- Optionally fills landmark .z from world landmarks (if available).

Model expectation:
    models/hand_landmarker.task
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np


# ---------- Solutions-like shim ----------

class _LandmarkShim:
    """Wraps normalized (x,y,z) per hand to mimic mp.solutions structures."""
    class _Hand:
        def __init__(self, landmarks):
            class _LM:
                __slots__ = ("x", "y", "z")
            self.landmark = []
            for (x, y, z) in landmarks:
                lm = _LM()
                lm.x, lm.y, lm.z = float(x), float(y), float(z)
                self.landmark.append(lm)

    def __init__(self, hands_xyz):
        # hands_xyz: List[List[(x,y,z)]]
        self.multi_hand_landmarks = [self._Hand(hand) for hand in hands_xyz]


@dataclass
class TasksMeta:
    """Optional metadata alongside landmarks."""
    handedness: List[List[Tuple[str, float]]]      # [[("Left", 0.98)], [("Right", 0.96)]...]
    presence: List[Optional[List[Optional[float]]]]  # often None for HandLandmarker
    visibility: List[Optional[List[Optional[float]]]] # often None for HandLandmarker
    world_landmarks: Optional[List[List[Tuple[float, float, float]]]] = None  # world-space (x,y,z)


class HandTracker:
    def __init__(
        self,
        max_hands: int = 2,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.7,
        use_flip: bool = False,
        model_asset_path: str = "models/hand_landmarker.task",
        running_mode: str = "IMAGE",               # 'IMAGE' | 'VIDEO' | 'LIVE_STREAM'
        use_world_z: bool = True,                  # if True, populate .z from world landmarks (if available)
        live_result_callback: Optional[
            Callable[[object, "HandTracker", np.ndarray, int], None]
        ] = None,  # (result, tracker, frame_rgb, timestamp_ms) for LIVE_STREAM
    ):
        """
        Args:
            max_hands: Maximum hands to detect.
            detection_confidence: Min detection confidence.
            tracking_confidence: Min presence/tracking confidence.
            use_flip: Flip frames horizontally inside tracker (mirror UX).
            model_asset_path: Path to the .task model file.
            running_mode: 'IMAGE' | 'VIDEO' | 'LIVE_STREAM'.
            use_world_z: If True, use world-landmark z to fill shim.z (image x/y remain normalized).
            live_result_callback: Required when running_mode='LIVE_STREAM'.
        """
        self.max_hands = int(max_hands)
        self.det_conf = float(detection_confidence)
        self.trk_conf = float(tracking_confidence)
        self.use_flip = bool(use_flip)
        self.model_asset_path = model_asset_path
        self.running_mode = running_mode.upper()
        self.use_world_z = bool(use_world_z)
        self._live_callback = live_result_callback

        # ---- Import MediaPipe Tasks (vision) ----
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        self._mp_python = mp_python
        self._mp_vision = mp_vision

        # ---- Resolve and validate model path ----
        model_p = Path(self.model_asset_path).expanduser().resolve()
        if not model_p.exists():
            raise FileNotFoundError(
                "HandTracker (Tasks) could not find the model file.\n"
                f"Expected at: {model_p}\n"
                f"Current working directory: {Path.cwd()}\n\n"
                "Download 'hand_landmarker.task' into your project:\n"
                "    <project_root>/models/hand_landmarker.task\n"
                "Or pass an absolute path via HandTracker(model_asset_path=...)."
            )

        # ---- Configure running mode ----
        rm_map = {
            "IMAGE": mp_vision.RunningMode.IMAGE,
            "VIDEO": mp_vision.RunningMode.VIDEO,
            "LIVE_STREAM": mp_vision.RunningMode.LIVE_STREAM,
        }
        if self.running_mode not in rm_map:
            raise ValueError("running_mode must be one of: IMAGE, VIDEO, LIVE_STREAM")
        rm = rm_map[self.running_mode]

        # ---- Build options ----
        base_options = self._mp_python.BaseOptions(model_asset_path=str(model_p))

        if rm == mp_vision.RunningMode.LIVE_STREAM:
            if self._live_callback is None:
                raise ValueError("LIVE_STREAM mode requires live_result_callback=...")

            def _cb(result, output_image, timestamp_ms: int):
                # Forward to user callback. output_image is mp.Image in SRGB.
                self._live_callback(result, self, output_image.numpy_view(), timestamp_ms)

            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=self.max_hands,
                min_hand_detection_confidence=self.det_conf,
                min_hand_presence_confidence=self.trk_conf,
                min_tracking_confidence=self.trk_conf,
                running_mode=rm,
                result_callback=_cb,
            )
        else:
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=self.max_hands,
                min_hand_detection_confidence=self.det_conf,
                min_hand_presence_confidence=self.trk_conf,
                min_tracking_confidence=self.trk_conf,
                running_mode=rm,
            )

        # ---- Create landmarker ----
        self._hands = self._mp_vision.HandLandmarker.create_from_options(options)
        self._rm = rm  # store enum; the instance doesn't expose running_mode

    # --------- Internal helpers ---------

    def _extract_from_tasks(self, tasks_result):
        """
        Convert Tasks result -> (solutions_like, hands_xyz, meta)
        - hands_xyz uses (x,y) normalized image coords and z from either 0.0 or world landmarks (if enabled).
        """
        hands_xyz: List[List[Tuple[float, float, float]]] = []
        handedness_out: List[List[Tuple[str, float]]] = []
        presence_out: List[Optional[List[Optional[float]]]] = []
        visibility_out: List[Optional[List[Optional[float]]]] = []
        world_xyz: Optional[List[List[Tuple[float, float, float]]]] = None

        if tasks_result is None:
            return _LandmarkShim(hands_xyz), hands_xyz, TasksMeta([], [], [], None)

        # Gather world landmarks if present
        if getattr(tasks_result, "hand_world_landmarks", None):
            world_xyz = []
            for wl in tasks_result.hand_world_landmarks:
                world_xyz.append([(float(p.x), float(p.y), float(p.z)) for p in wl])

        # Image landmarks + handedness
        if getattr(tasks_result, "hand_landmarks", None):
            for i, hand_lms in enumerate(tasks_result.hand_landmarks):
                # Prefer z from world landmarks if enabled and available
                if self.use_world_z and world_xyz is not None and i < len(world_xyz):
                    z_vals = [zw for (_xw, _yw, zw) in world_xyz[i]]
                else:
                    z_vals = [0.0] * len(hand_lms)

                coords = [(float(lm.x), float(lm.y), float(z_vals[j])) for j, lm in enumerate(hand_lms)]
                hands_xyz.append(coords)

                labels: List[Tuple[str, float]] = []
                if getattr(tasks_result, "handedness", None):
                    clist = tasks_result.handedness[i]
                    if clist and getattr(clist[0], "category_name", None) is not None:
                        labels = [(c.category_name, float(c.score)) for c in clist]
                    elif clist and getattr(clist[0], "label", None) is not None:
                        labels = [(c.label, float(getattr(c, "score", 0.0))) for c in clist]
                handedness_out.append(labels)

                # These may not be populated by HandLandmarker; keep structure
                presence_out.append([getattr(lm, "presence", None) for lm in hand_lms])
                visibility_out.append([getattr(lm, "visibility", None) for lm in hand_lms])

        meta = TasksMeta(
            handedness=handedness_out,
            presence=presence_out,
            visibility=visibility_out,
            world_landmarks=world_xyz
        )
        return _LandmarkShim(hands_xyz), hands_xyz, meta

    @staticmethod
    def _draw_manual(image_bgr: np.ndarray, hands_xyz: List[List[Tuple[float, float, float]]]):
        """Manual drawing of landmarks and connections (normalized coords -> pixels)."""
        if not hands_xyz:
            return
        h, w = image_bgr.shape[:2]
        chain = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20),
            (0,17),
        ]
        thickness = max(2, int(0.003 * (w + h)))
        radius = max(3, int(0.004 * (w + h)))
        for coords in hands_xyz:
            pts = [(int(x * w), int(y * h)) for (x, y, _z) in coords]
            for a, b in chain:
                cv2.line(image_bgr, pts[a], pts[b], (60, 240, 200), thickness)
            for (x, y) in pts:
                cv2.circle(image_bgr, (x, y), radius, (255, 255, 255), -1)

    # --------- Public API ---------

    def process(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: Optional[int] = None,
    ):
        """
        IMAGE/VIDEO modes: synchronous detection + drawing.

        Args:
            frame_bgr: BGR frame (np.ndarray).
            timestamp_ms: Required if running_mode == 'VIDEO' (monotonic increasing).
                          Ignored for 'IMAGE'.

        Returns:
            results_like: Solutions-compatible with .multi_hand_landmarks
            annotated_bgr: frame with landmarks drawn
            meta: TasksMeta with handedness/world landmarks, etc.

        LIVE_STREAM: use send_async(...) instead.
        """
        # Use the stored enum (the instance does not expose running_mode)
        if self._rm == self._mp_vision.RunningMode.LIVE_STREAM:
            raise RuntimeError("In LIVE_STREAM mode, call send_async(...) instead of process().")

        frame = frame_bgr.copy()
        if self.use_flip:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        if self._rm == self._mp_vision.RunningMode.VIDEO:
            if timestamp_ms is None:
                raise ValueError("timestamp_ms is required in VIDEO mode.")
            tasks_result = self._hands.detect_for_video(mp_image, timestamp_ms)
        else:  # IMAGE
            tasks_result = self._hands.detect(mp_image)

        results_like, hands_xyz, meta = self._extract_from_tasks(tasks_result)

        annotated = frame.copy()
        self._draw_manual(annotated, hands_xyz)
        return results_like, annotated, meta

    def send_async(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: int,
    ):
        """
        LIVE_STREAM mode: enqueue a frame and return immediately.
        Results arrive via the callback passed at construction time.
        """
        if self._rm != self._mp_vision.RunningMode.LIVE_STREAM:
            raise RuntimeError("send_async() is only valid in LIVE_STREAM mode.")

        frame = frame_bgr.copy()
        if self.use_flip:
            frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        self._hands.detect_async(mp_image, timestamp_ms)

    def close(self):
        try:
            if self._hands and hasattr(self._hands, "close"):
                self._hands.close()
        except Exception:
            pass