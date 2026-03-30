# src/hand_tracking/camera.py
import sys
import time
import threading
from typing import Optional, Tuple, Any

import cv2
import numpy as np
from kivy.clock import Clock

from gestures.classifiers import is_fist, is_thumbs_up
from gestures.temporal import HysteresisFlag
from gestures.finger_press import FingerPress

from mapping.finger_ids import (
    TIP_TO_PIP,
    TIP_TO_NAME,
    RIGHT_PLAY_ORDER,
    LEFT_PLAY_ORDER,
)

from mapping.scale_window import midi_to_name

THUMB_TIP = 4


class VideoController:
    """
    Webcam -> (optional) hand tracking -> video widget + overlay widget.

    Preserves your:
      - whole-hand gestures (fist/thumb) for scale window shifts
      - per-finger press detection -> note on/off
      - overlay.update_model(ll, lp, rl, rp, lf, lt, rf, rt,
                             left_present=..., right_present=...)

    Adds:
      - Windows backend selection + MJPG (often better FPS)
      - Attempts 1280x720, then 1920x1080 16:9; otherwise center-crops to 16:9
      - Low-latency buffer where supported
      - Mild FPS pacing to avoid pegging a CPU core
      - PERMANENT note labels pushed every frame from current scale blocks
      - Hand presence "green dots" now reflect a debounced **hand-up** state:
          hand-up := (hand detected AND NOT is_fist(hand))
      - Persistent pressed state: notes hold as long as finger stays past the
          threshold line, not just for the single frame the "on" event fires.
    """

    def __init__(
        self,
        video_widget: Any,
        overlay_widget: Any,
        hand_tracker: Optional[Any],
        scale: Optional[Any],
        audio_engine: Optional[Any] = None,
        cam_index: int = 0,
        target_fps: float = 30.0,
    ):
        self.video_widget = video_widget
        self.overlay = overlay_widget
        self.tracker = hand_tracker
        self.scale = scale
        self.audio_engine = audio_engine

        # --- Prefer a backend that allows mode selection (esp. on Windows) ---
        if sys.platform.startswith("win"):
            # CAP_DSHOW tends to allow MJPG selection + decent latency
            self.cam = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        else:
            self.cam = cv2.VideoCapture(cam_index)

        if not self.cam.isOpened():
            raise RuntimeError(f"Could not open webcam (index {cam_index}).")

        # Reduce latency if backend supports it
        try:
            self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # On Windows, MJPG often enables 1280x720/1920x1080 at usable FPS.
        try:
            if sys.platform.startswith("win"):
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self.cam.set(cv2.CAP_PROP_FOURCC, fourcc)
        except Exception:
            pass

        # --- Try to request 16:9 capture (720p, then 1080p) ---
        def _try_mode(w: int, h: int) -> bool:
            ok1 = self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, float(w))
            ok2 = self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))
            return ok1 and ok2

        requested_modes = [(1280, 720), (1920, 1080)]
        applied_mode: Optional[Tuple[int, int]] = None
        for (w_req, h_req) in requested_modes:
            if _try_mode(w_req, h_req):
                ok, frame = self.cam.read()
                if ok and frame is not None and frame.size > 0:
                    hh, ww = frame.shape[:2]
                    # Confirm ~16:9 within 2% tolerance
                    if abs((ww / float(hh)) - (16 / 9)) < 0.02:
                        applied_mode = (ww, hh)
                        break

        # If not confirmed 16:9, we'll crop on the fly in the loop.
        self._crop_to_16x9 = applied_mode is None

        self.running = False
        self._ts0: Optional[float] = None
        self._last_good_frame: Optional[np.ndarray] = None

        # whole-hand hysteresis (gestures)
        self.fist_L = HysteresisFlag();  self.thumb_L = HysteresisFlag()
        self.fist_R = HysteresisFlag();  self.thumb_R = HysteresisFlag()
        self._lf_prev = self._lt_prev = self._rf_prev = self._rt_prev = False

        # finger press detectors (per-hand), thumb flagged for lateral-curl axis
        self.det_left = {
            tip: FingerPress(tip, TIP_TO_PIP[tip], is_thumb=(tip == THUMB_TIP))
            for tip in TIP_TO_PIP.keys()
        }
        self.det_right = {
            tip: FingerPress(tip, TIP_TO_PIP[tip], is_thumb=(tip == THUMB_TIP))
            for tip in TIP_TO_PIP.keys()
        }

        # Persistent pressed state — updated only on "on"/"off" events so that
        # notes hold for as long as the finger stays past the threshold line.
        _finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
        self._pressed_left  = {n: False for n in _finger_names}
        self._pressed_right = {n: False for n in _finger_names}

        # presence (hand-up) debounce state
        self._up_left = False
        self._up_right = False
        self._upL_on = 0; self._upL_off = 0
        self._upR_on = 0; self._upR_off = 0
        self._PRES_ON_FRAMES = 2       # require N consecutive frames to turn ON
        self._PRES_OFF_FRAMES = 3      # require M consecutive frames to turn OFF

        # pacing
        self._target_dt = 1.0 / float(max(1.0, target_fps))
        self._thread: Optional[threading.Thread] = None

    # ---- Timebase ------------------------------------------------------------

    def _mono_ms(self) -> int:
        now = time.perf_counter()
        if self._ts0 is None:
            self._ts0 = now
        return int((now - self._ts0) * 1000.0)

    # ---- Safe MIDI emitters --------------------------------------------------

    def _note_on(self, midi: int, velocity: int = 100, tag: Optional[str] = None):
        if self.audio_engine is None:
            return
        if hasattr(self.audio_engine, "note_on"):
            try:
                if tag is not None:
                    self.audio_engine.note_on(midi, velocity, tag=tag)
                else:
                    self.audio_engine.note_on(midi, velocity)
            except Exception:
                pass

    def _note_off(self, midi: int, tag: Optional[str] = None):
        if self.audio_engine is None:
            return
        if hasattr(self.audio_engine, "note_off"):
            try:
                if tag is not None:
                    self.audio_engine.note_off(midi, tag=tag)
                else:
                    self.audio_engine.note_off(midi)
            except Exception:
                pass

    # ---- Lifecycle -----------------------------------------------------------

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, name="VideoControllerLoop", daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        try:
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.cam.release()
        except Exception:
            pass
        try:
            if self.tracker and hasattr(self.tracker, "close"):
                self.tracker.close()
        except Exception:
            pass
        try:
            if self.audio_engine and hasattr(self.audio_engine, "panic"):
                self.audio_engine.panic()
        except Exception:
            pass

    # ---- Helpers -------------------------------------------------------------

    @staticmethod
    def _center_crop_16x9(frame_bgr: np.ndarray) -> np.ndarray:
        """Center-crop any frame to the largest 16:9 region that fits."""
        h, w = frame_bgr.shape[:2]
        target_ratio = 16.0 / 9.0
        cur_ratio = w / float(h)
        if abs(cur_ratio - target_ratio) < 0.02:
            return frame_bgr
        if cur_ratio > target_ratio:
            new_w = int(round(h * target_ratio))
            x0 = (w - new_w) // 2
            return frame_bgr[:, x0:x0 + new_w]
        else:
            new_h = int(round(w / target_ratio))
            y0 = (h - new_h) // 2
            return frame_bgr[y0:y0 + new_h, :]

    # ---- Main loop -----------------------------------------------------------

    def _loop(self):
        next_tick = time.perf_counter()
        while self.running:
            # Mild pacing to avoid 100% CPU
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(max(0.0, next_tick - now))
            next_tick = now + self._target_dt

            ok, frame = self.cam.read()
            if not ok or frame is None or frame.size == 0:
                if self._last_good_frame is None:
                    time.sleep(0.02)
                    continue
                frame = self._last_good_frame.copy()
            else:
                self._last_good_frame = frame

            # Force 16:9 if needed
            if self._crop_to_16x9:
                try:
                    frame = self._center_crop_16x9(frame)
                except Exception:
                    pass

            ts = self._mono_ms()

            results, annotated, meta = None, frame, None
            if self.tracker:
                try:
                    out = self.tracker.process(frame, timestamp_ms=ts)
                    if isinstance(out, tuple) and len(out) >= 3:
                        results, annotated, meta = out
                    elif isinstance(out, tuple) and len(out) == 2:
                        results, annotated = out
                    else:
                        annotated = frame
                except Exception:
                    annotated = frame

            # video -> UI (convert BGR->RGB and mirror for UI)
            try:
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                rgb = cv2.flip(rgb, 1)
            except Exception:
                if annotated.ndim == 3 and annotated.shape[2] == 3:
                    rgb = annotated[:, :, ::-1].copy()
                else:
                    rgb = annotated

            if not rgb.flags.c_contiguous:
                rgb = np.ascontiguousarray(rgb)

            Clock.schedule_once(lambda dt, im=rgb: self.video_widget.set_frame(im))

            # ---- Overlay state prep ------------------------------------------

            left_labels  = {n: "—" for n in ["Thumb", "Index", "Middle", "Ring", "Pinky"]}
            right_labels = {n: "—" for n in ["Thumb", "Index", "Middle", "Ring", "Pinky"]}

            # Seed pressed state from persistent store so held notes stay lit
            # in the overlay even on frames where no new event fires.
            left_pressed  = dict(self._pressed_left)
            right_pressed = dict(self._pressed_right)

            left_fist = left_thumb = right_fist = right_thumb = False

            # PERMANENT NOTE LABELS: always compute from scale blocks
            has_scale = self.scale is not None
            if has_scale:
                try:
                    left_block  = self.scale.left_block()
                    right_block = self.scale.right_block()
                    left_names  = [midi_to_name(m) for m in (left_block or [])]
                    right_names = [midi_to_name(m) for m in (right_block or [])]
                    fingers = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                    for i, nm in enumerate(fingers):
                        if i < len(left_names):
                            left_labels[nm] = left_names[i]
                        if i < len(right_names):
                            right_labels[nm] = right_names[i]
                except Exception:
                    pass

            # Raw "hand-up" booleans for this frame (before debounce)
            raw_up_left  = False
            raw_up_right = False

            # Handedness extraction
            handed = []
            if meta is not None and hasattr(meta, "handedness") and meta.handedness is not None:
                try:
                    for hlist in meta.handedness:
                        if not hlist:
                            handed.append("Right")
                        else:
                            first = hlist[0]
                            if isinstance(first, (list, tuple)) and len(first) > 0:
                                handed.append(str(first[0]))
                            elif hasattr(first, "category_name"):
                                handed.append(str(first.category_name))
                            else:
                                handed.append("Right")
                except Exception:
                    handed = []

            # ---- Per-hand processing ----------------------------------------
            if results is not None and hasattr(results, "multi_hand_landmarks") and results.multi_hand_landmarks:
                for i, hand in enumerate(results.multi_hand_landmarks):
                    side = handed[i] if i < len(handed) else ("Right" if i == 0 else "Left")

                    # Whole-hand gestures
                    try:
                        fnow = is_fist(hand)
                        tnow = is_thumbs_up(hand)
                    except Exception:
                        fnow = False
                        tnow = False

                    # hand-up: detected and NOT a fist
                    this_up = not fnow

                    if side == "Right":
                        raw_up_right = raw_up_right or this_up
                        rf = self.fist_R.update(fnow)
                        rt = self.thumb_R.update(tnow)
                        if has_scale:
                            if rf and not self._rf_prev:
                                try: self.scale.right_scale_up()
                                except Exception: pass
                            if rt and not self._rt_prev:
                                try: self.scale.right_scale_down()
                                except Exception: pass
                        self._rf_prev, self._rt_prev = rf, rt
                        right_fist, right_thumb = rf, rt
                        block = self.scale.right_block() if has_scale else []
                        order = RIGHT_PLAY_ORDER
                    else:
                        raw_up_left = raw_up_left or this_up
                        lf = self.fist_L.update(fnow)
                        lt = self.thumb_L.update(tnow)
                        if has_scale:
                            if lf and not self._lf_prev:
                                try: self.scale.left_scale_down()
                                except Exception: pass
                            if lt and not self._lt_prev:
                                try: self.scale.left_scale_up()
                                except Exception: pass
                        self._lf_prev, self._lt_prev = lf, lt
                        left_fist, left_thumb = lf, lt
                        block = self.scale.left_block() if has_scale else []
                        order = LEFT_PLAY_ORDER

                    # Refresh labels from current block
                    if has_scale and block:
                        try:
                            names = [midi_to_name(m) for m in block]
                            lab_map = dict(zip(["Thumb", "Index", "Middle", "Ring", "Pinky"], names))
                            if side == "Right":
                                right_labels.update(lab_map)
                            else:
                                left_labels.update(lab_map)
                        except Exception:
                            pass

                    # ---- Per-finger press -> note on/off --------------------
                    try:
                        dets       = self.det_right       if side == "Right" else self.det_left
                        pressed_map = right_pressed       if side == "Right" else left_pressed
                        persist    = self._pressed_right  if side == "Right" else self._pressed_left

                        if has_scale and block:
                            for idx, tip in enumerate(order):
                                ev = dets[tip].update(hand.landmark)
                                fname = TIP_TO_NAME[tip]
                                midi  = block[idx] if idx < len(block) else None
                                if midi is None:
                                    continue

                                tag = f"{'R' if side == 'Right' else 'L'}-{fname}"

                                if ev == "on":
                                    self._note_on(midi, 100, tag=tag)
                                    # Update both the per-frame map and the
                                    # persistent store so the note holds.
                                    pressed_map[fname] = True
                                    persist[fname]     = True
                                elif ev == "off":
                                    self._note_off(midi, tag=tag)
                                    pressed_map[fname] = False
                                    persist[fname]     = False
                    except Exception:
                        pass

            # ---- Debounce hand-up presence ----------------------------------
            if raw_up_left:
                self._upL_on += 1; self._upL_off = 0
                if self._upL_on >= self._PRES_ON_FRAMES:
                    self._up_left = True
            else:
                self._upL_off += 1; self._upL_on = 0
                if self._upL_off >= self._PRES_OFF_FRAMES:
                    self._up_left = False

            if raw_up_right:
                self._upR_on += 1; self._upR_off = 0
                if self._upR_on >= self._PRES_ON_FRAMES:
                    self._up_right = True
            else:
                self._upR_off += 1; self._upR_on = 0
                if self._upR_off >= self._PRES_OFF_FRAMES:
                    self._up_right = False

            # ---- Push overlay -----------------------------------------------
            if hasattr(self.overlay, "update_model"):
                def _push_overlay(dt,
                                  ll=left_labels,  lp=left_pressed,
                                  rl=right_labels, rp=right_pressed,
                                  lf=left_fist,    lt=left_thumb,
                                  rf=right_fist,   rt=right_thumb,
                                  lhp=self._up_left, rhp=self._up_right):
                    try:
                        self.overlay.update_model(ll, lp, rl, rp, lf, lt, rf, rt,
                                                  left_present=lhp, right_present=rhp)
                    except TypeError:
                        try:
                            self.overlay.update_model(ll, lp, rl, rp, lf, lt, rf, rt)
                        except Exception:
                            pass

                Clock.schedule_once(_push_overlay)

            time.sleep(0.001)
