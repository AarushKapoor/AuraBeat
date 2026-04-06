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


class VideoController:
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
        self.recorder = None
        self.pitch_mapper = None
        self.muted = False

        if sys.platform.startswith("win"):
            self.cam = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        else:
            self.cam = cv2.VideoCapture(cam_index)

        if not self.cam.isOpened():
            raise RuntimeError(f"Could not open webcam (index {cam_index}).")

        try:
            self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        try:
            if sys.platform.startswith("win"):
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self.cam.set(cv2.CAP_PROP_FOURCC, fourcc)
        except Exception:
            pass

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
                    if abs((ww / float(hh)) - (16 / 9)) < 0.02:
                        applied_mode = (ww, hh)
                        break

        self._crop_to_16x9 = applied_mode is None

        self.running = False
        self._ts0: Optional[float] = None
        self._last_good_frame: Optional[np.ndarray] = None

        self.fist_L = HysteresisFlag();  self.thumb_L = HysteresisFlag()
        self.fist_R = HysteresisFlag();  self.thumb_R = HysteresisFlag()
        self._lf_prev = self._lt_prev = self._rf_prev = self._rt_prev = False

        self.det_left  = {tip: FingerPress(tip, TIP_TO_PIP[tip]) for tip in TIP_TO_PIP.keys()}
        self.det_right = {tip: FingerPress(tip, TIP_TO_PIP[tip]) for tip in TIP_TO_PIP.keys()}

        # presence debounce
        self._up_left = False
        self._up_right = False
        self._upL_on = 0; self._upL_off = 0
        self._upR_on = 0; self._upR_off = 0
        self._PRES_ON_FRAMES = 2
        self._PRES_OFF_FRAMES = 3

        # persistent pressed state (held across frames)
        self._held_left  = {n: False for n in ["Thumb", "Index", "Middle", "Ring", "Pinky"]}
        self._held_right = {n: False for n in ["Thumb", "Index", "Middle", "Ring", "Pinky"]}

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

            # ---- Overlay state prep ----
            left_labels  = {n: "—" for n in ["Thumb", "Index", "Middle", "Ring", "Pinky"]}
            right_labels = {n: "—" for n in ["Thumb", "Index", "Middle", "Ring", "Pinky"]}
            left_pressed  = dict(self._held_left)
            right_pressed = dict(self._held_right)
            left_fist = left_thumb = right_fist = right_thumb = False

            has_scale = self.scale is not None
            if has_scale:
                try:
                    left_block  = self.scale.left_block()
                    right_block = self.scale.right_block()
                    if self.pitch_mapper:
                        for nm in ["Thumb", "Index", "Middle", "Ring", "Pinky"]:
                            try:
                                midi = self.pitch_mapper.get_pitch("left", nm)
                                left_labels[nm] = midi_to_name(midi) if midi is not None else "—"
                            except Exception:
                                pass
                            try:
                                midi = self.pitch_mapper.get_pitch("right", nm)
                                right_labels[nm] = midi_to_name(midi) if midi is not None else "—"
                            except Exception:
                                pass
                except Exception:
                    pass

            raw_up_left  = False
            raw_up_right = False

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

            if results is not None and hasattr(results, "multi_hand_landmarks") and results.multi_hand_landmarks:
                for i, hand in enumerate(results.multi_hand_landmarks):
                    side = handed[i] if i < len(handed) else ("Right" if i == 0 else "Left")

                    try:
                        fnow = is_fist(hand)
                        tnow = is_thumbs_up(hand)
                    except Exception:
                        fnow = False
                        tnow = False

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

                    if self.pitch_mapper:
                        for nm in ["Thumb", "Index", "Middle", "Ring", "Pinky"]:
                            try:
                                midi = self.pitch_mapper.get_pitch("right" if side == "Right" else "left", nm)
                                name = midi_to_name(midi) if midi is not None else "—"
                                if side == "Right":
                                    right_labels[nm] = name
                                else:
                                    left_labels[nm] = name
                            except Exception:
                                pass

                    try:
                        dets = self.det_right if side == "Right" else self.det_left
                        pressed_map = right_pressed if side == "Right" else left_pressed
                        if has_scale and block:
                            for idx, tip in enumerate(order):
                                ev = dets[tip].update(hand.landmark)
                                fname = TIP_TO_NAME[tip]
                                midi = self.pitch_mapper.get_pitch("right" if side == "Right" else "left", fname)
                                if midi is None:
                                    continue

                                tag = f"{'R' if side == 'Right' else 'L'}-{fname}"

                                if ev == "on":
                                    if not self.muted:
                                        self._note_on(midi, 100, tag=tag)
                                    pressed_map[fname] = True
                                    if side == "Right":
                                        self._held_right[fname] = True
                                    else:
                                        self._held_left[fname] = True
                                    if self.recorder:
                                        self.recorder.update(side, fname, True)

                                elif ev == "off":
                                    if not self.muted:
                                        self._note_off(midi, tag=tag)
                                    pressed_map[fname] = False
                                    if side == "Right":
                                        self._held_right[fname] = False
                                    else:
                                        self._held_left[fname] = False
                                    if self.recorder:
                                        self.recorder.update(side, fname, False)

                    except Exception:
                        pass

            # Reset detectors for any hand that's no longer present
            if not raw_up_left:
                for det in self.det_left.values():
                    det.reset()
                for fname, held in self._held_left.items():
                    if held:
                        tag = f"L-{fname}"
                        midi = self.pitch_mapper.get_pitch("left", fname) if self.pitch_mapper else None
                        if midi is not None:
                            self._note_off(midi, tag=tag)
                self._held_left = {n: False for n in self._held_left}

            if not raw_up_right:
                for det in self.det_right.values():
                    det.reset()
                for fname, held in self._held_right.items():
                    if held:
                        tag = f"R-{fname}"
                        midi = self.pitch_mapper.get_pitch("right", fname) if self.pitch_mapper else None
                        if midi is not None:
                            self._note_off(midi, tag=tag)
                self._held_right = {n: False for n in self._held_right}

            # ---- Debounce "hand-up" presence ----
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

            if hasattr(self.overlay, "update_model"):
                def _push_overlay(dt,
                                  ll=left_labels, lp=left_pressed,
                                  rl=right_labels, rp=right_pressed,
                                  lf=left_fist, lt=left_thumb,
                                  rf=right_fist, rt=right_thumb,
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