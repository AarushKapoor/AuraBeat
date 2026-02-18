# src/hand_tracking/camera.py
import time
import cv2
from kivy.clock import Clock

# Import exact symbols you actually use
from gestures.classifiers import is_fist, is_thumbs_up
from gestures.temporal import HysteresisFlag
from gestures.finger_press import FingerPress

# Finger mapping & names/orders come from finger_ids
from mapping.finger_ids import (
    TIP_TO_PIP,
    TIP_TO_NAME,
    RIGHT_PLAY_ORDER,
    LEFT_PLAY_ORDER,
)

# For converting MIDI numbers to friendly names
from mapping.scale_window import midi_to_name

class VideoController:
    def __init__(self, video_widget, overlay_widget, hand_tracker, scale, audio_engine=None, cam_index=0):
        self.video_widget = video_widget
        self.overlay = overlay_widget
        self.tracker = hand_tracker
        self.scale = scale
        self.audio_engine = audio_engine

        self.cam = cv2.VideoCapture(cam_index)
        if not self.cam.isOpened():
            raise RuntimeError(f"Could not open webcam (index {cam_index}).")
        self.running = False
        self._ts0 = None

        # whole-hand hysteresis
        self.fist_L = HysteresisFlag();  self.thumb_L = HysteresisFlag()
        self.fist_R = HysteresisFlag();  self.thumb_R = HysteresisFlag()
        self._lf_prev = self._lt_prev = self._rf_prev = self._rt_prev = False

        # finger press detectors (per-hand)
        self.det_left  = {tip: FingerPress(tip, TIP_TO_PIP[tip]) for tip in TIP_TO_PIP.keys()}
        self.det_right = {tip: FingerPress(tip, TIP_TO_PIP[tip]) for tip in TIP_TO_PIP.keys()}

    def _mono_ms(self):
        now = time.perf_counter()
        if self._ts0 is None: self._ts0 = now
        return int((now - self._ts0) * 1000.0)

    # ---- Safe MIDI emitters (don’t crash if audio_engine is None or lacks methods) ----
    def _note_on(self, midi: int, velocity: int = 100):
        if self.audio_engine is None:
            return
        if hasattr(self.audio_engine, "note_on"):
            # Common API: audio_engine.note_on(midi, velocity)
            try:
                self.audio_engine.note_on(midi, velocity)
            except Exception:
                pass

    def _note_off(self, midi: int):
        if self.audio_engine is None:
            return
        if hasattr(self.audio_engine, "note_off"):
            # Common API: audio_engine.note_off(midi)
            try:
                self.audio_engine.note_off(midi)
            except Exception:
                pass

    def start(self):
        import threading
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

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

            ts = self._mono_ms()
            results, annotated, meta = None, frame, None
            if self.tracker:
                out = self.tracker.process(frame, timestamp_ms=ts)
                if isinstance(out, tuple) and len(out) >= 3:
                    results, annotated, meta = out
                elif isinstance(out, tuple) and len(out) == 2:
                    results, annotated = out

            # video → UI
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            rgb = cv2.flip(rgb, 1)
            Clock.schedule_once(lambda dt, im=rgb: self.video_widget.set_frame(im))

            # overlay defaults
            left_labels  = {n: "—" for n in ["Thumb", "Index", "Middle", "Ring", "Pinky"]}
            right_labels = {n: "—" for n in ["Thumb", "Index", "Middle", "Ring", "Pinky"]}
            left_pressed  = {n: False for n in left_labels}
            right_pressed = {n: False for n in right_labels}
            left_fist = left_thumb = right_fist = right_thumb = False

            handed = []
            if meta and getattr(meta, "handedness", None):
                for hlist in meta.handedness:
                    handed.append(hlist[0][0] if hlist else "Right")  # "Left"/"Right"

            if results and getattr(results, "multi_hand_landmarks", None):
                for i, hand in enumerate(results.multi_hand_landmarks):
                    side = handed[i] if i < len(handed) else ("Right" if i == 0 else "Left")

                    # whole-hand gestures → window shifts
                    fnow = is_fist(hand)
                    tnow = is_thumbs_up(hand)
                    if side == "Right":
                        rf = self.fist_R.update(fnow)
                        rt = self.thumb_R.update(tnow)
                        if rf and not self._rf_prev: self.scale.right_scale_up()
                        if rt and not self._rt_prev: self.scale.right_scale_down()
                        self._rf_prev, self._rt_prev = rf, rt
                        right_fist, right_thumb = rf, rt
                        block = self.scale.right_block()
                        order = RIGHT_PLAY_ORDER
                    else:
                        lf = self.fist_L.update(fnow)
                        lt = self.thumb_L.update(tnow)
                        if lf and not self._lf_prev: self.scale.left_scale_down()
                        if lt and not self._lt_prev: self.scale.left_scale_up()
                        self._lf_prev, self._lt_prev = lf, lt
                        left_fist, left_thumb = lf, lt
                        block = self.scale.left_block()
                        order = LEFT_PLAY_ORDER

                    # labels for Thumb..Pinky
                    names = [midi_to_name(m) for m in block]
                    lab_map = dict(zip(["Thumb", "Index", "Middle", "Ring", "Pinky"], names))
                    if side == "Right":
                        right_labels.update(lab_map)
                    else:
                        left_labels.update(lab_map)

                    # per-finger press → note on/off
                    dets, pressed_map = (self.det_right, right_pressed) if side == "Right" else (self.det_left, left_pressed)
                    for idx, tip in enumerate(order):
                        ev = dets[tip].update(hand.landmark)
                        fname = TIP_TO_NAME[tip]
                        midi = block[idx]
                        if ev == "on":
                            self._note_on(midi, 100)
                            pressed_map[fname] = True
                        elif ev == "off":
                            self._note_off(midi)
                            pressed_map[fname] = False

            # push overlay state
            Clock.schedule_once(
                lambda dt,
                ll=left_labels, lp=left_pressed,
                rl=right_labels, rp=right_pressed,
                lf=left_fist, lt=left_thumb,
                rf=right_fist, rt=right_thumb:
                    self.overlay.update_model(ll, lp, rl, rp, lf, lt, rf, rt)
            )

            time.sleep(0.001)