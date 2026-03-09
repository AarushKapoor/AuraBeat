# src/gestures/finger_press.py
from collections import deque
import numpy as np
from mapping.finger_ids import WRIST

def _pt(lm, idx): return np.array([lm[idx].x, lm[idx].y], dtype=np.float32)
def _dist(a, b): return float(np.linalg.norm(a - b))

def palm_size(lm):
    return max(1e-6, _dist(_pt(lm, WRIST), _pt(lm, 9)))  # wrist→middle MCP

def finger_axis(lm, tip, pip):
    v = _pt(lm, tip) - _pt(lm, pip); n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.array([0.0, -1.0], dtype=np.float32)

def flexion_score(lm, tip, pip):
    w = _pt(lm, WRIST); s = palm_size(lm)
    tip_w = _dist(_pt(lm, tip), w) / s
    pip_w = _dist(_pt(lm, pip), w) / s
    return max(0.0, (pip_w - tip_w))

class FingerPress:
    def __init__(self, tip_id, pip_id, history=8):
        self.tip, self.pip = tip_id, pip_id
        self.xy = deque(maxlen=history)
        self.state = "idle"
        self.flex0 = 0.0  # baseline from calibration

    def update(self, lm):
        self.xy.append(tuple(_pt(lm, self.tip)))
        ax = finger_axis(lm, self.tip, self.pip)
        v = 0.0
        if len(self.xy) >= 3:
            p0 = np.array(self.xy[-3]); p1 = np.array(self.xy[-1])
            v = float(np.dot(p1 - p0, ax))            # inward negative
        flex = flexion_score(lm, self.tip, self.pip)  # 0..~1
        df = flex - self.flex0

        PRESS_VEL = 0.015
        PRESS_FLEX = 0.08
        RELEASE_FLEX = 0.04

        ev = None
        if self.state in ("idle","hover"):
            if (v < -PRESS_VEL) or (df > PRESS_FLEX):
                self.state = "pressed"; ev = "on"
            else:
                self.state = "hover"
        elif self.state in ("pressed","sustain"):
            if df < RELEASE_FLEX:
                self.state = "released"; ev = "off"
            else:
                self.state = "sustain"
        elif self.state == "released":
            if df < RELEASE_FLEX*0.5:
                self.state = "idle"
        return ev