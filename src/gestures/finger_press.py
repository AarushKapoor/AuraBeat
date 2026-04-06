# src/gestures/finger_press.py
from collections import deque
import numpy as np

# ── MediaPipe landmark indices ──────────────────────────────────────────────
WRIST       = 0
THUMB_CMC   = 1   # base of thumb
THUMB_MCP   = 2
THUMB_IP    = 3
THUMB_TIP   = 4

INDEX_MCP   = 5;  INDEX_TIP  = 8
MIDDLE_MCP  = 9;  MIDDLE_TIP = 12
RING_MCP    = 13; RING_TIP   = 16
PINKY_MCP   = 17; PINKY_TIP  = 20


def _pt(lm, idx) -> np.ndarray:
    return np.array([lm[idx].x, lm[idx].y], dtype=np.float32)


def _knuckle_line_normal(lm) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (point_on_line, unit_normal) for the knuckle line.
    The line runs through MCP joints 5-9-13-17.
    Normal points from knuckle line TOWARD fingertips (away from palm).
    """
    p5  = _pt(lm, INDEX_MCP)
    p17 = _pt(lm, PINKY_MCP)

    # Direction along the knuckle line (index → pinky)
    line_dir = p17 - p5
    length = np.linalg.norm(line_dir)
    if length < 1e-6:
        line_dir = np.array([1.0, 0.0], dtype=np.float32)
    else:
        line_dir /= length

    # Perpendicular (rotate 90°): two candidates
    perp = np.array([-line_dir[1], line_dir[0]], dtype=np.float32)

    # Make sure the normal points TOWARD fingertips (away from wrist)
    wrist = _pt(lm, WRIST)
    mid_knuckle = _pt(lm, MIDDLE_MCP)
    wrist_to_knuckle = mid_knuckle - wrist
    if np.dot(perp, wrist_to_knuckle) < 0:
        perp = -perp

    return p5, perp  # anchor point, unit normal pointing toward fingertips


def _signed_dist_from_knuckle_line(lm, tip_idx: int) -> float:
    """
    Positive = tip is on the FINGERTIP side of the knuckle line (extended).
    Negative = tip has crossed to the PALM side (pressed).
    """
    anchor, normal = _knuckle_line_normal(lm)
    tip = _pt(lm, tip_idx)
    return float(np.dot(tip - anchor, normal))


def _thumb_signed_dist(lm) -> float:
    """
    Thumb uses a separate axis: CMC(1) → Index MCP(5).
    Positive = thumb tip is on the 'open' side, negative = crossed inward.
    """
    cmc = _pt(lm, THUMB_CMC)
    idx_mcp = _pt(lm, INDEX_MCP)

    axis = idx_mcp - cmc
    length = np.linalg.norm(axis)
    if length < 1e-6:
        return 0.0
    axis /= length

    # Normal perpendicular to thumb axis, pointing away from palm
    perp = np.array([-axis[1], axis[0]], dtype=np.float32)
    wrist = _pt(lm, WRIST)
    if np.dot(perp, idx_mcp - wrist) < 0:
        perp = -perp

    tip = _pt(lm, THUMB_TIP)
    return float(np.dot(tip - cmc, perp))


class FingerPress:
    """
    Detects press/release by checking whether a fingertip has crossed
    the knuckle line (or thumb axis) into palm territory.

    Returns "on" when the tip crosses inward past the threshold,
    and "off" when it crosses back out — giving natural note hold.
    """

    # How far PAST the knuckle line (in normalized coords) before triggering.
    # Negative = into the palm. Tweak if too sensitive or not sensitive enough.
    PRESS_THRESHOLD  = 0.0   # cross this to trigger ON
    RELEASE_THRESHOLD = 0.0   # cross back this far to trigger OFF

    # Smoothing window to reduce jitter
    SMOOTH_FRAMES = 1

    def __init__(self, tip_id: int, pip_id: int, history: int = 6):
        self.tip_id = tip_id
        self.is_thumb = (tip_id == THUMB_TIP)
        self._dist_buf: deque[float] = deque(maxlen=self.SMOOTH_FRAMES)
        self.state = "idle"   # "idle" | "pressed"

    def _get_dist(self, lm) -> float:
        if self.is_thumb:
            return _thumb_signed_dist(lm)
        return _signed_dist_from_knuckle_line(lm, self.tip_id)

    def update(self, lm) -> str | None:
        """
        Call every frame with hand.landmark list.
        Returns "on", "off", or None.
        """
        raw = self._get_dist(lm)
        self._dist_buf.append(raw)

        # Smoothed distance
        dist = float(np.mean(self._dist_buf))

        ev = None

        if self.state == "idle":
            if dist < self.PRESS_THRESHOLD:
                self.state = "pressed"
                ev = "on"

        elif self.state == "pressed":
            if dist > self.RELEASE_THRESHOLD:
                self.state = "idle"
                ev = "off"

        return ev

    def reset(self):
        """Call this if the hand disappears mid-note."""
        self.state = "idle"
        self._dist_buf.clear()