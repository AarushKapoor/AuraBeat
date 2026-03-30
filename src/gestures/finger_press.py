# src/gestures/finger_press.py
"""
Threshold-plane finger press detector.

For fingers (Index–Pinky):
    "Pressed" when the fingertip crosses below the knuckle line.
    The knuckle line is defined by the MCP row (landmarks 5, 9, 13, 17),
    projected onto the palm's down-axis in normalized 2D image space.

For the thumb:
    "Pressed" when the thumb tip curls laterally past the index MCP (landmark 5),
    measured along the palm's lateral axis in normalized 2D image space.

All distances are normalized by palm size (wrist → middle MCP) so the
detector is scale-invariant. A small margin and frame debounce prevent
single-frame flickers on the crossing boundary.
"""

from collections import deque
from typing import Literal
import numpy as np
from mapping.finger_ids import WRIST


# ---------- Landmark helpers ----------

def _pt(lm, idx) -> np.ndarray:
    return np.array([lm[idx].x, lm[idx].y], dtype=np.float32)


def _palm_size(lm) -> float:
    """Wrist → middle MCP distance (landmark 9). Used to normalize all distances."""
    return float(max(1e-6, np.linalg.norm(_pt(lm, 9) - _pt(lm, WRIST))))


def _palm_down_axis(lm) -> np.ndarray:
    """
    Unit vector pointing from the knuckle row toward the wrist (i.e. 'downward'
    relative to the palm). Fingers curl in this direction when pressing.

    Knuckle row centroid: average of MCP landmarks 5, 9, 13, 17.
    """
    mcps = np.mean([_pt(lm, i) for i in (5, 9, 13, 17)], axis=0)
    wrist = _pt(lm, WRIST)
    v = wrist - mcps
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.array([0.0, 1.0], dtype=np.float32)


def _palm_lateral_axis(lm) -> np.ndarray:
    """
    Unit vector pointing from pinky MCP (17) toward index MCP (5).
    Used as the thumb's crossing axis (lateral curl toward index side).
    """
    v = _pt(lm, 5) - _pt(lm, 17)
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.array([1.0, 0.0], dtype=np.float32)


def _knuckle_line_offset(lm, down_axis: np.ndarray) -> float:
    """
    Project the knuckle row centroid onto the down_axis.
    This is the baseline value that fingertips are compared against.
    """
    mcps = np.mean([_pt(lm, i) for i in (5, 9, 13, 17)], axis=0)
    return float(np.dot(mcps, down_axis))


def _tip_offset(lm, tip_id: int, axis: np.ndarray) -> float:
    """Project a fingertip onto the given axis."""
    return float(np.dot(_pt(lm, tip_id), axis))


# ---------- Detector ----------

class FingerPress:
    """
    Single-finger threshold-plane press detector.

    Parameters
    ----------
    tip_id : int
        MediaPipe landmark index of the fingertip.
    pip_id : int
        MediaPipe landmark index of the PIP joint (unused in threshold logic,
        kept for API compatibility with camera.py).
    is_thumb : bool
        If True, uses the lateral-curl axis instead of the down axis.
    margin : float
        Crossing margin as a fraction of palm size. Fingertip must exceed
        the knuckle line by this amount to count as pressed.
    thumb_margin : float
        Same concept for the thumb's lateral axis.
    debounce_on : int
        Consecutive frames the tip must be past the line before firing "on".
    debounce_off : int
        Consecutive frames the tip must be back before firing "off".
    """

    def __init__(
        self,
        tip_id: int,
        pip_id: int,
        is_thumb: bool = False,
        margin: float = 0.08,
        thumb_margin: float = 0.10,
        debounce_on: int = 2,
        debounce_off: int = 3,
    ):
        self.tip = tip_id
        self.pip = pip_id                  # kept for API compat
        self.is_thumb = is_thumb
        self.margin = margin
        self.thumb_margin = thumb_margin
        self._deb_on = debounce_on
        self._deb_off = debounce_off

        self.state: Literal["up", "down"] = "up"
        self._on_count = 0
        self._off_count = 0

    def update(self, lm) -> str | None:
        """
        Call once per frame with the full landmark list for one hand.

        Returns
        -------
        "on"  – fingertip just crossed the line (note start)
        "off" – fingertip just uncrossed the line (note end)
        None  – no state change this frame
        """
        palm = _palm_size(lm)

        if self.is_thumb:
            # Thumb: lateral curl toward index side
            lat = _palm_lateral_axis(lm)

            # Reference: index MCP (5) projected onto lateral axis
            ref = float(np.dot(_pt(lm, 5), lat))

            # Thumb tip projected onto the same axis
            tip_proj = _tip_offset(lm, self.tip, lat)

            # Thumb MCP (2) gives a natural zero-crossing anchor;
            # how far past the index MCP has the tip curled?
            crossed = (ref - tip_proj) / palm > self.thumb_margin

        else:
            # Fingers: downward curl past the knuckle line
            down = _palm_down_axis(lm)
            knuckle_proj = _knuckle_line_offset(lm, down)
            tip_proj = _tip_offset(lm, self.tip, down)

            # Tip is "below" the knuckle line when its down-axis projection
            # exceeds the knuckle centroid projection by more than margin*palm.
            crossed = (tip_proj - knuckle_proj) / palm > self.margin

        return self._debounce(crossed)

    def _debounce(self, crossed: bool) -> str | None:
        """
        Two independent counters (on / off) prevent single-frame flickers
        at the crossing boundary.
        """
        ev = None

        if crossed:
            self._on_count += 1
            self._off_count = 0
            if self.state == "up" and self._on_count >= self._deb_on:
                self.state = "down"
                ev = "on"
        else:
            self._off_count += 1
            self._on_count = 0
            if self.state == "down" and self._off_count >= self._deb_off:
                self.state = "up"
                ev = "off"

        return ev
