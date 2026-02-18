# src/gestures/classifiers.py
"""
Gesture classifiers used by the VideoController.

All functions expect a MediaPipe-style hand object with:
    hand.landmark[i].x
    hand.landmark[i].y

This file defines:
    - is_fist(hand)
    - is_thumbs_up(hand)
    - is_point(hand)
    - is_open(hand)
"""

import numpy as np

# MediaPipe Tasks landmark indices
WRIST       = 0
THUMB_TIP   = 4
INDEX_TIP   = 8
MIDDLE_TIP  = 12
RING_TIP    = 16
PINKY_TIP   = 20

INDEX_PIP   = 6
MIDDLE_PIP  = 10
RING_PIP    = 14
PINKY_PIP   = 18


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------

def _pt(lm, idx):
    """Return a 2D point for convenience."""
    return np.array([lm[idx].x, lm[idx].y], dtype=np.float32)

def _dist(a, b):
    """Euclidean distance in normalized screen space."""
    return float(np.linalg.norm(a - b))

def _finger_extended(lm, tip, pip) -> bool:
    """
    A finger is extended if TIP is farther from the wrist than the PIP joint.
    This uses a wrist-relative distance to reduce rotation sensitivity.
    """
    w = _pt(lm, WRIST)
    return _dist(_pt(lm, tip), w) > _dist(_pt(lm, pip), w)

def _thumb_extended(lm) -> bool:
    """
    Simplest thumb extension heuristic: tip farther from wrist than the joint.
    """
    w = _pt(lm, WRIST)
    tip = _pt(lm, THUMB_TIP)
    ip  = _pt(lm, THUMB_TIP - 1)   # landmark 3
    return _dist(tip, w) > _dist(ip, w)


# ---------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------

def is_open(hand) -> bool:
    """
    True when all four non-thumb fingers are extended.
    Thumb is ignored because some users keep it partially bent.
    """
    lm = hand.landmark
    return (
        _finger_extended(lm, INDEX_TIP,  INDEX_PIP) and
        _finger_extended(lm, MIDDLE_TIP, MIDDLE_PIP) and
        _finger_extended(lm, RING_TIP,   RING_PIP)   and
        _finger_extended(lm, PINKY_TIP,  PINKY_PIP)
    )


def is_fist(hand) -> bool:
    """
    True when all fingers are curled (none extended) AND thumb is not extended.
    This corresponds to ✊ Fist.
    """
    lm = hand.landmark

    curled = (
        not _finger_extended(lm, INDEX_TIP,  INDEX_PIP) and
        not _finger_extended(lm, MIDDLE_TIP, MIDDLE_PIP) and
        not _finger_extended(lm, RING_TIP,   RING_PIP) and
        not _finger_extended(lm, PINKY_TIP,  PINKY_PIP)
    )
    thumb_curled = not _thumb_extended(lm)

    return curled and thumb_curled


def is_point(hand) -> bool:
    """
    True when only the index finger is extended.
    Corresponds to ☝️.
    """
    lm = hand.landmark

    index_ext  = _finger_extended(lm, INDEX_TIP, INDEX_PIP)
    middle_cur = not _finger_extended(lm, MIDDLE_TIP, MIDDLE_PIP)
    ring_cur   = not _finger_extended(lm, RING_TIP,   RING_PIP)
    pinky_cur  = not _finger_extended(lm, PINKY_TIP,  PINKY_PIP)

    return index_ext and middle_cur and ring_cur and pinky_cur


def is_thumbs_up(hand) -> bool:
    """
    True when *only the thumb* is extended and other fingers are curled.

    This corresponds to 👍.

    We also include a mild vertical-orientation check:
        Thumb TIP above its IP joint in screen coords.
    This suppresses "thumbs up" false positives.
    """
    lm = hand.landmark

    thumb_ext = _thumb_extended(lm)

    others_curled = (
        not _finger_extended(lm, INDEX_TIP,  INDEX_PIP) and
        not _finger_extended(lm, MIDDLE_TIP, MIDDLE_PIP) and
        not _finger_extended(lm, RING_TIP,   RING_PIP) and
        not _finger_extended(lm, PINKY_TIP,  PINKY_PIP)
    )

    # Basic vertical check (y axis is downward in normalized coords)
    vertical_ok = lm[THUMB_TIP].y < lm[THUMB_TIP - 1].y

    return thumb_ext and others_curled and vertical_ok