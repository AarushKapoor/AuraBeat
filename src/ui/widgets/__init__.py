# src/ui/widgets/__init__.py

from .video import (
    RootView,
    VideoFeed,
    CircleButton,
    QuickMenu,
    GestureHUD,
)

from .air_overlay import AirOverlayPanel
from .piano_roll import PianoRollPanel
from .controls import (
    PillButton,
    CircleIconButton,
)

from .layout import (
    GradientBackground,
    LeftOptionsPanel,
)

__all__ = [
    "RootView",
    "VideoFeed",
    "CircleButton",
    "QuickMenu",
    "GestureHUD",
    "AirOverlayPanel",
    "PianoRollPanel",
    "PillButton",
    "CircleIconButton",
    "UpperDock",
    "LowerDock",
    "GradientBackground",
    "LeftOptionsPanel",
]

from .docks import UpperDock, LowerDock
