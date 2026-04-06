# mapping/pitch_mapper.py
from mapping import finger_ids


class PitchMapper:
    """
    Maps (hand, finger) → MIDI pitch.

    Supports:
      - Default scale-based mapping (your original behavior)
      - Optional custom mapping per finger (full 88-key range)
    """

    def __init__(self, scale_window):
        self.scale = scale_window

        # NEW: custom mapping dictionary
        # key: (hand, finger)  e.g. ("left", "Thumb")
        # value: midi_pitch    e.g. 60
        self.custom_map = {}

    # ---------------------------------------------------------
    # Public API: set a custom pitch for a finger
    # ---------------------------------------------------------
    def set_custom_pitch(self, hand, finger, midi_pitch):
        """
        Assign a specific MIDI pitch to a given finger.
        Example:
            set_custom_pitch("left", "Thumb", 60)
        """
        self.custom_map[(hand, finger)] = int(midi_pitch)

    # ---------------------------------------------------------
    # Main pitch lookup
    # ---------------------------------------------------------
    def get_pitch(self, hand, finger):
        """
        Returns the pitch for this hand+finger.
        Priority:
            1. Custom mapping (if exists)
            2. Default scale-based mapping
        """
        key = (hand, finger)

        # 1. Custom mapping override
        if key in self.custom_map:
            return self.custom_map[key]

        # 2. Default scale-based mapping
        base = 48 if hand == "left" else 60
        offset = finger_ids.FINGER_TO_INDEX[finger]
        return base + self.scale.get_interval(offset)

    # ---------------------------------------------------------
    # Pitch → X coordinate for piano roll
    # ---------------------------------------------------------
    def pitch_to_x(self, pitch, panel_width):
        """
        Simple 7-key repeating layout.
        Replace later with full 88-key mapping if desired.
        """
        key_index = pitch % 7
        key_w = panel_width / 7.0
        return key_index * key_w + key_w * 0.1
