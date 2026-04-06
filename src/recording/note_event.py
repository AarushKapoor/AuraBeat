class NoteEvent:
    def __init__(self, pitch, start, end=None, velocity=1.0, hand=None, finger=None):
        self.pitch = pitch
        self.start = start
        self.end = end
        self.velocity = velocity
        self.hand = hand
        self.finger = finger

    @property
    def duration(self):
        return None if self.end is None else self.end - self.start
