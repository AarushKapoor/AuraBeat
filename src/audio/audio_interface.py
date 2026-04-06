# src/audio/audio_interface.py

class AudioInterface:
    def __init__(self, engine):
        self.engine = engine

    def note_on(self, pitch, velocity=1.0):
        if self.engine:
            self.engine.note_on(pitch, velocity)

    def note_off(self, pitch):
        if self.engine:
            self.engine.note_off(pitch)
