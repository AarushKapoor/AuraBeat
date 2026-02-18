# src/gestures/temporal.py
from collections import deque

class HysteresisFlag:
    def __init__(self, maxlen=6, on_count=4, off_count=3):
        self.buf = deque(maxlen=maxlen)
        self.state = False
        self.on_count, self.off_count = on_count, off_count

    def update(self, now: bool) -> bool:
        self.buf.append(bool(now))
        ones = sum(self.buf); zeros = len(self.buf) - ones
        if ones >= self.on_count: self.state = True
        elif zeros >= self.off_count: self.state = False
        return self.state