from kivy.uix.label import Label
from kivy.uix.behaviors import ButtonBehavior
from kivy.properties import StringProperty


class FingerLabel(ButtonBehavior, Label):
    """
    Clickable label above each finger dot.
    Carries hand + finger identity.
    """
    hand = StringProperty()
    finger = StringProperty()
