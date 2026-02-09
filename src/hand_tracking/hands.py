"""
AuraBeat - Hand Tracking Module

This module handles real-time hand landmark detection using MediaPipe Hands.
It takes webcam frames as input and outputs detected hand landmarks along with
an annotated frame for visual feedback.
"""

import cv2
import mediapipe as mp


class HandTracker:
    def __init__(self,
                 max_hands=2,
                 detection_confidence=0.7,
                 tracking_confidence=0.7):
        """
        Initializes the MediaPipe Hands pipeline.

        Args:
            max_hands (int): Maximum number of hands to detect.
            detection_confidence (float): Minimum confidence for hand detection.
            tracking_confidence (float): Minimum confidence for hand tracking.
        """

        # MediaPipe Hands solution
        self.mp_hands = mp.solutions.hands

        # Create a Hands object that will perform detection and tracking
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

        # Utility for drawing landmarks on frames
        self.drawer = mp.solutions.drawing_utils

    def process(self, frame):
        """
        Processes a single video frame and detects hand landmarks.

        Args:
            frame (numpy.ndarray): BGR image frame from OpenCV.

        Returns:
            results: MediaPipe results object containing hand landmarks.
            frame: Annotated frame with hand landmarks drawn.
        """

        # Convert BGR (OpenCV default) to RGB (MediaPipe requirement)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run hand landmark detection
        results = self.hands.process(rgb_frame)

        # If hands are detected, draw landmarks on the original frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.drawer.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

        return results, frame
