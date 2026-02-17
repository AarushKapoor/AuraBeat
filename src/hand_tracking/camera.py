"""
AuraBeat - Camera Module

This module handles webcam capture using OpenCV.
It provides frames to the rest of the application.
"""

import cv2


class Camera:
    def __init__(self, device_index=0, width=640, height=480):
        """
        Initializes the webcam capture.

        Args:
            device_index (int): Index of the webcam (default 0).
            width (int): Frame width.
            height (int): Frame height.
        """

        self.cap = cv2.VideoCapture(device_index)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")

        # Optional: set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def get_frame(self):
        """
        Captures a single frame from the webcam.

        Returns:
            frame (numpy.ndarray): Captured BGR image frame.
        """

        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to read frame from webcam.")

        return frame

    def release(self):
        """
        Releases the webcam resource.
        """

        self.cap.release()
        cv2.destroyAllWindows()

