from src.hand_tracking.camera import Camera
from src.hand_tracking.hands import HandTracker
import cv2

camera = Camera()
tracker = HandTracker()

while True:
    frame = camera.get_frame()
    results, annotated_frame = tracker.process(frame)

    cv2.imshow("AuraBeat", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
