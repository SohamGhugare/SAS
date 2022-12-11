import cv2
import mediapipe as mp

class Detection:
    """Class with all detection utility functions"""
    
    def __init__(self) -> None:
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def detect_face(self, cap_idx: int):
        """Detect face using face detection"""

        mp_face_detection = mp.solutions.face_detection

        cap = cv2.VideoCapture(cap_idx)
        with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        ) as face_detection:
            while cap.isOpened():
                success, image = cap.read()

                # Checking for empty frames
                if not success:
                    print("[WARNING] Ignoring empty camera frame.")
                    continue
                
                # Performance boosting by marking the image as non-writeable
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_detection.process(image)

                # Annotating the feed
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.detections:
                    for detection in results.detections:
                        self.mp_drawing.draw_detection(image, detection)

                cv2.imshow("Face Detection", cv2.flip(image, 1))
                if cv2.waitKey(1) == ord("q"):
                    break
        cap.release()

                