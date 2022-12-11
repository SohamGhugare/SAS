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

    def face_mesh(self, cap_idx: int, faces: int = 1):
        """Detect face using face mesh"""

        mp_face_mesh = mp.solutions.face_mesh
        drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        cap = cv2.VideoCapture(cap_idx)
        with mp_face_mesh.FaceMesh(
            max_num_faces = faces,
            refine_landmarks=True,
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5
        ) as face_mesh:
            while cap.isOpened():
                success, image = cap.read()
                
                # Checking for empty frames
                if not success:
                    print("[WARNING] Ignoring empty camera frame.")
                    continue
                
                # Performance boosting by marking the image as non-writeable
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)

                # Annotating the feed
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        self.mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing_styles
                            .get_default_face_mesh_tesselation_style())
                        self.mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing_styles
                            .get_default_face_mesh_contours_style())
                        self.mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=self.mp_drawing_styles
                            .get_default_face_mesh_iris_connections_style())
        

                cv2.imshow('Face Mesh', cv2.flip(image, 1))
                if cv2.waitKey(1) == ord("q"):
                    break

        cap.release()            