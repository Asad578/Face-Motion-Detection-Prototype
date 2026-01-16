from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
from mediapipe.tasks.python.core.base_options import BaseOptions
import os

class FaceMeshService:
    _instance = None

    def __init__(self):
        # Path to the MediaPipe face landmarker model
        model_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "models",
            "face_landmarker.task"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Face landmarker model not found: {model_path}")

        # Base options with model path
        base_opts = BaseOptions(model_asset_path=model_path)

        # Options for IMAGE mode (single frames, like in main.py)
        options = FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=RunningMode.IMAGE,
            num_faces=1
        )

        # Create the FaceLandmarker instance
        self.mesh = FaceLandmarker.create_from_options(options)

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = FaceMeshService()
        return cls._instance.mesh
