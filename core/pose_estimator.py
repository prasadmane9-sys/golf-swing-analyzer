"""MediaPipe Pose wrapper — extracts 33 landmark positions from a frame."""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np


@dataclass
class Landmark:
    x: float       # normalized [0,1] horizontal
    y: float       # normalized [0,1] vertical (0=top)
    z: float       # depth (relative, less reliable)
    visibility: float  # [0,1]

    def as_tuple(self):
        return (self.x, self.y)

    def as_tuple3(self):
        return (self.x, self.y, self.z)


# MediaPipe landmark IDs used throughout the project
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28

# Skeleton connections for drawing
POSE_CONNECTIONS = [
    (NOSE, LEFT_SHOULDER), (NOSE, RIGHT_SHOULDER),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
]


class PoseEstimator:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        import mediapipe as mp
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(self, frame_bgr: np.ndarray) -> Optional[List[Landmark]]:
        """
        Run pose estimation on a BGR frame.
        Returns list of 33 Landmark objects, or None if no pose detected.
        """
        import cv2
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._pose.process(frame_rgb)
        frame_rgb.flags.writeable = True

        if not results.pose_landmarks:
            return None

        return [
            Landmark(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=lm.visibility,
            )
            for lm in results.pose_landmarks.landmark
        ]

    def get_landmark(self, landmarks: List[Landmark], idx: int) -> Optional[Landmark]:
        """Return landmark by ID, or None if visibility < 0.3."""
        if landmarks is None or idx >= len(landmarks):
            return None
        lm = landmarks[idx]
        return lm if lm.visibility >= 0.3 else None

    def close(self):
        self._pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
