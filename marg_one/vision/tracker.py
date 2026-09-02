"""
MARG-One Vision Subsystem: Dual-Hand 3D Kinematics and Skeleton Extractor.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from marg_one.vision.model_loader import ensure_model_downloaded, DEFAULT_MODEL_PATH


class LandmarkIndex:
    """Standard 21 MediaPipe hand landmark topological indices."""
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


# Skeleton anatomical bone connection pairs
HAND_CONNECTIONS: List[Tuple[int, int]] = [
    # Palm arch
    (LandmarkIndex.WRIST, LandmarkIndex.THUMB_CMC),
    (LandmarkIndex.WRIST, LandmarkIndex.INDEX_FINGER_MCP),
    (LandmarkIndex.WRIST, LandmarkIndex.PINKY_MCP),
    (LandmarkIndex.INDEX_FINGER_MCP, LandmarkIndex.MIDDLE_FINGER_MCP),
    (LandmarkIndex.MIDDLE_FINGER_MCP, LandmarkIndex.RING_FINGER_MCP),
    (LandmarkIndex.RING_FINGER_MCP, LandmarkIndex.PINKY_MCP),
    # Thumb digit
    (LandmarkIndex.THUMB_CMC, LandmarkIndex.THUMB_MCP),
    (LandmarkIndex.THUMB_MCP, LandmarkIndex.THUMB_IP),
    (LandmarkIndex.THUMB_IP, LandmarkIndex.THUMB_TIP),
    # Index digit
    (LandmarkIndex.INDEX_FINGER_MCP, LandmarkIndex.INDEX_FINGER_PIP),
    (LandmarkIndex.INDEX_FINGER_PIP, LandmarkIndex.INDEX_FINGER_DIP),
    (LandmarkIndex.INDEX_FINGER_DIP, LandmarkIndex.INDEX_FINGER_TIP),
    # Middle digit
    (LandmarkIndex.MIDDLE_FINGER_MCP, LandmarkIndex.MIDDLE_FINGER_PIP),
    (LandmarkIndex.MIDDLE_FINGER_PIP, LandmarkIndex.MIDDLE_FINGER_DIP),
    (LandmarkIndex.MIDDLE_FINGER_DIP, LandmarkIndex.MIDDLE_FINGER_TIP),
    # Ring digit
    (LandmarkIndex.RING_FINGER_MCP, LandmarkIndex.RING_FINGER_PIP),
    (LandmarkIndex.RING_FINGER_PIP, LandmarkIndex.RING_FINGER_DIP),
    (LandmarkIndex.RING_FINGER_DIP, LandmarkIndex.RING_FINGER_TIP),
    # Pinky digit
    (LandmarkIndex.PINKY_MCP, LandmarkIndex.PINKY_PIP),
    (LandmarkIndex.PINKY_PIP, LandmarkIndex.PINKY_DIP),
    (LandmarkIndex.PINKY_DIP, LandmarkIndex.PINKY_TIP),
]

FINGERTIP_INDICES: List[int] = [
    LandmarkIndex.THUMB_TIP,
    LandmarkIndex.INDEX_FINGER_TIP,
    LandmarkIndex.MIDDLE_FINGER_TIP,
    LandmarkIndex.RING_FINGER_TIP,
    LandmarkIndex.PINKY_TIP,
]


@dataclass
class HandLandmark:
    """Represents a single joint landmark point in image and normalized space."""
    index: int
    x: float          # Normalized X [0.0, 1.0]
    y: float          # Normalized Y [0.0, 1.0]
    z: float          # Normalized relative depth
    px: int           # Pixel coordinate X
    py: int           # Pixel coordinate Y
    name: str = ""


@dataclass
class HandData:
    """Structured telemetry data for a tracked hand instance."""
    handedness: str                                    # 'Left' or 'Right'
    score: float                                       # Detection/classification confidence [0.0, 1.0]
    landmarks: List[HandLandmark]                      # 21 spatial landmarks
    world_landmarks: List[Tuple[float, float, float]] # 3D coordinates in metric units
    bbox: Tuple[int, int, int, int]                    # (xmin, ymin, xmax, ymax) in pixels
    center: Tuple[int, int]                            # Geometric centroid (cx, cy)
    finger_states: Dict[str, bool] = field(default_factory=dict)  # {'thumb': bool, ...}

    def get_landmark_px(self, index: int) -> Tuple[int, int]:
        """Returns pixel (x, y) coordinates for landmark at index."""
        return (self.landmarks[index].px, self.landmarks[index].py)

    def get_landmark_norm(self, index: int) -> Tuple[float, float, float]:
        """Returns normalized (x, y, z) tuple for landmark at index."""
        lm = self.landmarks[index]
        return (lm.x, lm.y, lm.z)

    def distance_between(self, idx1: int, idx2: int, normalized: bool = True) -> float:
        """Computes Euclidean distance between two landmark points."""
        if normalized:
            lm1, lm2 = self.landmarks[idx1], self.landmarks[idx2]
            return math.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2 + (lm1.z - lm2.z)**2)
        lm1, lm2 = self.landmarks[idx1], self.landmarks[idx2]
        return math.hypot(lm1.px - lm2.px, lm1.py - lm2.py)

    def as_dict(self) -> Dict[str, Any]:
        """Serializes hand data for inter-module transport in MARG-One."""
        return {
            "handedness": self.handedness,
            "score": round(self.score, 4),
            "bbox": self.bbox,
            "center": self.center,
            "finger_states": self.finger_states,
            "landmarks": [
                {
                    "id": lm.index,
                    "x": round(lm.x, 5),
                    "y": round(lm.y, 5),
                    "z": round(lm.z, 5),
                    "px": lm.px,
                    "py": lm.py,
                    "name": lm.name,
                }
                for lm in self.landmarks
            ],
        }


class DualHandTracker:
    """
    Real-time multi-hand tracker and 3D skeleton builder for MARG-One.
    Supports continuous stream inference and feature extraction for up to 2 hands.
    """

    LANDMARK_NAMES = [
        "WRIST",
        "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
        "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
        "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
        "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
    ]

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        running_mode: vision.RunningMode = vision.RunningMode.IMAGE,
    ):
        self.model_path = ensure_model_downloaded(model_path)
        self.num_hands = num_hands
        self.running_mode = running_mode

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=self.num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=self.running_mode,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def _compute_finger_states(self, landmarks: List[HandLandmark]) -> Dict[str, bool]:
        """
        Determines binary extension state (True: Extended / False: Folded) for all 5 digits.
        Utilizes rotation-invariant distance ratio checks against wrist and MCP anchors.
        """
        wrist = landmarks[LandmarkIndex.WRIST]

        def dist(lm1: HandLandmark, lm2: HandLandmark) -> float:
            return math.hypot(lm1.px - lm2.px, lm1.py - lm2.py)

        # 4 Primary Digits (Index, Middle, Ring, Pinky)
        fingers = {
            "index": dist(wrist, landmarks[LandmarkIndex.INDEX_FINGER_TIP]) > dist(wrist, landmarks[LandmarkIndex.INDEX_FINGER_PIP]) * 1.15,
            "middle": dist(wrist, landmarks[LandmarkIndex.MIDDLE_FINGER_TIP]) > dist(wrist, landmarks[LandmarkIndex.MIDDLE_FINGER_PIP]) * 1.15,
            "ring": dist(wrist, landmarks[LandmarkIndex.RING_FINGER_TIP]) > dist(wrist, landmarks[LandmarkIndex.RING_FINGER_PIP]) * 1.15,
            "pinky": dist(wrist, landmarks[LandmarkIndex.PINKY_TIP]) > dist(wrist, landmarks[LandmarkIndex.PINKY_PIP]) * 1.15,
        }

        # Thumb Digit: evaluated relative to Pinky MCP lateral anchor
        pinky_mcp = landmarks[LandmarkIndex.PINKY_MCP]
        thumb_tip = landmarks[LandmarkIndex.THUMB_TIP]
        thumb_ip = landmarks[LandmarkIndex.THUMB_IP]
        fingers["thumb"] = dist(thumb_tip, pinky_mcp) > dist(thumb_ip, pinky_mcp) * 1.10

        return fingers

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: Optional[int] = None,
    ) -> List[HandData]:
        """
        Executes vision inference on an input frame and yields tracked HandData structures.
        """
        h, w, _ = frame_bgr.shape
        frame_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        if self.running_mode in (vision.RunningMode.LIVE_STREAM, vision.RunningMode.VIDEO):
            if timestamp_ms is None:
                raise ValueError("timestamp_ms must be provided in VIDEO/LIVE_STREAM mode.")
            result = self.detector.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self.detector.detect(mp_image)

        hands_data: List[HandData] = []
        if not result.hand_landmarks:
            return hands_data

        for i, landmarks_raw in enumerate(result.hand_landmarks):
            # Handedness and score
            handedness_label = "Unknown"
            score = 1.0
            if i < len(result.handedness) and len(result.handedness[i]) > 0:
                handedness_label = result.handedness[i][0].category_name
                score = result.handedness[i][0].score

            # Parse landmarks
            parsed_landmarks: List[HandLandmark] = []
            xs, ys = [], []
            for idx, lm in enumerate(landmarks_raw):
                px = max(0, min(w - 1, int(lm.x * w)))
                py = max(0, min(h - 1, int(lm.y * h)))
                xs.append(px)
                ys.append(py)
                name = self.LANDMARK_NAMES[idx] if idx < len(self.LANDMARK_NAMES) else f"LM_{idx}"
                parsed_landmarks.append(HandLandmark(
                    index=idx,
                    x=lm.x,
                    y=lm.y,
                    z=lm.z,
                    px=px,
                    py=py,
                    name=name
                ))

            # Bounding box calculation with boundary safeguards
            pad = 16
            xmin = max(0, min(xs) - pad)
            ymin = max(0, min(ys) - pad)
            xmax = min(w - 1, max(xs) + pad)
            ymax = min(h - 1, max(ys) + pad)
            bbox = (xmin, ymin, xmax, ymax)

            # Centroid
            cx = int(sum(xs) / len(xs))
            cy = int(sum(ys) / len(ys))

            # World metric coordinates
            world_lms: List[Tuple[float, float, float]] = []
            if i < len(result.hand_world_landmarks):
                for wlm in result.hand_world_landmarks[i]:
                    world_lms.append((wlm.x, wlm.y, wlm.z))

            # Finger kinematics
            finger_states = self._compute_finger_states(parsed_landmarks)

            hand = HandData(
                handedness=handedness_label,
                score=score,
                landmarks=parsed_landmarks,
                world_landmarks=world_lms,
                bbox=bbox,
                center=(cx, cy),
                finger_states=finger_states,
            )
            hands_data.append(hand)

        return hands_data

    def close(self):
        """Releases underlying model resources."""
        if hasattr(self, "detector") and self.detector:
            self.detector.close()
