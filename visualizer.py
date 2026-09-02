"""
Visualizer module for rendering hand skeletons, landmarks, bounding boxes, and HUD metrics.
"""

from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np

from hand_tracker import (
    HandData,
    HAND_CONNECTIONS,
    FINGERTIP_INDICES,
    LandmarkIndex
)


class VisualizerTheme:
    """Color palettes and styles for Left and Right hands (BGR colors)."""
    # Left Hand (Cyan / Aqua theme)
    LEFT_BONE_COLOR = (230, 216, 0)        # Bright cyan
    LEFT_JOINT_COLOR = (200, 180, 0)
    LEFT_TIP_COLOR = (255, 255, 255)       # White
    LEFT_BOX_COLOR = (230, 216, 0)
    LEFT_TEXT_BG = (180, 140, 0)

    # Right Hand (Coral / Orange-Pink theme)
    RIGHT_BONE_COLOR = (50, 130, 255)      # Bright orange/coral
    RIGHT_JOINT_COLOR = (30, 100, 230)
    RIGHT_TIP_COLOR = (255, 255, 255)
    RIGHT_BOX_COLOR = (50, 130, 255)
    RIGHT_TEXT_BG = (30, 90, 200)

    # General HUD
    HUD_BG = (25, 25, 30)
    HUD_TEXT = (240, 240, 240)
    ACCENT_GREEN = (80, 220, 100)
    ACCENT_RED = (80, 80, 240)
    ACCENT_PURPLE = (220, 100, 200)


class HandVisualizer:
    """
    Renders 3D hand skeletons, landmark points, bounding boxes, and diagnostic HUD on OpenCV frames.
    """

    def __init__(
        self,
        show_skeleton: bool = True,
        show_landmarks: bool = True,
        show_bbox: bool = True,
        show_finger_status: bool = True,
        show_landmark_ids: bool = False,
    ):
        self.show_skeleton = show_skeleton
        self.show_landmarks = show_landmarks
        self.show_bbox = show_bbox
        self.show_finger_status = show_finger_status
        self.show_landmark_ids = show_landmark_ids

    def _get_hand_colors(self, handedness: str):
        if handedness.lower() == "left":
            return (
                VisualizerTheme.LEFT_BONE_COLOR,
                VisualizerTheme.LEFT_JOINT_COLOR,
                VisualizerTheme.LEFT_TIP_COLOR,
                VisualizerTheme.LEFT_BOX_COLOR,
                VisualizerTheme.LEFT_TEXT_BG,
            )
        else:
            return (
                VisualizerTheme.RIGHT_BONE_COLOR,
                VisualizerTheme.RIGHT_JOINT_COLOR,
                VisualizerTheme.RIGHT_TIP_COLOR,
                VisualizerTheme.RIGHT_BOX_COLOR,
                VisualizerTheme.RIGHT_TEXT_BG,
            )

    def draw_skeleton(self, frame: np.ndarray, hands: List[HandData]) -> np.ndarray:
        """
        Draws bones and joint landmarks for each detected hand onto the frame.
        """
        for hand in hands:
            bone_col, joint_col, tip_col, box_col, text_bg = self._get_hand_colors(hand.handedness)

            # 1. Draw Bone Connections
            if self.show_skeleton:
                for idx1, idx2 in HAND_CONNECTIONS:
                    pt1 = hand.get_landmark_px(idx1)
                    pt2 = hand.get_landmark_px(idx2)
                    cv2.line(frame, pt1, pt2, bone_col, 2, cv2.LINE_AA)

            # 2. Draw Landmark Joints
            if self.show_landmarks:
                for lm in hand.landmarks:
                    is_tip = lm.index in FINGERTIP_INDICES
                    radius = 5 if is_tip else 3
                    color = tip_col if is_tip else joint_col

                    # Outer circle
                    cv2.circle(frame, (lm.px, lm.py), radius, color, -1, cv2.LINE_AA)
                    # Border
                    cv2.circle(frame, (lm.px, lm.py), radius + 1, (20, 20, 20), 1, cv2.LINE_AA)

                    if self.show_landmark_ids:
                        cv2.putText(
                            frame,
                            str(lm.index),
                            (lm.px + 4, lm.py - 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

            # 3. Draw Bounding Box & Label
            if self.show_bbox:
                xmin, ymin, xmax, ymax = hand.bbox
                # Draw rounded-corner style bbox
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_col, 1, cv2.LINE_AA)

                # Label text
                label = f"{hand.handedness} ({int(hand.score * 100)}%)"
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Label background
                label_ymin = max(0, ymin - th - 8)
                cv2.rectangle(
                    frame,
                    (xmin, label_ymin),
                    (xmin + tw + 8, label_ymin + th + 6),
                    text_bg,
                    -1,
                )
                cv2.putText(
                    frame,
                    label,
                    (xmin + 4, label_ymin + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # 4. Finger Extension State Indicators
            if self.show_finger_status and hand.finger_states:
                xmin, ymin, xmax, ymax = hand.bbox
                states = hand.finger_states
                f_str = f"T:{1 if states.get('thumb') else 0} I:{1 if states.get('index') else 0} M:{1 if states.get('middle') else 0} R:{1 if states.get('ring') else 0} P:{1 if states.get('pinky') else 0}"
                cv2.putText(
                    frame,
                    f_str,
                    (xmin, min(frame.shape[0] - 5, ymax + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    box_col,
                    1,
                    cv2.LINE_AA,
                )

        return frame

    def draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        num_hands: int,
        sign_info: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Renders a modern top HUD banner showing FPS, Hands Detected, and the Current Sign / Value Output.
        """
        h, w, _ = frame.shape
        banner_h = 56

        # Semi-transparent top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), VisualizerTheme.HUD_BG, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Bottom accent line
        cv2.line(frame, (0, banner_h), (w, banner_h), (60, 60, 70), 1)

        # FPS indicator
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame,
            fps_text,
            (16, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            VisualizerTheme.ACCENT_GREEN if fps >= 25 else VisualizerTheme.ACCENT_RED,
            2,
            cv2.LINE_AA,
        )

        # Hands count indicator
        hands_col = VisualizerTheme.ACCENT_GREEN if num_hands == 2 else (VisualizerTheme.LEFT_BONE_COLOR if num_hands == 1 else (150, 150, 150))
        hands_text = f"Hands: {num_hands} / 2"
        cv2.putText(
            frame,
            hands_text,
            (140, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            hands_col,
            2,
            cv2.LINE_AA,
        )

        # Sign / Value Output Box
        if sign_info:
            sign_name = sign_info.get("sign_name", "None")
            sign_val = sign_info.get("value", "")
            display_text = f"Sign: {sign_name}"
            if sign_val is not None and str(sign_val) != "":
                display_text += f" -> [ {sign_val} ]"
            
            # Draw badge in center/right
            (tw, th), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            bx = max(310, w // 2 - tw // 2)
            cv2.rectangle(
                frame,
                (bx - 12, 10),
                (bx + tw + 12, 46),
                (45, 45, 60),
                -1,
            )
            cv2.rectangle(
                frame,
                (bx - 12, 10),
                (bx + tw + 12, 46),
                VisualizerTheme.ACCENT_PURPLE,
                1,
            )
            cv2.putText(
                frame,
                display_text,
                (bx, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            prompt_text = "Show 2 hands to camera..."
            cv2.putText(
                frame,
                prompt_text,
                (320, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (170, 170, 180),
                1,
                cv2.LINE_AA,
            )

        # Bottom help controls banner
        help_text = "[Q: Quit]  [S: Toggle Skeleton]  [B: Toggle BBox]  [F: Toggle Finger Status]  [I: Toggle LM IDs]"
        cv2.putText(
            frame,
            help_text,
            (16, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (180, 180, 190),
            1,
            cv2.LINE_AA,
        )

        return frame
