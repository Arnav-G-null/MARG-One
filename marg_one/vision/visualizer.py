"""
MARG-One Vision Subsystem: 3D Skeleton, Bounding Box, and Telemetry Visualizer.
"""

from typing import List, Optional, Dict, Any, Tuple
import math
import cv2
import numpy as np

from marg_one.vision.tracker import (
    HandData,
    HAND_CONNECTIONS,
    FINGERTIP_INDICES,
    LandmarkIndex,
)


class VisualizerTheme:
    """Color themes and palettes formatted in BGR for OpenCV."""
    # Left Hand (Cyan Palette)
    LEFT_BONE_COLOR = (230, 216, 0)
    LEFT_JOINT_COLOR = (200, 180, 0)
    LEFT_TIP_COLOR = (255, 255, 255)
    LEFT_BOX_COLOR = (230, 216, 0)
    LEFT_TEXT_BG = (180, 140, 0)

    # Right Hand (Coral / Orange Palette)
    RIGHT_BONE_COLOR = (50, 130, 255)
    RIGHT_JOINT_COLOR = (30, 100, 230)
    RIGHT_TIP_COLOR = (255, 255, 255)
    RIGHT_BOX_COLOR = (50, 130, 255)
    RIGHT_TEXT_BG = (30, 90, 200)

    # Mouse & Palm Controller Palette
    PALM_RETICLE_OPEN = (255, 210, 0)      # High-visibility Cyan/Aqua
    PALM_RETICLE_CLOSED = (60, 255, 80)    # Neon Green on click
    CLOSURE_RING_BG = (80, 80, 90)
    CLOSURE_RING_ACTIVE = (60, 255, 80)

    # Telemetry HUD
    HUD_BG = (25, 25, 30)
    HUD_TEXT = (240, 240, 240)
    ACCENT_GREEN = (80, 220, 100)
    ACCENT_RED = (80, 80, 240)
    ACCENT_PURPLE = (220, 100, 200)


class HandVisualizer:
    """
    Renders 3D hand skeletons, topological landmarks, bounding boxes, palm center reticles, and HUD overlays.
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
        return (
            VisualizerTheme.RIGHT_BONE_COLOR,
            VisualizerTheme.RIGHT_JOINT_COLOR,
            VisualizerTheme.RIGHT_TIP_COLOR,
            VisualizerTheme.RIGHT_BOX_COLOR,
            VisualizerTheme.RIGHT_TEXT_BG,
        )

    def draw_skeleton(self, frame: np.ndarray, hands: List[HandData]) -> np.ndarray:
        """
        Renders topological connections, joint vertices, and bounding bounds on the input frame.
        """
        for hand in hands:
            bone_col, joint_col, tip_col, box_col, text_bg = self._get_hand_colors(hand.handedness)

            # 1. Bone connections
            if self.show_skeleton:
                for idx1, idx2 in HAND_CONNECTIONS:
                    pt1 = hand.get_landmark_px(idx1)
                    pt2 = hand.get_landmark_px(idx2)
                    cv2.line(frame, pt1, pt2, bone_col, 2, cv2.LINE_AA)

            # 2. Landmarks
            if self.show_landmarks:
                for lm in hand.landmarks:
                    is_tip = lm.index in FINGERTIP_INDICES
                    radius = 5 if is_tip else 3
                    color = tip_col if is_tip else joint_col

                    cv2.circle(frame, (lm.px, lm.py), radius, color, -1, cv2.LINE_AA)
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

            # 3. Bounding Box & Handedness Tag
            if self.show_bbox:
                xmin, ymin, xmax, ymax = hand.bbox
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_col, 1, cv2.LINE_AA)

                label = f"{hand.handedness} ({int(hand.score * 100)}%)"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

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

            # 4. Finger extension indicators
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

    def draw_cursor_overlay(self, frame: np.ndarray, telemetry: Dict[str, Any]) -> np.ndarray:
        """
        Renders palm center reticle, dynamic closure meter, and click/drag status.
        """
        palm_center = telemetry.get("palm_center_px")
        if palm_center is None:
            return frame

        cx, cy = palm_center
        is_closed = telemetry.get("is_closed", False)
        closure_score = telemetry.get("closure_score", 0.0)
        is_locked = telemetry.get("is_locked", False)
        state = telemetry.get("state", "IDLE")

        reticle_color = VisualizerTheme.PALM_RETICLE_CLOSED if is_closed else VisualizerTheme.PALM_RETICLE_OPEN
        radius = 24

        # 1. Outer Background Arc
        cv2.circle(frame, (cx, cy), radius, VisualizerTheme.CLOSURE_RING_BG, 3, cv2.LINE_AA)

        # 2. Dynamic Closure Gauge (Arc indicating how close to a fist the hand is)
        angle_end = int(closure_score * 360)
        if angle_end > 0:
            cv2.ellipse(
                frame,
                (cx, cy),
                (radius, radius),
                -90,
                0,
                angle_end,
                VisualizerTheme.CLOSURE_RING_ACTIVE if is_closed else (255, 230, 50),
                4,
                cv2.LINE_AA,
            )

        # 3. Center Target Dot & Crosshairs
        cv2.circle(frame, (cx, cy), 6, reticle_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 7, (20, 20, 20), 1, cv2.LINE_AA)

        # Crosshair lines
        line_len = 8
        cv2.line(frame, (cx - radius - line_len, cy), (cx - radius + 2, cy), reticle_color, 2, cv2.LINE_AA)
        cv2.line(frame, (cx + radius - 2, cy), (cx + radius + line_len, cy), reticle_color, 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - radius - line_len), (cx, cy - radius + 2), reticle_color, 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy + radius - 2), (cx, cy + radius + line_len), reticle_color, 2, cv2.LINE_AA)

        # 4. Status Tag near Reticle
        status_text = "CLICK / DRAG" if is_closed else "NAVIGATING"
        if is_locked:
            status_text = "CLICK [LOCKED]"

        (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        tag_bg = (30, 140, 40) if is_closed else (40, 40, 55)
        cv2.rectangle(
            frame,
            (cx - tw // 2 - 4, cy + radius + 6),
            (cx + tw // 2 + 4, cy + radius + th + 10),
            tag_bg,
            -1,
        )
        cv2.putText(
            frame,
            status_text,
            (cx - tw // 2, cy + radius + th + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
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
        cursor_telemetry: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Renders the top telemetry banner showing FPS, active hand count, sign output, and cursor state.
        """
        h, w, _ = frame.shape
        banner_h = 56

        # Header background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), VisualizerTheme.HUD_BG, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
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

        # Hands count
        hands_col = VisualizerTheme.ACCENT_GREEN if num_hands >= 1 else (150, 150, 150)
        cv2.putText(
            frame,
            f"Hands: {num_hands} / 2",
            (140, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            hands_col,
            2,
            cv2.LINE_AA,
        )

        # Mouse / Cursor Status Badge
        if cursor_telemetry:
            c_state = cursor_telemetry.get("state", "IDLE")
            pos = cursor_telemetry.get("screen_pos", (0, 0))
            is_closed = cursor_telemetry.get("is_closed", False)
            closure = int(cursor_telemetry.get("closure_score", 0.0) * 100)
            hand_lbl = cursor_telemetry.get("active_hand", "None")

            badge_col = VisualizerTheme.ACCENT_GREEN if is_closed else (VisualizerTheme.ACCENT_PURPLE if c_state == "MOVING" else (100, 100, 110))
            status_text = f"Cursor: ({pos[0]}, {pos[1]}) | [{c_state}] | Fist: {closure}% | Hand: {hand_lbl}"

            (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            bx = max(300, w // 2 - tw // 2)
            cv2.rectangle(frame, (bx - 10, 10), (bx + tw + 10, 46), (40, 40, 52), -1)
            cv2.rectangle(frame, (bx - 10, 10), (bx + tw + 10, 46), badge_col, 2 if is_closed else 1)
            cv2.putText(
                frame,
                status_text,
                (bx, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        elif sign_info:
            sign_name = sign_info.get("sign_name", "None")
            sign_val = sign_info.get("value", "")
            display_text = f"Sign: {sign_name}"
            if sign_val is not None and str(sign_val) != "":
                display_text += f" -> [ {sign_val} ]"

            (tw, th), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
            bx = max(310, w // 2 - tw // 2)
            cv2.rectangle(frame, (bx - 12, 10), (bx + tw + 12, 46), (45, 45, 60), -1)
            cv2.rectangle(frame, (bx - 12, 10), (bx + tw + 12, 46), VisualizerTheme.ACCENT_PURPLE, 1)
            cv2.putText(
                frame,
                display_text,
                (bx, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                frame,
                "Ready for Gestures / Palm Mouse Control...",
                (320, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (170, 170, 180),
                1,
                cv2.LINE_AA,
            )

        # Bottom help status bar
        controls_text = "[Q: Quit]  [M: Toggle Mouse Control]  [S: Skeleton]  [B: BBox]  [F: Fingers]  [Open Palm: Move | Fist: Click/Drag]"
        cv2.putText(
            frame,
            controls_text,
            (16, h - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (180, 180, 190),
            1,
            cv2.LINE_AA,
        )

        return frame
