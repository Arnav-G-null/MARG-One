"""
MARG-One Control Subsystem: Hand Cursor Controller.
Enables real-time mouse navigation using index finger kinematics and pinch-to-click action.
"""

from typing import List, Tuple, Optional, Dict, Any
import math
import sys
import os
import numpy as np

from marg_one.vision.tracker import HandData, LandmarkIndex


class BaseMouseDriver:
    """Abstract interface for operating system mouse hardware inputs."""

    def set_position(self, x: int, y: int) -> None:
        raise NotImplementedError

    def left_down(self) -> None:
        raise NotImplementedError

    def left_up(self) -> None:
        raise NotImplementedError

    def get_screen_size(self) -> Tuple[int, int]:
        raise NotImplementedError


class WindowsMouseDriver(BaseMouseDriver):
    """High-performance native Windows mouse driver utilizing ctypes user32 APIs."""

    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_RIGHTDOWN = 0x0008
    MOUSEEVENTF_RIGHTUP = 0x0010

    def __init__(self):
        import ctypes
        self.user32 = ctypes.windll.user32
        # Enable Per-Monitor DPI Awareness if possible
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                self.user32.SetProcessDPIAware()
            except Exception:
                pass

        self.screen_w = self.user32.GetSystemMetrics(0)
        self.screen_h = self.user32.GetSystemMetrics(1)

    def set_position(self, x: int, y: int) -> None:
        clamped_x = max(0, min(self.screen_w - 1, int(x)))
        clamped_y = max(0, min(self.screen_h - 1, int(y)))
        self.user32.SetCursorPos(clamped_x, clamped_y)

    def left_down(self) -> None:
        self.user32.mouse_event(self.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)

    def left_up(self) -> None:
        self.user32.mouse_event(self.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

    def get_screen_size(self) -> Tuple[int, int]:
        return (self.screen_w, self.screen_h)


class MockMouseDriver(BaseMouseDriver):
    """Mock mouse driver for testing, telemetry logging, and headless environments."""

    def __init__(self, screen_w: int = 1920, screen_h: int = 1080):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.cursor_pos = (screen_w // 2, screen_h // 2)
        self.is_left_down = False
        self.history = []

    def set_position(self, x: int, y: int) -> None:
        self.cursor_pos = (int(x), int(y))
        self.history.append(("MOVE", self.cursor_pos))

    def left_down(self) -> None:
        self.is_left_down = True
        self.history.append(("LEFT_DOWN", self.cursor_pos))

    def left_up(self) -> None:
        self.is_left_down = False
        self.history.append(("LEFT_UP", self.cursor_pos))

    def get_screen_size(self) -> Tuple[int, int]:
        return (self.screen_w, self.screen_h)


class CursorState:
    IDLE = "IDLE"
    MOVING = "MOVING"
    CLICK_DOWN = "CLICK_DOWN"
    DRAGGING = "DRAGGING"
    RELEASED = "RELEASED"


class HandCursorController:
    """
    Translates hand tracking landmarks into operating system mouse navigation and clicks.
    
    Control Rules:
    - Moving: Index finger of either hand points / navigates.
    - Clicking / Dragging: Pinching index fingertip and thumb fingertip together triggers Left Down.
    - Release: Opening pinch triggers Left Up.
    """

    def __init__(
        self,
        driver: Optional[BaseMouseDriver] = None,
        margin_x: float = 0.0,
        margin_y: float = 0.0,
        smoothing_factor: float = 0.35,
        pinch_threshold: float = 38.0,
        enable_active_control: bool = True,
    ):
        if driver is None:
            if sys.platform.startswith("win"):
                self.driver = WindowsMouseDriver()
            else:
                self.driver = MockMouseDriver()
        else:
            self.driver = driver

        self.screen_w, self.screen_h = self.driver.get_screen_size()
        self.margin_x = margin_x
        self.margin_y = margin_y
        self.smoothing_factor = smoothing_factor
        self.pinch_threshold = pinch_threshold
        self.enable_active_control = enable_active_control

        # Internal state tracking
        self.prev_screen_x = self.screen_w / 2.0
        self.prev_screen_y = self.screen_h / 2.0
        self.is_pinched = False
        self.current_state = CursorState.IDLE
        self.active_hand_label = None

    def get_interaction_box(self, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        """
        Computes the active interaction bounding box within camera frame boundaries.
        Returns (x1, y1, x2, y2).
        """
        x1 = int(frame_w * self.margin_x)
        y1 = int(frame_h * self.margin_y)
        x2 = int(frame_w * (1.0 - self.margin_x))
        y2 = int(frame_h * (1.0 - self.margin_y))
        return (x1, y1, x2, y2)

    def _select_controlling_hand(self, hands: List[HandData]) -> Optional[HandData]:
        """
        Selects the hand to use for cursor tracking.
        Prioritizes any hand whose index finger is extended.
        """
        if not hands:
            return None

        for hand in hands:
            if hand.finger_states.get("index", False):
                return hand

        # If neither has index strictly flagged extended, check if pinched
        for hand in hands:
            idx_pt = hand.get_landmark_px(LandmarkIndex.INDEX_FINGER_TIP)
            th_pt = hand.get_landmark_px(LandmarkIndex.THUMB_TIP)
            if math.hypot(idx_pt[0] - th_pt[0], idx_pt[1] - th_pt[1]) <= self.pinch_threshold * 1.2:
                return hand

        return None

    def update(
        self,
        hands: List[HandData],
        frame_shape: Tuple[int, int, int],
    ) -> Dict[str, Any]:
        """
        Processes tracked hands in the current frame and drives cursor / click mechanics.
        
        Returns telemetry dictionary containing cursor coordinates, states, and debug metrics.
        """
        frame_h, frame_w, _ = frame_shape
        box_x1, box_y1, box_x2, box_y2 = self.get_interaction_box(frame_w, frame_h)
        box_w = max(1, box_x2 - box_x1)
        box_h = max(1, box_y2 - box_y1)

        active_hand = self._select_controlling_hand(hands)

        telemetry = {
            "state": CursorState.IDLE,
            "active_hand": None,
            "screen_pos": (int(self.prev_screen_x), int(self.prev_screen_y)),
            "raw_index_pos": None,
            "raw_thumb_pos": None,
            "pinch_distance": 0.0,
            "is_pinched": self.is_pinched,
            "interaction_box": (box_x1, box_y1, box_x2, box_y2),
        }

        if active_hand is None:
            # Release click if hand disappears while dragging
            if self.is_pinched:
                self.is_pinched = False
                if self.enable_active_control:
                    self.driver.left_up()
                self.current_state = CursorState.RELEASED
            else:
                self.current_state = CursorState.IDLE
            telemetry["state"] = self.current_state
            return telemetry

        self.active_hand_label = active_hand.handedness
        telemetry["active_hand"] = active_hand.handedness

        # Get landmark points
        index_tip_px = active_hand.get_landmark_px(LandmarkIndex.INDEX_FINGER_TIP)
        thumb_tip_px = active_hand.get_landmark_px(LandmarkIndex.THUMB_TIP)
        telemetry["raw_index_pos"] = index_tip_px
        telemetry["raw_thumb_pos"] = thumb_tip_px

        # 1. Pinch Detection (Index Tip to Thumb Tip Distance)
        pinch_dist = math.hypot(
            index_tip_px[0] - thumb_tip_px[0],
            index_tip_px[1] - thumb_tip_px[1]
        )
        telemetry["pinch_distance"] = round(pinch_dist, 1)

        currently_pinched = pinch_dist < self.pinch_threshold

        # 2. Cursor Coordinates Remapping (Index Finger Position)
        # Normalize index fingertip within active interaction box
        norm_x = (index_tip_px[0] - box_x1) / float(box_w)
        norm_y = (index_tip_px[1] - box_y1) / float(box_h)

        # Clamp to [0.0, 1.0]
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))

        # Target screen position
        target_screen_x = norm_x * self.screen_w
        target_screen_y = norm_y * self.screen_h

        # 3. Dynamic Motion Smoothing
        # Faster movement reduces smoothing for instant responsiveness
        dx = target_screen_x - self.prev_screen_x
        dy = target_screen_y - self.prev_screen_y
        velocity = math.hypot(dx, dy)
        
        # Adaptive alpha: between smoothing_factor and 0.85
        alpha = min(0.85, max(self.smoothing_factor, velocity / 100.0 * 0.5))

        smooth_x = self.prev_screen_x + alpha * dx
        smooth_y = self.prev_screen_y + alpha * dy

        self.prev_screen_x = smooth_x
        self.prev_screen_y = smooth_y

        screen_px = int(smooth_x)
        screen_py = int(smooth_y)
        telemetry["screen_pos"] = (screen_px, screen_py)

        # 4. Mouse Hardware Movement
        if self.enable_active_control:
            self.driver.set_position(screen_px, screen_py)

        # 5. Pinch Click State Transitions
        if currently_pinched and not self.is_pinched:
            # Transition: Idle/Moving -> Click Down
            self.is_pinched = True
            if self.enable_active_control:
                self.driver.left_down()
            self.current_state = CursorState.CLICK_DOWN
        elif currently_pinched and self.is_pinched:
            # Holding pinch -> Dragging
            self.current_state = CursorState.DRAGGING
        elif not currently_pinched and self.is_pinched:
            # Transition: Click Down/Dragging -> Release
            self.is_pinched = False
            if self.enable_active_control:
                self.driver.left_up()
            self.current_state = CursorState.RELEASED
        else:
            self.current_state = CursorState.MOVING

        telemetry["is_pinched"] = self.is_pinched
        telemetry["state"] = self.current_state
        return telemetry
