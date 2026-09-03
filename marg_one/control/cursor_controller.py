"""
MARG-One Control Subsystem: Precision Palm-Center Hand Cursor & Pinch-to-Click Controller.
Uses 1 Euro Filter precision smoothing, anatomical palm center tracking, and index-thumb pinch clicking.
"""

from typing import List, Tuple, Optional, Dict, Any
import math
import sys
import time
import os
import numpy as np

from marg_one.vision.tracker import HandData, LandmarkIndex, FINGERTIP_INDICES
from marg_one.control.one_euro_filter import Point2DOneEuroFilter


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
    Precision Palm-Center Mouse Navigation & Pinch-to-Click Controller.

    Key Features:
    - Palm-Center Tracking: Tracks the rigid anatomical palm centroid (Wrist + 4 MCPs) for ultra-stable cursor positioning.
    - 1 Euro Filter: Adaptive speed-based low-pass filter eliminating micro-jitter with instant responsiveness.
    - Index-Thumb Pinch Click: Left click triggers when index fingertip and thumb fingertip touch.
    - Click-Lock Stabilization: Temporarily locks cursor coordinates for 120ms during pinch contact so clicks don't slip.
    - Seamless Drag & Drop: Holding pinch while moving maintains continuous mouse dragging.
    - Full-Frame Mapping: Entire camera frame maps directly to your screen with adjustable sensitivity gain.
    """

    PALM_LANDMARK_INDICES = [
        LandmarkIndex.WRIST,
        LandmarkIndex.INDEX_FINGER_MCP,
        LandmarkIndex.MIDDLE_FINGER_MCP,
        LandmarkIndex.RING_FINGER_MCP,
        LandmarkIndex.PINKY_MCP,
    ]

    def __init__(
        self,
        driver: Optional[BaseMouseDriver] = None,
        speed_gain: float = 1.35,
        min_cutoff: float = 0.5,
        beta: float = 0.005,
        pinch_threshold: float = 38.0,
        release_multiplier: float = 1.35,
        click_lock_duration: float = 0.12,
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
        self.speed_gain = speed_gain
        self.pinch_threshold = pinch_threshold
        self.release_threshold = pinch_threshold * release_multiplier
        self.click_lock_duration = click_lock_duration
        self.enable_active_control = enable_active_control

        # 1 Euro Filter for ultra-smooth 2D tracking
        self.filter = Point2DOneEuroFilter(min_cutoff=min_cutoff, beta=beta)

        # State management
        self.prev_screen_x = self.screen_w / 2.0
        self.prev_screen_y = self.screen_h / 2.0
        self.is_pinched = False
        self.click_timestamp = 0.0
        self.current_state = CursorState.IDLE
        self.active_hand_label = None

    def compute_palm_center(self, hand: HandData) -> Tuple[int, int, float, float]:
        """
        Computes the rigid anatomical palm center.
        Returns (pixel_x, pixel_y, norm_x, norm_y).
        """
        px_sum = 0.0
        py_sum = 0.0
        nx_sum = 0.0
        ny_sum = 0.0

        for idx in self.PALM_LANDMARK_INDICES:
            lm = hand.landmarks[idx]
            px_sum += lm.px
            py_sum += lm.py
            nx_sum += lm.x
            ny_sum += lm.y

        count = float(len(self.PALM_LANDMARK_INDICES))
        return (
            int(round(px_sum / count)),
            int(round(py_sum / count)),
            nx_sum / count,
            ny_sum / count,
        )

    def compute_pinch(self, hand: HandData) -> Tuple[float, bool]:
        """
        Computes the Euclidean distance between Index Fingertip and Thumb Fingertip
        and evaluates the pinch state with hysteresis.
        """
        index_tip = hand.landmarks[LandmarkIndex.INDEX_FINGER_TIP]
        thumb_tip = hand.landmarks[LandmarkIndex.THUMB_TIP]

        pinch_dist = math.hypot(index_tip.px - thumb_tip.px, index_tip.py - thumb_tip.py)

        # Scale-aware dynamic threshold
        wrist = hand.landmarks[LandmarkIndex.WRIST]
        middle_mcp = hand.landmarks[LandmarkIndex.MIDDLE_FINGER_MCP]
        palm_scale = max(10.0, math.hypot(wrist.px - middle_mcp.px, wrist.py - middle_mcp.py))

        effective_close_thresh = max(25.0, min(self.pinch_threshold, palm_scale * 0.40))
        effective_release_thresh = effective_close_thresh * 1.35

        if not self.is_pinched:
            is_pinched = pinch_dist < effective_close_thresh
        else:
            is_pinched = not (pinch_dist > effective_release_thresh)

        return (pinch_dist, is_pinched)

    def _select_controlling_hand(self, hands: List[HandData]) -> Optional[HandData]:
        """Selects the active hand to control the cursor."""
        if not hands:
            return None
        return hands[0]

    def update(
        self,
        hands: List[HandData],
        frame_shape: Tuple[int, int, int],
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Updates cursor position and left click state based on palm center tracking and pinch action.
        """
        if timestamp is None:
            timestamp = time.time()

        frame_h, frame_w, _ = frame_shape
        active_hand = self._select_controlling_hand(hands)

        telemetry = {
            "state": CursorState.IDLE,
            "active_hand": None,
            "screen_pos": (int(self.prev_screen_x), int(self.prev_screen_y)),
            "palm_center_px": None,
            "raw_index_pos": None,
            "raw_thumb_pos": None,
            "pinch_distance": 0.0,
            "is_pinched": self.is_pinched,
            "is_locked": False,
        }

        if active_hand is None:
            # Release click if hand drops out
            if self.is_pinched:
                self.is_pinched = False
                if self.enable_active_control:
                    self.driver.left_up()
                self.current_state = CursorState.RELEASED
            else:
                self.current_state = CursorState.IDLE
            self.filter.reset()
            telemetry["state"] = self.current_state
            return telemetry

        self.active_hand_label = active_hand.handedness
        telemetry["active_hand"] = active_hand.handedness

        # 1. Compute Anatomical Palm Center
        palm_px_x, palm_px_y, norm_x, norm_y = self.compute_palm_center(active_hand)
        telemetry["palm_center_px"] = (palm_px_x, palm_px_y)

        # 2. Get Index and Thumb Tips for Pinch Detection
        index_tip_px = active_hand.get_landmark_px(LandmarkIndex.INDEX_FINGER_TIP)
        thumb_tip_px = active_hand.get_landmark_px(LandmarkIndex.THUMB_TIP)
        telemetry["raw_index_pos"] = index_tip_px
        telemetry["raw_thumb_pos"] = thumb_tip_px

        pinch_dist, currently_pinched = self.compute_pinch(active_hand)
        telemetry["pinch_distance"] = round(pinch_dist, 1)

        # 3. Coordinate Scaling with Sensitivity Gain
        cx = 0.5
        cy = 0.5
        scaled_x = cx + (norm_x - cx) * self.speed_gain
        scaled_y = cy + (norm_y - cy) * self.speed_gain

        clamped_x = max(0.0, min(1.0, scaled_x))
        clamped_y = max(0.0, min(1.0, scaled_y))

        raw_screen_x = clamped_x * self.screen_w
        raw_screen_y = clamped_y * self.screen_h

        # 4. 1 Euro Filter Precision Smoothing
        filtered_screen_x, filtered_screen_y = self.filter.filter(
            raw_screen_x, raw_screen_y, timestamp=timestamp
        )

        # 5. Click-Lock Stabilization on Pinch
        is_locked = False
        if currently_pinched and not self.is_pinched:
            # Transition: UNPINCHED -> PINCHED (Left Click Down)
            self.is_pinched = True
            self.click_timestamp = timestamp
            if self.enable_active_control:
                self.driver.left_down()
            self.current_state = CursorState.CLICK_DOWN
            is_locked = True
        elif currently_pinched and self.is_pinched:
            # Holding pinch
            if timestamp - self.click_timestamp < self.click_lock_duration:
                # Freeze position during initial pinch contact to prevent slip
                is_locked = True
                self.current_state = CursorState.CLICK_DOWN
            else:
                self.current_state = CursorState.DRAGGING
        elif not currently_pinched and self.is_pinched:
            # Transition: PINCHED -> UNPINCHED (Left Click Up)
            self.is_pinched = False
            if self.enable_active_control:
                self.driver.left_up()
            self.current_state = CursorState.RELEASED
        else:
            self.current_state = CursorState.MOVING

        telemetry["is_locked"] = is_locked

        if not is_locked:
            self.prev_screen_x = filtered_screen_x
            self.prev_screen_y = filtered_screen_y

        final_x = int(round(self.prev_screen_x))
        final_y = int(round(self.prev_screen_y))
        telemetry["screen_pos"] = (final_x, final_y)

        # 6. Apply to Hardware Driver
        if self.enable_active_control and self.current_state != CursorState.IDLE:
            self.driver.set_position(final_x, final_y)

        telemetry["is_pinched"] = self.is_pinched
        telemetry["state"] = self.current_state
        return telemetry
