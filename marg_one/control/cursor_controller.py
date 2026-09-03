"""
MARG-One Control Subsystem: Precision Palm-Center Hand Cursor & Closed-Palm Click Controller.
Uses 1 Euro Filter precision smoothing, anatomical palm center tracking, and closed-palm (fist) clicking.
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
    Precision Palm-Center Mouse Navigation & Closed-Palm Click Controller.

    Key Features:
    - Palm-Center Tracking: Tracks the rigid anatomical palm centroid (Wrist + MCPs) for maximum physical stability.
    - 1 Euro Filter: Precision adaptive filter eliminating micro-jitter with instantaneous motion response.
    - Closed-Palm Click (Fist): Opening the hand moves the cursor; closing the palm triggers Left Click Down.
    - Click-Lock Stabilization: Locks cursor position for 120ms during fist closure so clicks land precisely without drifting.
    - Seamless Drag & Drop: Holding closed palm and moving executes continuous mouse dragging.
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
        click_close_threshold: float = 0.62,
        release_threshold: float = 0.45,
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
        self.click_close_threshold = click_close_threshold
        self.release_threshold = release_threshold
        self.click_lock_duration = click_lock_duration
        self.enable_active_control = enable_active_control

        # 1 Euro Filter for ultra-smooth 2D tracking
        self.filter = Point2DOneEuroFilter(min_cutoff=min_cutoff, beta=beta)

        # State management
        self.prev_screen_x = self.screen_w / 2.0
        self.prev_screen_y = self.screen_h / 2.0
        self.is_closed = False
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

    def compute_hand_closure(self, hand: HandData, palm_center_px: Tuple[int, int]) -> Tuple[float, bool]:
        """
        Computes a continuous closure ratio [0.0 = fully open, 1.0 = fully closed fist].
        Uses normalized distance from all 5 fingertips to palm center relative to palm scale.
        """
        wrist = hand.landmarks[LandmarkIndex.WRIST]
        middle_mcp = hand.landmarks[LandmarkIndex.MIDDLE_FINGER_MCP]
        palm_scale = max(10.0, math.hypot(wrist.px - middle_mcp.px, wrist.py - middle_mcp.py))

        tip_distances = []
        for tip_idx in FINGERTIP_INDICES:
            tip = hand.landmarks[tip_idx]
            dist = math.hypot(tip.px - palm_center_px[0], tip.py - palm_center_px[1])
            tip_distances.append(dist)

        avg_tip_dist = sum(tip_distances) / len(tip_distances)
        ratio = avg_tip_dist / (palm_scale * 1.6)
        closure_score = max(0.0, min(1.0, 1.0 - ratio))

        # Check binary finger states
        closed_fingers_count = sum(1 for is_ext in hand.finger_states.values() if not is_ext)

        # Hysteresis decision
        if not self.is_closed:
            is_closed = (closure_score >= self.click_close_threshold) or (closed_fingers_count >= 4)
        else:
            is_closed = not ((closure_score < self.release_threshold) or (closed_fingers_count <= 2))

        return (closure_score, is_closed)

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
        Updates cursor position and left click state based on palm tracking.
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
            "closure_score": 0.0,
            "is_closed": self.is_closed,
            "is_locked": False,
        }

        if active_hand is None:
            # Release click if hand drops out
            if self.is_closed:
                self.is_closed = False
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

        # 2. Compute Hand Closure (Fist)
        closure_score, currently_closed = self.compute_hand_closure(active_hand, (palm_px_x, palm_px_y))
        telemetry["closure_score"] = round(closure_score, 3)

        # 3. Coordinate Scaling with Sensitivity Gain
        # Center-anchored scaling for comfortable full-screen reach
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

        # 5. Click-Lock Stabilization
        is_locked = False
        if currently_closed and not self.is_closed:
            # Transition: OPEN -> CLOSED (Left Click Down)
            self.is_closed = True
            self.click_timestamp = timestamp
            if self.enable_active_control:
                self.driver.left_down()
            self.current_state = CursorState.CLICK_DOWN
            is_locked = True
        elif currently_closed and self.is_closed:
            # Holding fist closed
            if timestamp - self.click_timestamp < self.click_lock_duration:
                # Freeze position during initial clench to prevent slip
                is_locked = True
                self.current_state = CursorState.CLICK_DOWN
            else:
                self.current_state = CursorState.DRAGGING
        elif not currently_closed and self.is_closed:
            # Transition: CLOSED -> OPEN (Left Click Up)
            self.is_closed = False
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

        telemetry["is_closed"] = self.is_closed
        telemetry["state"] = self.current_state
        return telemetry
