"""
Automated unit and precision tests for MARG-One Palm Cursor Controller and 1 Euro Filter.
"""

import sys
import os
import math
import numpy as np

# Ensure project root is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marg_one.vision.tracker import HandData, HandLandmark, LandmarkIndex, FINGERTIP_INDICES
from marg_one.control import (
    HandCursorController,
    CursorState,
    MockMouseDriver,
    OneEuroFilter,
    Point2DOneEuroFilter,
)


def create_mock_hand(palm_center=(320, 240), is_closed=False) -> HandData:
    """Creates a mock HandData instance for testing palm tracking and closure."""
    cx, cy = palm_center
    landmarks = []

    # Palm base points
    palm_offsets = {
        LandmarkIndex.WRIST: (0, 40),
        LandmarkIndex.INDEX_FINGER_MCP: (-25, -20),
        LandmarkIndex.MIDDLE_FINGER_MCP: (0, -25),
        LandmarkIndex.RING_FINGER_MCP: (20, -22),
        LandmarkIndex.PINKY_MCP: (35, -15),
    }

    # Fingertip offsets (far when open, tucked close to palm center when closed)
    if is_closed:
        tip_offsets = {
            LandmarkIndex.THUMB_TIP: (-10, -5),
            LandmarkIndex.INDEX_FINGER_TIP: (-8, -10),
            LandmarkIndex.MIDDLE_FINGER_TIP: (0, -12),
            LandmarkIndex.RING_FINGER_TIP: (8, -10),
            LandmarkIndex.PINKY_TIP: (12, -8),
        }
    else:
        tip_offsets = {
            LandmarkIndex.THUMB_TIP: (-45, -15),
            LandmarkIndex.INDEX_FINGER_TIP: (-30, -85),
            LandmarkIndex.MIDDLE_FINGER_TIP: (0, -95),
            LandmarkIndex.RING_FINGER_TIP: (25, -90),
            LandmarkIndex.PINKY_TIP: (45, -75),
        }

    for i in range(21):
        if i in palm_offsets:
            dx, dy = palm_offsets[i]
        elif i in tip_offsets:
            dx, dy = tip_offsets[i]
        else:
            dx, dy = (0, 0)

        px = cx + dx
        py = cy + dy

        landmarks.append(HandLandmark(
            index=i,
            x=px / 640.0,
            y=py / 480.0,
            z=0.0,
            px=px,
            py=py,
            name=f"LM_{i}",
        ))

    finger_states = {
        "thumb": not is_closed,
        "index": not is_closed,
        "middle": not is_closed,
        "ring": not is_closed,
        "pinky": not is_closed,
    }

    return HandData(
        handedness="Right",
        score=0.98,
        landmarks=landmarks,
        world_landmarks=[],
        bbox=(cx - 60, cy - 100, cx + 60, cy + 60),
        center=(cx, cy),
        finger_states=finger_states,
    )


def test_one_euro_filter():
    print("[*] Testing 1 Euro Filter precision and noise attenuation...")
    np.random.seed(42)
    filt = Point2DOneEuroFilter(min_cutoff=0.5, beta=0.005)

    # Stationary signal with high-frequency noise
    t = 0.0
    true_x = 500.0
    errors_raw = []
    errors_filtered = []

    for _ in range(80):
        t += 0.033
        noise = float(np.random.normal(0, 4.0))
        noisy_x = true_x + noise
        fx, _ = filt.filter(noisy_x, 500.0, timestamp=t)
        errors_raw.append(abs(noisy_x - true_x))
        errors_filtered.append(abs(fx - true_x))

    mean_raw_err = float(np.mean(errors_raw))
    mean_filt_err = float(np.mean(errors_filtered))
    assert mean_filt_err < mean_raw_err * 0.6, "1 Euro filter failed to suppress noise"
    print(f"    - Noise attenuation verified (Raw: {mean_raw_err:.2f}px -> Filtered: {mean_filt_err:.2f}px)")


def test_palm_cursor_controller():
    print("[*] Testing Palm-Center HandCursorController...")
    mock_driver = MockMouseDriver(screen_w=1920, screen_h=1080)
    controller = HandCursorController(
        driver=mock_driver,
        speed_gain=1.35,
        min_cutoff=0.5,
        beta=0.005,
        click_close_threshold=0.60,
        release_threshold=0.45,
        click_lock_duration=0.10,
        enable_active_control=True,
    )

    frame_shape = (480, 640, 3)

    # 1. Idle State (No Hands)
    t1 = controller.update([], frame_shape, timestamp=1.0)
    assert t1["state"] == CursorState.IDLE
    print("    - Idle state verified.")

    # 2. Open Palm (Moving)
    hand_open = create_mock_hand(palm_center=(320, 240), is_closed=False)
    t2 = controller.update([hand_open], frame_shape, timestamp=1.033)
    assert t2["state"] == CursorState.MOVING
    assert t2["is_closed"] is False
    assert ("MOVE", t2["screen_pos"]) in mock_driver.history
    print(f"    - Open Palm moving verified. Target: {t2['screen_pos']}")

    # 3. Closed Palm (Fist -> Click Down + Lock)
    hand_closed = create_mock_hand(palm_center=(320, 240), is_closed=True)
    t3 = controller.update([hand_closed], frame_shape, timestamp=1.066)
    assert t3["state"] == CursorState.CLICK_DOWN
    assert t3["is_closed"] is True
    assert t3["is_locked"] is True
    assert mock_driver.is_left_down is True
    print("    - Closed Palm fist click down verified.")

    # 4. Sustained Fist + Movement (Drag Mode after lock duration)
    hand_drag = create_mock_hand(palm_center=(380, 260), is_closed=True)
    t4 = controller.update([hand_drag], frame_shape, timestamp=1.25)
    assert t4["state"] == CursorState.DRAGGING
    assert mock_driver.is_left_down is True
    print("    - Continuous dragging state verified.")

    # 5. Open Palm Release (Click Up)
    hand_open_again = create_mock_hand(palm_center=(380, 260), is_closed=False)
    t5 = controller.update([hand_open_again], frame_shape, timestamp=1.283)
    assert t5["state"] == CursorState.RELEASED
    assert mock_driver.is_left_down is False
    print("    - Open Palm release verified.")

    print("[SUCCESS] All Palm Cursor Controller tests passed!")


if __name__ == "__main__":
    test_one_euro_filter()
    test_palm_cursor_controller()
