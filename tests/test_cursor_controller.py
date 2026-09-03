"""
Automated unit tests for MARG-One Hand Cursor Controller.
"""

import sys
import os
import math
import numpy as np

# Ensure project root is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marg_one.vision.tracker import HandData, HandLandmark, LandmarkIndex
from marg_one.control import (
    HandCursorController,
    CursorState,
    MockMouseDriver,
)


def create_mock_hand(index_pos=(320, 240), thumb_pos=(370, 240), index_extended=True) -> HandData:
    """Creates a mock HandData instance for testing cursor mechanics."""
    landmarks = []
    for i in range(21):
        if i == LandmarkIndex.INDEX_FINGER_TIP:
            px, py = index_pos
        elif i == LandmarkIndex.THUMB_TIP:
            px, py = thumb_pos
        elif i == LandmarkIndex.WRIST:
            px, py = (index_pos[0], index_pos[1] + 150)
        else:
            px, py = (index_pos[0] + 20, index_pos[1] + 50)

        landmarks.append(HandLandmark(
            index=i,
            x=px / 640.0,
            y=py / 480.0,
            z=0.0,
            px=px,
            py=py,
            name=f"LM_{i}",
        ))

    return HandData(
        handedness="Right",
        score=0.98,
        landmarks=landmarks,
        world_landmarks=[],
        bbox=(100, 100, 500, 400),
        center=(index_pos[0], index_pos[1] + 70),
        finger_states={"thumb": True, "index": index_extended, "middle": False, "ring": False, "pinky": False},
    )


def test_cursor_controller():
    print("[*] Testing HandCursorController with MockMouseDriver...")
    mock_driver = MockMouseDriver(screen_w=1920, screen_h=1080)
    controller = HandCursorController(
        driver=mock_driver,
        margin_x=0.1,
        margin_y=0.1,
        smoothing_factor=0.5,
        pinch_threshold=35.0,
        enable_active_control=True,
    )

    frame_shape = (480, 640, 3)

    # 1. Test Idle State (No Hands)
    t1 = controller.update([], frame_shape)
    assert t1["state"] == CursorState.IDLE, f"Expected IDLE, got {t1['state']}"
    print("    - Idle state verified.")

    # 2. Test Movement Mode (Index finger extended, thumb far away: dist=60px)
    hand_moving = create_mock_hand(index_pos=(320, 240), thumb_pos=(380, 240), index_extended=True)
    t2 = controller.update([hand_moving], frame_shape)
    assert t2["state"] == CursorState.MOVING, f"Expected MOVING, got {t2['state']}"
    assert t2["screen_pos"][0] > 0 and t2["screen_pos"][1] > 0, "Invalid screen coordinates"
    assert ("MOVE", t2["screen_pos"]) in mock_driver.history, "Move event not recorded"
    print(f"    - Movement state verified. Screen target: {t2['screen_pos']}")

    # 3. Test Pinch Trigger (Left Click Down: dist=20px < 35px)
    hand_pinch = create_mock_hand(index_pos=(320, 240), thumb_pos=(335, 245), index_extended=True)
    t3 = controller.update([hand_pinch], frame_shape)
    assert t3["state"] == CursorState.CLICK_DOWN, f"Expected CLICK_DOWN, got {t3['state']}"
    assert mock_driver.is_left_down is True, "Expected left mouse button to be DOWN"
    print("    - Pinch-to-click transition verified.")

    # 4. Test Continuous Pinch Hold (Dragging)
    hand_drag = create_mock_hand(index_pos=(350, 260), thumb_pos=(365, 265), index_extended=True)
    t4 = controller.update([hand_drag], frame_shape)
    assert t4["state"] == CursorState.DRAGGING, f"Expected DRAGGING, got {t4['state']}"
    assert mock_driver.is_left_down is True, "Expected left mouse button to remain DOWN"
    print("    - Dragging state verified.")

    # 5. Test Pinch Release (Left Click Up: dist=70px > 35px)
    hand_release = create_mock_hand(index_pos=(350, 260), thumb_pos=(420, 260), index_extended=True)
    t5 = controller.update([hand_release], frame_shape)
    assert t5["state"] == CursorState.RELEASED, f"Expected RELEASED, got {t5['state']}"
    assert mock_driver.is_left_down is False, "Expected left mouse button to be RELEASED"
    print("    - Click release transition verified.")

    print("[SUCCESS] All HandCursorController tests passed!")


if __name__ == "__main__":
    test_cursor_controller()
