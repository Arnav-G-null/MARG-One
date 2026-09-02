"""
Automated unit and integration test for MARG-One Vision pipeline.
"""

import sys
import os
import numpy as np

# Ensure project root is in PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from marg_one.vision import (
    DualHandTracker,
    HandData,
    HandLandmark,
    HandVisualizer,
    SignProcessor,
)


def create_synthetic_hand_data(handedness="Right", offset_x=200, offset_y=200) -> HandData:
    """Generates synthetic 21 landmarks for testing skeleton logic."""
    landmarks = []
    coords = [
        (0, 50),
        (-20, 30), (-35, 15), (-45, 0), (-55, -15),
        (-25, -20), (-30, -50), (-32, -70), (-35, -90),
        (-5, -25), (-5, -55), (-5, -80), (-5, -100),
        (15, -22), (18, -50), (20, -75), (22, -95),
        (35, -15), (42, -40), (45, -60), (48, -80),
    ]
    xs, ys = [], []
    for idx, (dx, dy) in enumerate(coords):
        px = offset_x + dx
        py = offset_y + dy
        xs.append(px)
        ys.append(py)
        landmarks.append(HandLandmark(
            index=idx,
            x=px / 640.0,
            y=py / 480.0,
            z=0.0,
            px=px,
            py=py,
            name=f"LM_{idx}"
        ))

    return HandData(
        handedness=handedness,
        score=0.96,
        landmarks=landmarks,
        world_landmarks=[],
        bbox=(min(xs) - 10, min(ys) - 10, max(xs) + 10, max(ys) + 10),
        center=(offset_x, offset_y),
        finger_states={"thumb": True, "index": True, "middle": True, "ring": True, "pinky": True},
    )


def test_marg_one_vision_pipeline():
    print("[*] Running MARG-One DualHandTracker test...")
    tracker = DualHandTracker(num_hands=2)

    # Blank frame test
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    hands = tracker.process_frame(dummy_frame)
    assert isinstance(hands, list), "Expected list of HandData"
    print(f"    - Tracker executed successfully. Detected on blank frame: {len(hands)} hands")

    # HandVisualizer test
    print("[*] Running MARG-One HandVisualizer test...")
    visualizer = HandVisualizer()
    hand_left = create_synthetic_hand_data("Left", offset_x=200, offset_y=240)
    hand_right = create_synthetic_hand_data("Right", offset_x=440, offset_y=240)

    test_frame = dummy_frame.copy()
    rendered_frame = visualizer.draw_skeleton(test_frame, [hand_left, hand_right])
    rendered_frame = visualizer.draw_hud(rendered_frame, fps=30.0, num_hands=2, sign_info={"sign_name": "TEST_SIGN", "value": "100"})
    assert rendered_frame.shape == (480, 640, 3), "Frame dimensions altered"
    print("    - Visualizer rendered skeleton and HUD successfully.")

    # SignProcessor test
    print("[*] Running MARG-One SignProcessor test...")
    processor = SignProcessor()
    sign_result = processor.process_hands([hand_left, hand_right])
    assert sign_result is not None, "SignProcessor returned None for dual hands"
    assert sign_result["hands_count"] == 2, "Expected 2 hands processed"
    print(f"    - SignProcessor detected: {sign_result['sign_name']} (Value: {sign_result['value']})")

    tracker.close()
    print("[SUCCESS] All MARG-One Vision tests passed!")


if __name__ == "__main__":
    test_marg_one_vision_pipeline()
