"""
Automated unit and integration test for Hand Tracker, Visualizer, and Sign Processor.
"""

import numpy as np
import cv2
from hand_tracker import DualHandTracker, HandData, HandLandmark, LandmarkIndex
from visualizer import HandVisualizer
from sign_processor import SignProcessor


def create_synthetic_hand_data(handedness="Right", offset_x=200, offset_y=200) -> HandData:
    """Generates synthetic 21 landmarks for testing skeleton logic."""
    landmarks = []
    # 21 approximate positions for an open hand
    coords = [
        (0, 50),   # 0: Wrist
        (-20, 30), (-35, 15), (-45, 0), (-55, -15),  # 1-4: Thumb
        (-25, -20), (-30, -50), (-32, -70), (-35, -90), # 5-8: Index
        (-5, -25), (-5, -55), (-5, -80), (-5, -100),    # 9-12: Middle
        (15, -22), (18, -50), (20, -75), (22, -95),     # 13-16: Ring
        (35, -15), (42, -40), (45, -60), (48, -80),     # 17-20: Pinky
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


def test_hand_tracking_pipeline():
    print("[*] Running DualHandTracker test...")
    tracker = DualHandTracker(num_hands=2)
    
    # Test with blank frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    hands = tracker.process_frame(dummy_frame)
    assert isinstance(hands, list), "Expected list of HandData"
    print(f"    - Tracker executed successfully. Detected on blank frame: {len(hands)} hands (Expected 0)")

    # Test Visualizer with synthetic dual hands
    print("[*] Running HandVisualizer test...")
    visualizer = HandVisualizer()
    hand_left = create_synthetic_hand_data("Left", offset_x=200, offset_y=240)
    hand_right = create_synthetic_hand_data("Right", offset_x=440, offset_y=240)

    test_frame = dummy_frame.copy()
    rendered_frame = visualizer.draw_skeleton(test_frame, [hand_left, hand_right])
    rendered_frame = visualizer.draw_hud(rendered_frame, fps=30.0, num_hands=2, sign_info={"sign_name": "TEST_SIGN", "value": "100"})
    assert rendered_frame.shape == (480, 640, 3), "Frame dimensions altered"
    print("    - Visualizer rendered skeleton and HUD successfully.")

    # Test Sign Processor
    print("[*] Running SignProcessor test...")
    processor = SignProcessor()
    sign_result = processor.process_hands([hand_left, hand_right])
    assert sign_result is not None, "SignProcessor returned None for dual hands"
    assert sign_result["hands_count"] == 2, "Expected 2 hands processed"
    print(f"    - SignProcessor detected: {sign_result['sign_name']} (Value: {sign_result['value']})")

    tracker.close()
    print("[SUCCESS] All tests passed!")


if __name__ == "__main__":
    test_hand_tracking_pipeline()
