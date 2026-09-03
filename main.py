"""
MARG-One Vision & Control: Dual-Hand Kinematics, 3D Skeleton Extraction, Sign Interpretation & Cursor Driving.
Main interactive camera stream runner.
"""

import argparse
import time
import sys
import os
import cv2

# Support both package and standalone execution
try:
    from marg_one.vision import DualHandTracker, HandVisualizer, SignProcessor
    from marg_one.control import HandCursorController, CursorState
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from marg_one.vision import DualHandTracker, HandVisualizer, SignProcessor
    from marg_one.control import HandCursorController, CursorState


def main():
    parser = argparse.ArgumentParser(description="MARG-One Dual-Hand Tracking, Skeleton & Cursor System")
    parser.add_argument("--cam", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Camera width resolution (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Camera height resolution (default: 720)")
    parser.add_argument("--no-flip", action="store_true", help="Disable mirror horizontal flip")
    parser.add_argument("--conf", type=float, default=0.5, help="Minimum detection confidence (default: 0.5)")
    parser.add_argument("--mouse", action="store_true", help="Enable active hand mouse cursor control mode")
    parser.add_argument("--smooth", type=float, default=0.35, help="Cursor smoothing factor (default: 0.35)")
    parser.add_argument("--pinch", type=float, default=38.0, help="Pinch threshold in pixels (default: 38.0)")
    args = parser.parse_args()

    print("=" * 68)
    print("      MARG-ONE : VISION & INTERACTION SUBSYSTEM              ")
    print("=" * 68)
    print(f"[*] Camera Index:     {args.cam}")
    print(f"[*] Resolution:       {args.width}x{args.height}")
    print(f"[*] Confidence:       {args.conf}")
    print(f"[*] Mode:             {'HAND CURSOR CONTROLLER' if args.mouse else 'GESTURE & SIGN RECOGNITION'}")
    print(f"[*] Interactive Controls:")
    print(f"    - 'q' or ESC : Exit application")
    print(f"    - 'm'        : Toggle active mouse control mode")
    print(f"    - 's'        : Toggle hand skeletons")
    print(f"    - 'b'        : Toggle bounding boxes")
    print(f"    - 'f'        : Toggle finger states HUD")
    print(f"    - 'z'        : Toggle active screen zone overlay")
    print("=" * 68)

    # Initialize Camera
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[ERROR] Could not access webcam device at index {args.cam}.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Initialize Subsystems
    print("[*] Initializing MediaPipe Dual-Hand Tracker...")
    tracker = DualHandTracker(
        num_hands=2,
        min_detection_confidence=args.conf,
        min_presence_confidence=args.conf,
        min_tracking_confidence=args.conf,
    )
    visualizer = HandVisualizer()
    sign_processor = SignProcessor()
    cursor_controller = HandCursorController(
        smoothing_factor=args.smooth,
        pinch_threshold=args.pinch,
        enable_active_control=args.mouse,
    )

    mouse_mode_active = args.mouse
    print("[*] Vision loop initialized. Press 'q' to terminate.")

    fps = 0.0
    prev_time = time.time()
    last_reported_sign = None

    try:
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                print("[WARN] Camera frame capture failure.")
                break

            # Mirror frame for natural interaction
            if not args.no_flip:
                frame = cv2.flip(frame, 1)

            # Compute Frame Rate
            curr_time = time.time()
            time_diff = curr_time - prev_time
            prev_time = curr_time
            if time_diff > 0:
                current_fps = 1.0 / time_diff
                fps = 0.9 * fps + 0.1 * current_fps if fps > 0 else current_fps

            # 1. Infer Hands
            hands = tracker.process_frame(frame)

            sign_info = None
            cursor_telemetry = None

            if mouse_mode_active:
                # 2A. Update Hand Mouse Controller
                cursor_telemetry = cursor_controller.update(hands, frame.shape)
            else:
                # 2B. Evaluate Signs & Gestures
                sign_info = sign_processor.process_hands(hands)
                if sign_info and sign_info.get("sign_name") != last_reported_sign:
                    last_reported_sign = sign_info.get("sign_name")
                    print(f"[MARG-One:Sign] {last_reported_sign} -> Value: {sign_info.get('value')} (Hands: {len(hands)})")

            # 3. Render Visuals
            frame = visualizer.draw_skeleton(frame, hands)
            if mouse_mode_active and cursor_telemetry:
                frame = visualizer.draw_cursor_overlay(frame, cursor_telemetry)
            frame = visualizer.draw_hud(
                frame,
                fps=fps,
                num_hands=len(hands),
                sign_info=sign_info,
                cursor_telemetry=cursor_telemetry if mouse_mode_active else None,
            )

            # Display
            cv2.imshow("MARG-One: Dual-Hand System", frame)

            # Key Handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("[*] Shutdown signal received.")
                break
            elif key in (ord('m'), ord('M')):
                mouse_mode_active = not mouse_mode_active
                cursor_controller.enable_active_control = mouse_mode_active
                status = "ACTIVATED" if mouse_mode_active else "DEACTIVATED (Gesture Mode)"
                print(f"[*] Mouse control mode: {status}")
            elif key in (ord('s'), ord('S')):
                visualizer.show_skeleton = not visualizer.show_skeleton
            elif key in (ord('b'), ord('B')):
                visualizer.show_bbox = not visualizer.show_bbox
            elif key in (ord('f'), ord('F')):
                visualizer.show_finger_status = not visualizer.show_finger_status
            elif key in (ord('z'), ord('Z')):
                visualizer.show_interaction_box = not visualizer.show_interaction_box

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        print("[*] MARG-One Vision subsystem exited cleanly.")


if __name__ == "__main__":
    main()
