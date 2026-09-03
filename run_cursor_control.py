"""
MARG-One Vision-Based Hand Cursor Controller Application.
Controls PC mouse cursor via index finger tracking and triggers left click via pinch.
"""

import argparse
import time
import sys
import os
import cv2

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from marg_one.vision import DualHandTracker, HandVisualizer
from marg_one.control import HandCursorController, CursorState


def main():
    parser = argparse.ArgumentParser(description="MARG-One Hand Cursor & Pinch-to-Click Controller")
    parser.add_argument("--cam", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Frame height (default: 720)")
    parser.add_argument("--smooth", type=float, default=0.35, help="Cursor smoothing factor [0.1 - 0.8] (default: 0.35)")
    parser.add_argument("--pinch", type=float, default=38.0, help="Pinch threshold in pixels (default: 38.0)")
    parser.add_argument("--margin-x", type=float, default=0.0, help="Horizontal boundary margin (default: 0.0 for full frame)")
    parser.add_argument("--margin-y", type=float, default=0.0, help="Vertical boundary margin (default: 0.0 for full frame)")
    parser.add_argument("--no-flip", action="store_true", help="Disable mirror horizontal flip")
    parser.add_argument("--no-action", action="store_true", help="Disable actual OS mouse control (Telemetry preview only)")
    args = parser.parse_args()

    print("=" * 68)
    print("      MARG-ONE : HAND CURSOR & PINCH-TO-CLICK CONTROLLER     ")
    print("=" * 68)
    print(f"[*] Camera Index:       {args.cam}")
    print(f"[*] Resolution:         {args.width}x{args.height}")
    print(f"[*] Smoothing Factor:   {args.smooth}")
    print(f"[*] Pinch Threshold:    {args.pinch} px")
    print(f"[*] Active Mouse Drive: {not args.no_action}")
    print(f"[*] Control Rules:")
    print(f"    - Point Index Finger : Navigates PC cursor smoothly")
    print(f"    - Pinch (Index+Thumb): Triggers Left Click / Drag")
    print(f"    - Release Pinch      : Releases Left Click")
    print(f"[*] Keyboard Shortcuts:")
    print(f"    - 'q' or ESC : Exit application")
    print(f"    - 'm'        : Toggle active OS mouse control on/off")
    print(f"    - 'z'        : Toggle active screen zone overlay")
    print(f"    - 's'        : Toggle hand skeletons")
    print("=" * 68)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[ERROR] Could not access webcam device at index {args.cam}.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Initialize Modules
    tracker = DualHandTracker(num_hands=2, min_detection_confidence=0.5)
    visualizer = HandVisualizer()
    cursor_controller = HandCursorController(
        margin_x=args.margin_x,
        margin_y=args.margin_y,
        smoothing_factor=args.smooth,
        pinch_threshold=args.pinch,
        enable_active_control=not args.no_action,
    )

    fps = 0.0
    prev_time = time.time()
    last_state = CursorState.IDLE

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Camera frame capture failure.")
                break

            if not args.no_flip:
                frame = cv2.flip(frame, 1)

            # FPS calculation
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                current_fps = 1.0 / dt
                fps = 0.9 * fps + 0.1 * current_fps if fps > 0 else current_fps

            # 1. Track Hands
            hands = tracker.process_frame(frame)

            # 2. Update Cursor Controller
            telemetry = cursor_controller.update(hands, frame.shape)
            current_state = telemetry.get("state")

            # Telemetry logging on state transition
            if current_state != last_state:
                last_state = current_state
                pos = telemetry.get("screen_pos")
                print(f"[CURSOR STATE] {current_state} at Screen ({pos[0]}, {pos[1]}) | Pinch: {telemetry.get('pinch_distance')}px")

            # 3. Render Visual Feedback
            frame = visualizer.draw_skeleton(frame, hands)
            frame = visualizer.draw_cursor_overlay(frame, telemetry)
            frame = visualizer.draw_hud(frame, fps=fps, num_hands=len(hands), cursor_telemetry=telemetry)

            cv2.imshow("MARG-One: Hand Cursor Controller", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("[*] Exiting hand cursor controller...")
                break
            elif key in (ord('m'), ord('M')):
                cursor_controller.enable_active_control = not cursor_controller.enable_active_control
                status = "ENABLED" if cursor_controller.enable_active_control else "DISABLED (Preview Only)"
                print(f"[*] Active OS mouse control: {status}")
            elif key in (ord('z'), ord('Z')):
                visualizer.show_interaction_box = not visualizer.show_interaction_box
            elif key in (ord('s'), ord('S')):
                visualizer.show_skeleton = not visualizer.show_skeleton

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        print("[*] Hand cursor controller terminated cleanly.")


if __name__ == "__main__":
    main()
