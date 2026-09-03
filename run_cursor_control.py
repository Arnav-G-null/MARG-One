"""
MARG-One Vision-Based Palm-Center & Pinch-to-Click Cursor Controller Application.
Controls PC mouse cursor via anatomical palm tracking and triggers left click via index-thumb pinch.
Uses 1 Euro Filter for ultra-smooth, zero-jitter tracking.
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
    parser = argparse.ArgumentParser(description="MARG-One Precision Palm-Center & Pinch-Click Controller")
    parser.add_argument("--cam", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Frame height (default: 720)")
    parser.add_argument("--speed", type=float, default=1.35, help="Cursor speed / reach gain multiplier (default: 1.35)")
    parser.add_argument("--min-cutoff", type=float, default=0.5, help="1 Euro filter min cutoff frequency (default: 0.5)")
    parser.add_argument("--beta", type=float, default=0.005, help="1 Euro filter speed coefficient (default: 0.005)")
    parser.add_argument("--pinch", type=float, default=38.0, help="Pinch threshold in pixels (default: 38.0)")
    parser.add_argument("--no-flip", action="store_true", help="Disable mirror horizontal flip")
    parser.add_argument("--no-action", action="store_true", help="Disable actual OS mouse control (Telemetry preview only)")
    args = parser.parse_args()

    print("=" * 72)
    print("      MARG-ONE : PRECISION PALM-CENTER & PINCH-CLICK CONTROLLER      ")
    print("=" * 72)
    print(f"[*] Camera Index:       {args.cam}")
    print(f"[*] Resolution:         {args.width}x{args.height}")
    print(f"[*] Speed Gain:         {args.speed}x")
    print(f"[*] 1 Euro Filter:      min_cutoff={args.min_cutoff}, beta={args.beta}")
    print(f"[*] Pinch Threshold:    {args.pinch} px")
    print(f"[*] Active Mouse Drive: {not args.no_action}")
    print(f"[*] Control Rules:")
    print(f"    - Move Palm         : Navigates PC cursor with ultra-stable palm center")
    print(f"    - Pinch (Thumb+Index): Triggers Left Click (Locks position to prevent slip)")
    print(f"    - Hold Pinch & Move : Executes continuous Mouse Drag & Drop")
    print(f"    - Release Pinch     : Releases Left Click")
    print(f"[*] Keyboard Shortcuts:")
    print(f"    - 'q' or ESC : Exit application cleanly")
    print(f"    - 'm'        : Toggle active OS mouse control on/off")
    print(f"    - 's'        : Toggle hand skeleton lines")
    print("=" * 72)

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
        speed_gain=args.speed,
        min_cutoff=args.min_cutoff,
        beta=args.beta,
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

            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                current_fps = 1.0 / dt
                fps = 0.9 * fps + 0.1 * current_fps if fps > 0 else current_fps

            # 1. Track Hands
            hands = tracker.process_frame(frame)

            # 2. Update Palm-Center Cursor Controller
            telemetry = cursor_controller.update(hands, frame.shape, timestamp=curr_time)
            current_state = telemetry.get("state")

            # Console telemetry logging on state transition
            if current_state != last_state:
                last_state = current_state
                pos = telemetry.get("screen_pos")
                pinch_d = telemetry.get("pinch_distance", 0.0)
                print(f"[PALM MOUSE] State: {current_state:<12} | Screen: ({pos[0]:>4}, {pos[1]:>4}) | Pinch Dist: {pinch_d}px")

            # 3. Render Visual Feedback
            frame = visualizer.draw_skeleton(frame, hands)
            frame = visualizer.draw_cursor_overlay(frame, telemetry)
            frame = visualizer.draw_hud(frame, fps=fps, num_hands=len(hands), cursor_telemetry=telemetry)

            cv2.imshow("MARG-One: Precision Palm Mouse Controller", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("[*] Exiting palm cursor controller...")
                break
            elif key in (ord('m'), ord('M')):
                cursor_controller.enable_active_control = not cursor_controller.enable_active_control
                status = "ENABLED" if cursor_controller.enable_active_control else "DISABLED (Preview Only)"
                print(f"[*] Active OS mouse control: {status}")
            elif key in (ord('s'), ord('S')):
                visualizer.show_skeleton = not visualizer.show_skeleton

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        print("[*] Hand cursor controller terminated cleanly.")


if __name__ == "__main__":
    main()
