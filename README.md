# MARG-One: Modular Multimodal AI & Robotics Framework

## Subsystems: Vision Kinematics, Gesture Recognition & Hand Cursor Control

---

## 1. Overview

MARG-One is a modular robotics and intelligent computing framework designed for real-time perception, spatial reasoning, and multimodal human-machine interaction.

The system integrates two core subsystems:
1. **Vision Subsystem**: Real-time dual-hand tracking, 3D anatomical skeleton extraction, digit kinematic analysis, and extensible gesture-to-value mapping.
2. **Control Subsystem**: Vision-based PC mouse cursor navigation driven by index finger tracking with pinch-to-click actuation.

---

## 2. Key Capabilities

- **Concurrent Dual-Hand Tracking**: Detects and tracks up to two hands simultaneously with low inference latency.
- **21 3D Topological Landmarks**: Full spatial extraction per hand with normalized $(x, y, z)$, pixel $(u, v)$, and real-world metric coordinates $(x, y, z \text{ in meters})$.
- **Handedness Disambiguation**: Left vs. Right hand classification with per-hand prediction confidence scoring.
- **Digit Kinematic Evaluation**: Rotation-invariant binary extension determination for Thumb, Index, Middle, Ring, and Pinky digits.
- **Hand Cursor Navigation**: Seamlessly controls the operating system mouse cursor via the index finger of either hand.
- **Pinch-to-Click & Drag**: Intuitive thumb-index pinch detection triggering left mouse clicks, click-and-drag operations, and instant release.
- **Motion Smoothing & Jitter Reduction**: Exponential Moving Average (EMA) and velocity-sensitive interpolation for smooth cursor motion.
- **Telemetry HUD**: Anti-aliased visualizer displaying joint nodes, bone lines, bounding bounds, active interaction zones, and real-time FPS.

---

## 3. Architecture & Repository Structure

```
MARG-One/
|-- marg_one/                     # Core MARG-One Package
|   |-- __init__.py               # Top-level package exports
|   |-- core/                     # Base framework interfaces
|   |   `-- __init__.py           # Subsystem telemetry & state definitions
|   |-- vision/                   # Vision subsystem
|   |   |-- __init__.py           # Vision module entry exports
|   |   |-- model_loader.py       # Automated model asset manager
|   |   |-- tracker.py            # DualHandTracker & HandData kinematic structures
|   |   |-- visualizer.py         # HandVisualizer rendering engine
|   |   `-- sign_processor.py     # SignProcessor gesture rule evaluator
|   `-- control/                  # Control subsystem
|       |-- __init__.py           # Control module entry exports
|       `-- cursor_controller.py  # HandCursorController & OS mouse driver
|-- tests/                        # Automated test suites
|   |-- __init__.py
|   |-- test_vision.py            # Vision pipeline test suite
|   `-- test_cursor_controller.py # Mouse control unit and integration tests
|-- .gitignore                    # Version control exclusion rules
|-- main.py                       # Unified interactive runner (Gestures + Cursor)
|-- run_cursor_control.py         # Dedicated hand cursor controller launcher
|-- requirements.txt              # Production dependencies
|-- setup.py                      # Package installation configuration
`-- README.md                     # System documentation
```

---

## 4. Subsystem Integration & Usage

### 4.1 Vision & Sign Interpretation

```python
from marg_one.vision import DualHandTracker, SignProcessor, HandVisualizer
import cv2

tracker = DualHandTracker(num_hands=2, min_detection_confidence=0.5)
sign_processor = SignProcessor()
visualizer = HandVisualizer()

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hands = tracker.process_frame(frame)
    telemetry = sign_processor.process_hands(hands)
    if telemetry:
        print(f"Sign: {telemetry['sign_name']} -> Value: {telemetry['value']}")

    frame = visualizer.draw_skeleton(frame, hands)
    cv2.imshow("MARG-One Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
tracker.close()
```

### 4.2 Hand Cursor & Mouse Controller

```python
from marg_one.vision import DualHandTracker
from marg_one.control import HandCursorController
import cv2

tracker = DualHandTracker(num_hands=2)
cursor_controller = HandCursorController(
    smoothing_factor=0.35,
    pinch_threshold=38.0,
    enable_active_control=True,
)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hands = tracker.process_frame(frame)
    telemetry = cursor_controller.update(hands, frame.shape)
    
    # State: IDLE, MOVING, CLICK_DOWN, DRAGGING, RELEASED
    print(f"Cursor: {telemetry['screen_pos']} | State: {telemetry['state']}")

cap.release()
```

---

## 5. Topological Landmark Mapping

Each detected hand yields 21 anatomical landmark nodes:

```
          8 (INDEX_TIP)    12 (MIDDLE_TIP)    16 (RING_TIP)    20 (PINKY_TIP)
          |                |                  |                |
          7 (INDEX_DIP)    11 (MIDDLE_DIP)    15 (RING_DIP)    19 (PINKY_DIP)
          |                |                  |                |
  4 (TIP) 6 (INDEX_PIP)    10 (MIDDLE_PIP)    14 (RING_PIP)    18 (PINKY_PIP)
    \     |                |                  |                |
     3    5 (INDEX_MCP)----9 (MIDDLE_MCP)----13 (RING_MCP)-----17 (PINKY_MCP)
      \     \                                                 /
       2     \                                               /
        \     \                                             /
         1     \                                           /
          \     \                                         /
           \---- 0 (WRIST) ------------------------------/
```

### Landmark Index Reference

| Index | Joint Identifier | Description |
|---|---|---|
| `0` | `WRIST` | Base wrist reference origin |
| `1 - 4` | `THUMB_CMC`, `THUMB_MCP`, `THUMB_IP`, `THUMB_TIP` | Thumb carpal to distal tip |
| `5 - 8` | `INDEX_FINGER_MCP`, `PIP`, `DIP`, `TIP` | Index digit joints and tip (Cursor Driver) |
| `9 - 12` | `MIDDLE_FINGER_MCP`, `PIP`, `DIP`, `TIP` | Middle digit joints and tip |
| `13 - 16` | `RING_FINGER_MCP`, `PIP`, `DIP`, `TIP` | Ring digit joints and tip |
| `17 - 20` | `PINKY_MCP`, `PIP`, `DIP`, `TIP` | Pinky digit joints and tip |

---

## 6. Cursor Control Mechanics

- **Movement**: Tracked using the index fingertip (`LandmarkIndex.INDEX_FINGER_TIP`, node 8) across the full camera frame, mapped directly to display bounds $(W_{screen}, H_{screen})$ without restricting boundaries.
- **Pinch-to-Click**: Euclidean distance between Index Tip (node 8) and Thumb Tip (node 4):
  - $\text{Distance} < \text{Threshold}$ ($\approx 38\text{px}$): Triggers OS Left Mouse Down.
  - Holding pinch while moving maintains OS Left Mouse Drag (selection, window drag).
  - $\text{Distance} \ge \text{Threshold}$: Triggers OS Left Mouse Up.

---

## 7. Installation & Setup

### Requirements
- Python 3.9 or higher (tested on Python 3.13)
- Windows OS (for native `ctypes` mouse driver; mock driver used on other platforms)
- Webcam / Camera input device

### Installation
```bash
# Clone the repository
git clone https://github.com/Arnav-G-null/MARG-One.git
cd MARG-One

# Install dependencies
pip install -r requirements.txt

# (Optional) Install as editable package
pip install -e .
```

---

## 8. Running Applications

### 8.1 Unified Launcher (`main.py`)
```bash
# Run in Gesture / Sign Recognition mode
python main.py

# Run in Hand Mouse Controller mode
python main.py --mouse
```

### 8.2 Dedicated Hand Cursor Launcher (`run_cursor_control.py`)
```bash
# Launch cursor controller
python run_cursor_control.py

# Launch with custom sensitivity and pinch distance
python run_cursor_control.py --smooth 0.40 --pinch 35.0
```

### Runtime Keyboard Controls

| Key | Function |
|---|---|
| `q` or `ESC` | Exit application cleanly |
| `m` | Toggle active mouse control mode on/off |
| `s` | Toggle hand skeleton lines |
| `b` | Toggle bounding boxes |
| `f` | Toggle digit state indicators (`T:1 I:1 M:0 R:0 P:0`) |
| `z` | Toggle active screen interaction zone boundary |

---

## 9. Automated Testing

Run the automated offline test suites:

```bash
# Test Hand Cursor Controller
python tests/test_cursor_controller.py

# Test Vision Subsystem
python tests/test_vision.py
```

---

## 10. License & Maintainers

- **Author**: Arnav Garg
- **Repository**: [https://github.com/Arnav-G-null/MARG-One](https://github.com/Arnav-G-null/MARG-One)
- **Project**: MARG-One Multimodal Framework
