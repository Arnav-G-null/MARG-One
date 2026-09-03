# MARG-One: Modular Multimodal AI & Robotics Framework

## Subsystems: Vision Kinematics, Gesture Recognition & Precision Palm Cursor Control

---

## 1. Overview

MARG-One is a modular robotics and intelligent computing framework designed for real-time perception, spatial reasoning, and multimodal human-machine interaction.

The system integrates two core subsystems:
1. **Vision Subsystem**: Real-time dual-hand tracking, 3D anatomical skeleton extraction, digit kinematic analysis, and extensible gesture-to-value mapping.
2. **Control Subsystem**: Vision-based PC mouse cursor navigation driven by anatomical **Palm-Center tracking**, **Closed-Palm (Fist) left-clicking**, and the **1 Euro Filter** for zero-jitter, pixel-accurate control.

---

## 2. Key Capabilities

- **Concurrent Dual-Hand Tracking**: Detects and tracks up to two hands simultaneously with low inference latency.
- **21 3D Topological Landmarks**: Full spatial extraction per hand with normalized $(x, y, z)$, pixel $(u, v)$, and real-world metric coordinates $(x, y, z \text{ in meters})$.
- **Handedness Disambiguation**: Left vs. Right hand classification with per-hand prediction confidence scoring.
- **Digit Kinematic Evaluation**: Rotation-invariant binary extension determination for Thumb, Index, Middle, Ring, and Pinky digits.
- **Anatomical Palm-Center Navigation**: Tracks the rigid anatomical centroid of the palm (Wrist + MCP joints) for maximum physical stability.
- **Closed-Palm (Fist) Left Clicking**: Natural, fatigue-free clicking mechanism: open palm moves the cursor; closing the palm executes a Left Click Down.
- **Click-Lock Anti-Slip Stabilization**: Temporarily locks cursor coordinates for 120ms during fist closure so clicks land precisely without drifting off target.
- **Continuous Mouse Dragging**: Holding a closed fist while moving enables drag-and-drop (window movement, text selection, file transfer).
- **1 Euro Filter Precision Smoothing**: Industry-standard speed-adaptive low-pass filter (Casiez et al., CHI 2012) eliminating micro-jitter during slow moves while preserving instant response during rapid sweeps.
- **Telemetry HUD**: Anti-aliased visualizer displaying joint nodes, bone lines, bounding bounds, dynamic hand closure gauge, and real-time FPS.

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
|       |-- one_euro_filter.py    # 1 Euro Filter precision smoothing engine
|       `-- cursor_controller.py  # Precision Palm Cursor & Fist-Click driver
|-- tests/                        # Automated test suites
|   |-- __init__.py
|   |-- test_vision.py            # Vision pipeline test suite
|   `-- test_cursor_controller.py # Mouse control unit and precision tests
|-- .gitignore                    # Version control exclusion rules
|-- main.py                       # Unified interactive runner (Gestures + Cursor)
|-- run_cursor_control.py         # Dedicated palm cursor controller launcher
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

### 4.2 Precision Palm Cursor & Fist-Click Controller

```python
from marg_one.vision import DualHandTracker
from marg_one.control import HandCursorController
import cv2

tracker = DualHandTracker(num_hands=2)
cursor_controller = HandCursorController(
    speed_gain=1.35,
    min_cutoff=0.5,
    beta=0.005,
    click_close_threshold=0.62,
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
    print(f"Cursor: {telemetry['screen_pos']} | State: {telemetry['state']} | Fist: {telemetry['closure_score']}")

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
| `0` | `WRIST` | Palm centroid anchor base |
| `1 - 4` | `THUMB_CMC`, `THUMB_MCP`, `THUMB_IP`, `THUMB_TIP` | Thumb carpal to distal tip |
| `5 - 8` | `INDEX_FINGER_MCP`, `PIP`, `DIP`, `TIP` | Index digit joints and tip |
| `9 - 12` | `MIDDLE_FINGER_MCP`, `PIP`, `DIP`, `TIP` | Middle digit joints and tip |
| `13 - 16` | `RING_FINGER_MCP`, `PIP`, `DIP`, `TIP` | Ring digit joints and tip |
| `17 - 20` | `PINKY_MCP`, `PIP`, `DIP`, `TIP` | Pinky digit joints and tip |

---

## 6. Cursor Control Mechanics

- **Movement**: Driven by the anatomical **Palm Center** (centroid of landmarks `0, 5, 9, 13, 17`), mapped linearly across the full camera frame with adaptive speed gain.
- **Jitter Filtering**: 1 Euro Filter adapts its cutoff frequency dynamically based on hand speed:
  - Stationary: Cutoff drops low to completely absorb camera noise and sensor jitter.
  - Moving: Cutoff increases automatically for instantaneous cursor responsiveness.
- **Left Click**: **Closing the Palm (Fist)**:
  - Continuous closure ratio computed from all 5 fingertips relative to the palm scale.
  - Clenching fist ($\ge 62\%$ closure) triggers OS Left Mouse Down.
  - **Click-Lock Stabilization**: Freezes cursor coordinates for 120ms upon closure to guarantee the click hits the exact desired pixel without drift.
- **Drag & Drop**: Keeping the palm closed while moving executes continuous OS Mouse Drag.
- **Release**: Opening the palm back up triggers OS Left Mouse Up.

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

# Run in Precision Palm Cursor mode
python main.py --mouse
```

### 8.2 Dedicated Palm Cursor Launcher (`run_cursor_control.py`)
```bash
# Launch palm cursor controller
python run_cursor_control.py

# Launch with custom speed gain and fist threshold
python run_cursor_control.py --speed 1.50 --close-thresh 0.60
```

### Runtime Keyboard Controls

| Key | Function |
|---|---|
| `q` or `ESC` | Exit application cleanly |
| `m` | Toggle active mouse control mode on/off |
| `s` | Toggle hand skeleton lines |
| `b` | Toggle bounding boxes |
| `f` | Toggle digit state indicators (`T:1 I:1 M:0 R:0 P:0`) |

---

## 9. Automated Testing

Run the automated offline test suites:

```bash
# Test Precision Palm Cursor & 1 Euro Filter
python tests/test_cursor_controller.py

# Test Vision Subsystem
python tests/test_vision.py
```

---

## 10. License & Maintainers

- **Author**: Arnav Garg
- **Repository**: [https://github.com/Arnav-G-null/MARG-One](https://github.com/Arnav-G-null/MARG-One)
- **Project**: MARG-One Multimodal Framework
