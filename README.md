# MARG-One: Modular Multimodal AI & Robotics Framework

## Vision Subsystem: Dual-Hand 3D Kinematics & Gesture Recognition

---

## 1. Overview

MARG-One is a modular robotics and intelligent computing framework designed for real-time perception, spatial reasoning, and multimodal human-machine interaction. 

This repository houses the core **Vision Subsystem**, providing real-time dual-hand tracking, 3D anatomical skeleton extraction, digit kinematic analysis, and extensible gesture-to-value mapping for downstream robotic controls, assistive interfaces, and interactive systems.

---

## 2. Key Capabilities

- **Concurrent Dual-Hand Tracking**: Detects and tracks up to two hands simultaneously with low inference latency.
- **21 3D Topological Landmarks**: Full spatial extraction per hand with normalized $(x, y, z)$, pixel $(u, v)$, and real-world metric coordinates $(x, y, z \text{ in meters})$.
- **Handedness Disambiguation**: Left vs. Right hand classification with per-hand prediction confidence scoring.
- **Digit Kinematic Evaluation**: Rotation-invariant binary extension determination for Thumb, Index, Middle, Ring, and Pinky digits.
- **Extensible Value Mapping**: Decoupled sign interpreter providing structured value returns and custom evaluator hooks.
- **Diagnostic Telemetry HUD**: High-contrast, anti-aliased visualizer displaying joint nodes, anatomical bone lines, bounding bounds, and real-time FPS.

---

## 3. Architecture & Repository Structure

```
MARG-One/
|-- marg_one/                     # Core MARG-One Package
|   |-- __init__.py               # Top-level package exports
|   |-- core/                     # Base framework interfaces
|   |   `-- __init__.py           # Subsystem telemetry & state definitions
|   `-- vision/                   # Vision subsystem
|       |-- __init__.py           # Vision module entry exports
|       |-- model_loader.py       # Automated model asset manager
|       |-- tracker.py            # DualHandTracker & HandData kinematic structures
|       |-- visualizer.py         # HandVisualizer rendering engine
|       `-- sign_processor.py     # SignProcessor gesture rule evaluator
|-- tests/                        # Automated test suites
|   |-- __init__.py
|   `-- test_vision.py            # Headless unit and pipeline integration tests
|-- .gitignore                    # Version control exclusion rules
|-- main.py                       # Standalone camera runner application
|-- requirements.txt              # Production dependencies
|-- setup.py                      # Package installation configuration
`-- README.md                     # System documentation
```

---

## 4. Integration into Bigger Systems

The Vision Subsystem is packaged as an independent module (`marg_one.vision`) that seamlessly integrates into larger autonomous or interactive pipelines.

### Module Usage Example

```python
from marg_one.vision import DualHandTracker, SignProcessor, HandVisualizer
import cv2

# Initialize tracker
tracker = DualHandTracker(num_hands=2, min_detection_confidence=0.5)
sign_processor = SignProcessor()
visualizer = HandVisualizer()

# Register custom gesture rule
def custom_rule(hands):
    if len(hands) == 2:
        # Custom logic using 3D landmarks or finger patterns
        return {"sign_name": "CUSTOM_ACTION", "value": 0x42, "confidence": 0.99}
    return None

sign_processor.register_custom_sign(custom_rule)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Extract 3D hand data
    hands = tracker.process_frame(frame)

    # Evaluate signs
    telemetry = sign_processor.process_hands(hands)
    if telemetry:
        print(f"Action: {telemetry['sign_name']} -> Value: {telemetry['value']}")

    # Render diagnostics
    frame = visualizer.draw_skeleton(frame, hands)
    cv2.imshow("MARG-One Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
tracker.close()
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
| `5 - 8` | `INDEX_FINGER_MCP`, `PIP`, `DIP`, `TIP` | Index digit joints and tip |
| `9 - 12` | `MIDDLE_FINGER_MCP`, `PIP`, `DIP`, `TIP` | Middle digit joints and tip |
| `13 - 16` | `RING_FINGER_MCP`, `PIP`, `DIP`, `TIP` | Ring digit joints and tip |
| `17 - 20` | `PINKY_MCP`, `PIP`, `DIP`, `TIP` | Pinky digit joints and tip |

---

## 6. Coordinate Frames

1. **Normalized Image Frame**:
   - $x \in [0.0, 1.0]$: Horizontal position relative to image width.
   - $y \in [0.0, 1.0]$: Vertical position relative to image height.
   - $z$: Relative depth coordinate normalized to wrist scale.

2. **Pixel Coordinate Frame**:
   - $px = \lfloor x \times W \rfloor$, $py = \lfloor y \times H \rfloor$.

3. **World Coordinate Frame**:
   - 3D metric coordinates $(X, Y, Z)$ measured in meters with origin centered at the hand's geometric anchor.

---

## 7. Sign & Value Evaluation Engine

The `SignProcessor` interprets single and dual-hand skeletal states into discrete payloads:

```python
{
    "sign_name": "TWO_HAND_HEART",
    "value": "HEART_SIGN_VALUE",
    "confidence": 0.95,
    "hands_count": 2,
    "details": {
        "index_dist": 28.4,
        "thumb_dist": 31.2
    }
}
```

### Pre-Configured Sign Patterns

| Sign Identifier | Hand Mode | Digit Bitmask (`TIMRP`) | Output Value |
|---|---|---|---|
| `TWO_HAND_HEART` | Dual | Tips proximate ($< 65\text{px}$) | `HEART_SIGN_VALUE` |
| `BOTH_OPEN_PALMS` | Dual | `11111` / `11111` | `OPEN_PALMS_VALUE` |
| `DOUBLE_PEACE` | Dual | `01100` / `01100` | `DOUBLE_PEACE_VALUE` |
| `DOUBLE_FIST` | Dual | `00000` / `00000` | `DOUBLE_FIST_VALUE` |
| `THUMBS_UP` | Single | `10000` | `THUMBS_UP` |
| `PEACE / V` | Single | `01100` | `PEACE` |
| `OPEN_PALM` | Single | `11111` | `PALM` |
| `FIST` | Single | `00000` | `FIST` |

---

## 8. Installation & Setup

### Requirements
- Python 3.9 or higher (tested on Python 3.13)
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

## 9. Running Standalone Application

```bash
# Launch default camera feed (Index 0, 1280x720)
python main.py

# Launch with custom arguments
python main.py --cam 0 --width 1920 --height 1080 --conf 0.6
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--cam` | `int` | `0` | Camera capture device index |
| `--width` | `int` | `1280` | Frame capture width resolution |
| `--height` | `int` | `720` | Frame capture height resolution |
| `--conf` | `float` | `0.5` | Detection and tracking confidence threshold |
| `--no-flip` | `flag` | `False` | Disables mirror horizontal frame flip |

### Runtime Keyboard Shortcuts

| Key | Function |
|---|---|
| `q` or `ESC` | Exit application cleanly |
| `s` | Toggle hand skeleton rendering |
| `b` | Toggle bounding boxes |
| `f` | Toggle digit state telemetry (`T:1 I:1 M:0 R:0 P:0`) |
| `i` | Toggle topological landmark ID numbering |

---

## 10. Automated Testing

Run the automated offline unit and pipeline verification suite:

```bash
python tests/test_vision.py
```

---

## 11. License & Maintainers

- **Author**: Arnav Garg
- **Repository**: [https://github.com/Arnav-G-null/MARG-One](https://github.com/Arnav-G-null/MARG-One)
- **Project**: MARG-One Multimodal Framework
