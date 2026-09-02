"""
MARG-One: Modular Multimodal AI & Robotics Framework.
"""

__version__ = "0.1.0"
__author__ = "Arnav Garg"

from marg_one.vision.tracker import DualHandTracker, HandData, HandLandmark, LandmarkIndex
from marg_one.vision.visualizer import HandVisualizer, VisualizerTheme
from marg_one.vision.sign_processor import SignProcessor

__all__ = [
    "DualHandTracker",
    "HandData",
    "HandLandmark",
    "LandmarkIndex",
    "HandVisualizer",
    "VisualizerTheme",
    "SignProcessor",
]
