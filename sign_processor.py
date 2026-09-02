"""
Sign and Gesture Processing Module.
Evaluates single-hand and dual-hand skeleton data to produce custom return values.
"""

from typing import List, Dict, Any, Optional, Tuple
import math
from hand_tracker import HandData, LandmarkIndex


class SignProcessor:
    """
    Evaluates 3D hand skeleton landmarks and finger states to recognize signs and return values.
    You can easily define your own signs and mapped return values in this class.
    """

    def __init__(self):
        # Register custom sign handlers if desired
        self.custom_rules = {}

    def _get_left_and_right(self, hands: List[HandData]) -> Tuple[Optional[HandData], Optional[HandData]]:
        """Separates detected hands into left and right instances."""
        left_hand = None
        right_hand = None
        for hand in hands:
            if hand.handedness.lower() == "left" and left_hand is None:
                left_hand = hand
            elif hand.handedness.lower() == "right" and right_hand is None:
                right_hand = hand
            elif left_hand is None:
                left_hand = hand
            elif right_hand is None:
                right_hand = hand
        return left_hand, right_hand

    def _get_finger_pattern(self, hand: HandData) -> str:
        """Returns a compact 5-character string of finger states: T I M R P (1=open, 0=closed)."""
        st = hand.finger_states
        t = "1" if st.get("thumb") else "0"
        i = "1" if st.get("index") else "0"
        m = "1" if st.get("middle") else "0"
        r = "1" if st.get("ring") else "0"
        p = "1" if st.get("pinky") else "0"
        return f"{t}{i}{m}{r}{p}"

    def process_hands(self, hands: List[HandData]) -> Optional[Dict[str, Any]]:
        """
        Main evaluation method.
        Analyzes the detected hands (1 or 2) and returns recognized sign metadata and values.
        """
        if not hands:
            return None

        left_hand, right_hand = self._get_left_and_right(hands)

        # -------------------------------------------------------------
        # 1. TWO-HAND SIGNS (When both hands are visible to camera)
        # -------------------------------------------------------------
        if len(hands) >= 2 and left_hand and right_hand:
            # Extract features between both hands
            left_index_tip = left_hand.get_landmark_px(LandmarkIndex.INDEX_FINGER_TIP)
            right_index_tip = right_hand.get_landmark_px(LandmarkIndex.INDEX_FINGER_TIP)
            left_thumb_tip = left_hand.get_landmark_px(LandmarkIndex.THUMB_TIP)
            right_thumb_tip = right_hand.get_landmark_px(LandmarkIndex.THUMB_TIP)

            index_tip_distance = math.hypot(
                left_index_tip[0] - right_index_tip[0],
                left_index_tip[1] - right_index_tip[1]
            )
            thumb_tip_distance = math.hypot(
                left_thumb_tip[0] - right_thumb_tip[0],
                left_thumb_tip[1] - right_thumb_tip[1]
            )

            left_pattern = self._get_finger_pattern(left_hand)
            right_pattern = self._get_finger_pattern(right_hand)

            # Example Two-Hand Sign 1: Heart Gesture (thumbs touching & index fingers touching/forming arch)
            if index_tip_distance < 60 and thumb_tip_distance < 60:
                return {
                    "sign_name": "TWO_HAND_HEART",
                    "value": "HEART_SIGN_VALUE",  # Custom value to be defined by user
                    "confidence": 0.95,
                    "hands_count": 2,
                    "details": {"index_dist": index_tip_distance, "thumb_dist": thumb_tip_distance},
                }

            # Example Two-Hand Sign 2: Both Open Palms (all fingers extended on both hands)
            if left_pattern == "11111" and right_pattern == "11111":
                return {
                    "sign_name": "BOTH_OPEN_PALMS",
                    "value": "OPEN_PALMS_VALUE",
                    "confidence": 0.98,
                    "hands_count": 2,
                    "details": {"left": left_pattern, "right": right_pattern},
                }

            # Example Two-Hand Sign 3: Both Peace / Victory Signs (Index + Middle extended)
            if left_pattern in ("01100", "11100") and right_pattern in ("01100", "11100"):
                return {
                    "sign_name": "DOUBLE_PEACE",
                    "value": "DOUBLE_PEACE_VALUE",
                    "confidence": 0.94,
                    "hands_count": 2,
                    "details": {"left": left_pattern, "right": right_pattern},
                }

            # Example Two-Hand Sign 4: Both Fists (all fingers closed)
            if left_pattern == "00000" and right_pattern == "00000":
                return {
                    "sign_name": "DOUBLE_FIST",
                    "value": "DOUBLE_FIST_VALUE",
                    "confidence": 0.92,
                    "hands_count": 2,
                    "details": {"left": left_pattern, "right": right_pattern},
                }

            # Default Dual-Hand generic state
            return {
                "sign_name": f"DUAL_HANDS ({left_pattern} | {right_pattern})",
                "value": f"VALUE_L{left_pattern}_R{right_pattern}",
                "confidence": (left_hand.score + right_hand.score) / 2.0,
                "hands_count": 2,
                "details": {
                    "left_pattern": left_pattern,
                    "right_pattern": right_pattern,
                    "index_tip_distance": round(index_tip_distance, 1),
                },
            }

        # -------------------------------------------------------------
        # 2. SINGLE-HAND SIGNS (When 1 hand is visible)
        # -------------------------------------------------------------
        hand = hands[0]
        pattern = self._get_finger_pattern(hand)

        if pattern == "11111":
            sign_name = "OPEN_PALM"
            value = "PALM"
        elif pattern == "00000":
            sign_name = "FIST"
            value = "FIST"
        elif pattern in ("01100", "11100"):
            sign_name = "PEACE / V"
            value = "PEACE"
        elif pattern in ("10000", "10001"):
            sign_name = "THUMBS_UP"
            value = "THUMBS_UP"
        elif pattern == "01000":
            sign_name = "POINTING"
            value = "POINT"
        elif pattern in ("11001", "01001"):
            sign_name = "ROCK / HORNS"
            value = "ROCK"
        else:
            sign_name = f"{hand.handedness.upper()}_{pattern}"
            value = f"VAL_{pattern}"

        return {
            "sign_name": sign_name,
            "value": value,
            "confidence": hand.score,
            "hands_count": 1,
            "details": {"hand": hand.handedness, "pattern": pattern},
        }
