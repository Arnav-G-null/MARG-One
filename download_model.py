"""
Model downloader for MediaPipe Hand Landmarker Task model.
"""

import os
import urllib.request
import sys

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")


def ensure_model_downloaded(model_path: str = DEFAULT_MODEL_PATH) -> str:
    """
    Checks if the MediaPipe hand landmarker task model exists.
    If not, downloads it from Google's official storage.
    """
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        return model_path

    print(f"[HandTracker] Downloading MediaPipe Hand Landmarker model to '{model_path}'...")
    try:
        def _reporthook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, int(block_num * block_size * 100 / total_size))
                sys.stdout.write(f"\rDownloading: {percent}% ({block_num * block_size // 1024} KB / {total_size // 1024} KB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(MODEL_URL, model_path, reporthook=_reporthook)
        print(f"\n[HandTracker] Download completed successfully: {model_path}")
    except Exception as e:
        print(f"\n[HandTracker] Failed to download model from {MODEL_URL}: {e}")
        raise

    return model_path


if __name__ == "__main__":
    ensure_model_downloaded()
