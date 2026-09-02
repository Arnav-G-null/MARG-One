"""
Model downloader and asset management for MARG-One Vision subsystem.
"""

import os
import urllib.request
import sys

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_MODEL_PATH = os.path.join(_PROJECT_ROOT, "hand_landmarker.task")


def ensure_model_downloaded(model_path: str = DEFAULT_MODEL_PATH) -> str:
    """
    Verifies that the MediaPipe Hand Landmarker task asset exists.
    Downloads the official model binary from Google Cloud Storage if absent.
    """
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        return model_path

    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    print(f"[MARG-One:Vision] Downloading Hand Landmarker asset to '{model_path}'...")
    try:
        def _reporthook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, int(block_num * block_size * 100 / total_size))
                sys.stdout.write(f"\rDownloading model: {percent}% ({block_num * block_size // 1024} KB / {total_size // 1024} KB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(MODEL_URL, model_path, reporthook=_reporthook)
        print(f"\n[MARG-One:Vision] Download verified: {model_path}")
    except Exception as exc:
        print(f"\n[MARG-One:Vision] Error downloading model from {MODEL_URL}: {exc}")
        raise

    return model_path


if __name__ == "__main__":
    ensure_model_downloaded()
