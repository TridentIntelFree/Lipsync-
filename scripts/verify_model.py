"""Run the whole pipeline, weights included, and report what happened.

This exists because the environment the project was written in cannot reach
huggingface.co, so the final step — turning mouth crops into words — was never
executed there. CI can reach it, so CI is where that claim gets tested.

It checks that the model loads, that a real video flows through it end to end,
and that a transcript comes out. It does **not** check accuracy: the test clip is
a still photograph with synthetic motion, so whatever text appears is meaningless
as a reading. Plumbing is the point.

Exits non-zero on failure, so the workflow goes red rather than quietly passing.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Runnable straight from a checkout, without needing `pip install -e .` first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# A public test asset from MediaPipe's own repository, fetched rather than
# committed so no face image lives in this project.
FACE_URL = (
    "https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/"
    "mediapipe/objc/testdata/sergey.png"
)


def step(message: str) -> None:
    print(f"\n=== {message}", flush=True)


def make_clip(destination: Path, seconds: float = 2.0, fps: int = 25) -> Path:
    """Build a short clip: a real face, moved and rotated so alignment has work to do."""
    import cv2
    import imageio_ffmpeg
    import urllib.request

    source = destination.parent / "face.png"
    urllib.request.urlretrieve(FACE_URL, source)
    face = cv2.imread(str(source))
    if face is None:
        raise SystemExit(f"Could not read the test image from {FACE_URL}")

    width, height = 640, 480
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(destination)],
        stdin=subprocess.PIPE,
    )
    for index in range(int(seconds * fps)):
        t = index / fps
        matrix = cv2.getRotationMatrix2D((300, 300), 4 * np.sin(t * 2), 1 + 0.03 * np.sin(t * 3))
        matrix[0, 2] += 12 * np.sin(t * 1.5)
        matrix[1, 2] += 8 * np.cos(t * 1.1)
        warped = cv2.warpAffine(face, matrix, (600, 600), borderMode=cv2.BORDER_REPLICATE)
        canvas = np.zeros((height, width, 3), np.uint8)
        canvas[20:460, 100:540] = cv2.resize(warped, (440, 440))
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed to write the test clip")
    return destination


def main() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="verify-"))

    step("Building a test clip")
    clip = make_clip(workdir / "clip.mp4")
    print(f"    {clip} ({clip.stat().st_size} bytes)")

    step("Downloading model weights from HuggingFace")
    started = time.time()
    from lipsync.recognize import download_weights, weights_present

    if weights_present():
        print("    already cached")
    else:
        for name, path in download_weights(progress=False).items():
            print(f"    {name}: {path.stat().st_size / 1e6:.0f} MB")
    print(f"    took {time.time() - started:.0f}s")

    step("Preprocessing: decode, detect, align, quality")
    import lipsync

    prepared = lipsync.prepare(str(clip))
    print(f"    {len(prepared.mouth_rois)} mouth crops, "
          f"detection rate {prepared.detection_rate:.0%}")
    print("    " + prepared.quality.summary().replace("\n", "\n    "))
    if len(prepared.mouth_rois) == 0:
        print("FAIL: no mouth crops produced")
        return 1

    step("Loading the recogniser")
    started = time.time()
    from lipsync.backend import AutoAVSRRecognizer

    recogniser = AutoAVSRRecognizer()
    print(f"    loaded in {time.time() - started:.0f}s on {recogniser.device}")
    print(f"    vocabulary: {len(recogniser.token_list)} tokens")

    step("Running recognition")
    started = time.time()
    from lipsync.recognize import to_backend_tensor

    tensor = to_backend_tensor(prepared.mouth_rois)
    print(f"    input tensor {tuple(tensor.shape)}")
    text = recogniser.transcribe(tensor)
    print(f"    took {time.time() - started:.0f}s")

    print("\n" + "=" * 62)
    print(f"TRANSCRIPT: {text!r}")
    print("=" * 62)
    print(
        "\nThe clip is a still photograph with synthetic motion, so the text\n"
        "above is not a reading of anything and its content means nothing.\n"
        "What this proves is that weights load and decoding runs end to end."
    )

    if not isinstance(text, str):
        print("\nFAIL: recogniser did not return a string")
        return 1

    print("\nPASS: the full pipeline executed, weights included.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
