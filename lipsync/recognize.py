"""Turn a mouth-ROI sequence into text.

The acoustic-free recogniser is the Auto-AVSR visual model from Imperial College
London: a 3D convolution and ResNet-18 front end, a Conformer encoder, and a
Transformer decoder, decoded with joint CTC/attention beam search against an
external subword language model. It reports 19.1% word error rate on LRS3.

This module deliberately does *not* reimplement that network. The weights are
ESPnet-format checkpoints, and a hand-rolled re-implementation that merely loads
without error would produce plausible-looking wrong text — the worst possible
failure for this project. Instead the reference implementation is imported, and
this module owns what it can own end to end: fetching and caching the weights,
converting our preprocessed tensor into the exact layout the model expects, and
attaching the input-quality caveats to whatever comes back.

Install the recogniser with::

    pip install -e '.[recognize]'
    python -m lipsync.recognize --install-backend

See docs/RECOGNIZER.md for what the backend is and why it is a separate step.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from .assets import cache_dir, fetch
from .quality import QualityReport, Verdict

# Ungated mirrors of the Auto-AVSR release. No account or token is needed; the
# upstream Google Drive links require a Google session, these do not.
WEIGHTS = {
    "vsr_model": (
        "https://huggingface.co/Amanvir/LRS3_V_WER19.1/resolve/main/model.pth",
        "LRS3_V_WER19.1/model.pth",
    ),
    "vsr_config": (
        "https://huggingface.co/Amanvir/LRS3_V_WER19.1/resolve/main/model.json",
        "LRS3_V_WER19.1/model.json",
    ),
    "lm_model": (
        "https://huggingface.co/Amanvir/lm_en_subword/resolve/main/model.pth",
        "lm_en_subword/model.pth",
    ),
    "lm_config": (
        "https://huggingface.co/Amanvir/lm_en_subword/resolve/main/model.json",
        "lm_en_subword/model.json",
    ),
}

# Decoding parameters from the reference configuration.
BEAM_SIZE = 40
CTC_WEIGHT = 0.1
LM_WEIGHT = 0.6
LENGTH_PENALTY = 0.0


class BackendMissing(RuntimeError):
    """Raised when the recognition backend is not installed."""


@dataclasses.dataclass
class Transcript:
    """A transcript, inseparable from how much the input justifies believing it."""

    text: str
    quality: QualityReport
    frames: int

    @property
    def trustworthy(self) -> bool:
        return self.quality.verdict is Verdict.GOOD

    def caveat(self) -> str:
        """A plain statement of how much weight this transcript can bear."""
        if self.quality.verdict is Verdict.UNUSABLE:
            return (
                "This transcript is not evidence of anything. The input failed "
                "quality checks, and the language model produces fluent English "
                "regardless of whether the video contained recoverable speech."
            )
        if self.quality.verdict is Verdict.MARGINAL:
            return (
                "Treat this as a hypothesis, not a reading. Input quality is "
                "below what the model was trained on, so expect substantially "
                "more than the benchmark 19% word error rate."
            )
        return (
            "Input quality is good. Even so, the benchmark error rate is roughly "
            "one word in five on clean frontal video, and homophenes (p/b/m, "
            "f/v, and others) are visually identical — the model is guessing "
            "between them from context."
        )


def download_weights(progress: bool = True) -> dict[str, Path]:
    """Fetch every checkpoint this recogniser needs, returning local paths."""
    paths: dict[str, Path] = {}
    for key, (url, name) in WEIGHTS.items():
        if progress:
            print(f"Fetching {name} ...")
        paths[key] = fetch(url, name, progress=progress)
    return paths


def weights_present() -> bool:
    """True when every checkpoint is already cached locally."""
    return all((cache_dir() / name).exists() for _, name in WEIGHTS.values())


def _require_backend():
    """Check the recognition backend can be imported, or say what is missing.

    The espnet subset ships with this package, so the only thing that can
    actually be absent is PyTorch.
    """
    from .backend import _use_vendored_espnet

    _use_vendored_espnet()
    try:
        import torch  # noqa: F401
        from espnet.asr.asr_utils import torch_load  # noqa: F401
        from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E  # noqa: F401
    except ImportError as exc:
        raise BackendMissing(
            f"The recognition backend is unavailable: {exc}\n\n"
            "Preprocessing (video -> aligned mouth ROIs) works without it; only "
            "the final video-to-text step needs it.\n\n"
            "  pip install -e '.[recognize]'\n\n"
            "See docs/RECOGNIZER.md for details and licence terms — the "
            "checkpoints are released for non-commercial use."
        ) from exc


def to_backend_tensor(mouth_rois: np.ndarray):
    """Convert ``(T, 96, 96)`` uint8 ROIs into the backend's ``(1, T, 88, 88)`` tensor."""
    import torch

    from .pipeline import to_model_tensor

    return torch.from_numpy(to_model_tensor(mouth_rois))


def recognize(prepared, allow_unusable: bool = False) -> Transcript:
    """Run visual speech recognition over a ``PreparedVideo``.

    Refuses unusable input by default: producing confident text from a video
    that cannot support it is the failure mode most likely to mislead someone.
    """
    if prepared.quality.verdict is Verdict.UNUSABLE and not allow_unusable:
        problems = "; ".join(c.message for c in prepared.quality.problems)
        raise ValueError(
            f"Input quality is unusable ({problems}). Any transcript would be "
            "invention. Pass allow_unusable=True to override."
        )

    _require_backend()
    download_weights(progress=False)

    from .backend import AutoAVSRRecognizer

    recognizer = AutoAVSRRecognizer()
    text = recognizer.transcribe(to_backend_tensor(prepared.mouth_rois))

    return Transcript(
        text=text, quality=prepared.quality, frames=len(prepared.mouth_rois)
    )


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Manage recogniser weights.")
    parser.add_argument(
        "--download", action="store_true", help="fetch checkpoints into the cache"
    )
    parser.add_argument(
        "--check", action="store_true", help="report whether weights and backend are ready"
    )
    args = parser.parse_args()

    if args.download:
        for key, path in download_weights().items():
            print(f"  {key}: {path}")
        return 0

    if args.check:
        print(f"cache:   {cache_dir()}")
        print(f"weights: {'present' if weights_present() else 'not downloaded'}")
        try:
            _require_backend()
            print("backend: installed")
        except BackendMissing:
            print("backend: not installed")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
