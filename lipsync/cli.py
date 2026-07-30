"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .pipeline import prepare
from .quality import Verdict


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lipsync",
        description="Infer speech from silent video of a speaking face.",
    )
    parser.add_argument("video", help="path to a video file")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="report input quality and stop, without running recognition",
    )
    parser.add_argument(
        "--max-seconds", type=float, default=None, help="only process the first N seconds"
    )
    parser.add_argument(
        "--allow-unusable",
        action="store_true",
        help="transcribe even when the input fails quality checks (output will be invention)",
    )
    parser.add_argument(
        "--save-rois",
        metavar="PATH",
        help="write the aligned mouth crops to an .npy file for inspection",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--version", action="version", version=f"lipsync {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    prepared = prepare(args.video, max_seconds=args.max_seconds)

    if args.save_rois:
        import numpy as np

        np.save(args.save_rois, prepared.mouth_rois)

    payload: dict = {
        "video": str(prepared.info.path),
        "resolution": f"{prepared.info.width}x{prepared.info.height}",
        "source_fps": prepared.info.fps,
        "frames_analysed": len(prepared.mouth_rois),
        "duration_seconds": round(prepared.duration_seconds, 2),
        "detection_rate": round(prepared.detection_rate, 3),
        "quality": prepared.quality.verdict.value,
        "checks": [
            {"name": c.name, "verdict": c.verdict.value, "message": c.message}
            for c in prepared.quality.checks
        ],
    }

    if args.check_only:
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(
                f"{prepared.info.width}x{prepared.info.height} at "
                f"{prepared.info.fps:g} fps, {len(prepared.mouth_rois)} frames "
                f"({prepared.duration_seconds:.1f}s)\n"
            )
            print(prepared.quality.summary())
        return 0 if prepared.quality.verdict is not Verdict.UNUSABLE else 2

    from .recognize import BackendMissing, recognize

    try:
        transcript = recognize(prepared, allow_unusable=args.allow_unusable)
    except BackendMissing as exc:
        print(f"{exc}", file=sys.stderr)
        return 3
    except ValueError as exc:
        print(f"{exc}", file=sys.stderr)
        print("\n" + prepared.quality.summary(), file=sys.stderr)
        return 2

    payload["transcript"] = transcript.text
    payload["caveat"] = transcript.caveat()

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(prepared.quality.summary())
        print(f"\nTranscript:\n  {transcript.text or '(nothing recognised)'}")
        print(f"\n{transcript.caveat()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
