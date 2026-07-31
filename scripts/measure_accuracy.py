"""Measure how well this actually reads lips, against known sentences.

Everything else in this project tests plumbing. This tests the only question
that decides whether the tool is worth having: given real footage of a real
person really speaking, how much of it comes back?

Uses CREMA-D — 7,442 clips of 91 actors speaking twelve fixed sentences, frontal,
evenly lit, one speaker per clip. The sentence is encoded in the filename, so the
ground truth is exact rather than approximate. Conditions are close to ideal,
which makes the resulting number an optimistic ceiling: real-world footage will
be worse, not better.

CREMA-D is released under the Open Database License:
https://github.com/CheyneyComputerScience/CREMA-D

Clips are downloaded at run time, never committed.
"""

from __future__ import annotations

import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MEDIA = (
    "https://media.githubusercontent.com/media/CheyneyComputerScience/"
    "CREMA-D/master/VideoFlash"
)

# The twelve sentences, keyed by the code in each filename.
SENTENCES = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be cold",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before",
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a couple of minutes",
}

# Neutral delivery only: acted emotion distorts mouth shape in ways that would
# confound the measurement rather than inform it.
CLIPS = [
    "1001_DFA_NEU_XX", "1001_IEO_NEU_XX", "1003_IOM_NEU_XX", "1004_TAI_NEU_XX",
    "1005_WSI_NEU_XX", "1006_ITS_NEU_XX", "1007_TSI_NEU_XX", "1008_IWW_NEU_XX",
    "1009_MTI_NEU_XX", "1010_IWL_NEU_XX", "1011_ITH_NEU_XX", "1012_TIE_NEU_XX",
    "1013_DFA_NEU_XX", "1014_IEO_NEU_XX", "1015_IOM_NEU_XX", "1016_TAI_NEU_XX",
]


def fetch(name: str, into: Path) -> Path | None:
    target = into / f"{name}.flv"
    try:
        urllib.request.urlretrieve(f"{MEDIA}/{name}.flv", target)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        return None
    # A missing LFS object comes back as a tiny pointer file, not a video.
    return target if target.stat().st_size > 20_000 else None


def main() -> int:
    from lipsync import prepare
    from lipsync.backend import AutoAVSRRecognizer
    from lipsync.quality import Verdict
    from lipsync.recognize import download_weights, to_backend_tensor, weights_present
    from lipsync.score import Score, score

    workdir = Path(tempfile.mkdtemp(prefix="accuracy-"))

    if not weights_present():
        print("Downloading weights ...", flush=True)
        download_weights(progress=False)

    print("Loading recogniser ...", flush=True)
    recogniser = AutoAVSRRecognizer()

    print(f"\n{'clip':<20}{'expected':<40}{'heard':<40}{'WER':>7}  quality")
    print("-" * 115)

    subs = dels = ins = ref_words = 0
    used = skipped = 0

    for name in CLIPS:
        code = name.split("_")[1]
        expected = SENTENCES.get(code)
        if expected is None:
            continue

        clip = fetch(name, workdir)
        if clip is None:
            print(f"{name:<20}{'(download failed, skipped)':<40}")
            continue

        try:
            prepared = prepare(str(clip))
        except Exception as exc:  # noqa: BLE001
            print(f"{name:<20}{'(' + type(exc).__name__ + ', skipped)':<40}")
            continue

        if prepared.quality.verdict is Verdict.UNUSABLE:
            reasons = "; ".join(c.name for c in prepared.quality.problems)
            print(f"{name:<20}{expected[:38]:<40}{'— refused by the gate':<40}{'':>7}  {reasons}")
            skipped += 1
            continue

        heard = recogniser.transcribe(to_backend_tensor(prepared.mouth_rois))
        result = score(expected, heard)

        subs += result.substitutions
        dels += result.deletions
        ins += result.insertions
        ref_words += result.reference_words
        used += 1

        print(
            f"{name:<20}{expected[:38]:<40}{heard[:38]:<40}"
            f"{result.wer:>6.0%}  {prepared.quality.verdict.value}"
        )

    print("-" * 115)
    if used == 0:
        print("\nNo clips could be measured. Nothing can be concluded from this run.")
        return 1

    overall = Score(subs, dels, ins, ref_words)
    print(f"\nMeasured on {used} clips ({skipped} refused by the quality gate)")
    print(f"OVERALL: {overall.summary()}")
    print(f"         {overall.accuracy:.0%} of words recovered")

    print(
        "\nCREMA-D is frontal, evenly lit, one speaker, clearly enunciated —\n"
        "close to best case. Real footage will do worse than this, not better.\n"
        "For reference, the published figure for this checkpoint is 19% WER on\n"
        "LRS3, and human lip readers manage roughly 30-45% of English."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
