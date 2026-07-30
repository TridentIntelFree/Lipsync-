"""CLI behaviour, especially the paths a user hits when something is wrong."""

from __future__ import annotations

import numpy as np
import pytest

from lipsync.cli import main
from lipsync.quality import Verdict
from tests.test_video import make_video


@pytest.fixture(scope="module")
def faceless_video(tmp_path_factory):
    """Noise: decodable, but containing no face."""
    return make_video(tmp_path_factory.mktemp("cli") / "noface.mp4", seconds=1)


def test_check_only_on_faceless_video_reports_rather_than_crashing(
    faceless_video, capsys
):
    """Regression: this used to raise DetectionError out of prepare()."""
    assert main([str(faceless_video), "--check-only"]) == 2
    out = capsys.readouterr().out
    assert "UNUSABLE" in out
    assert "no face found in any frame" in out


def test_transcribing_faceless_video_refuses(faceless_video, capsys):
    assert main([str(faceless_video)]) == 2
    assert "would be invention" in capsys.readouterr().err


def test_json_output_is_wellformed(faceless_video, capsys):
    import json

    assert main([str(faceless_video), "--check-only", "--json"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["quality"] == "unusable"
    assert payload["detection_rate"] == 0.0
    assert payload["checks"][0]["verdict"] == "unusable"


def test_missing_file_is_an_error(tmp_path):
    from lipsync.video import VideoError

    with pytest.raises(VideoError):
        main([str(tmp_path / "absent.mp4"), "--check-only"])


def test_prepare_on_faceless_video_yields_empty_rois(faceless_video):
    from lipsync.pipeline import prepare

    prepared = prepare(str(faceless_video))
    assert prepared.quality.verdict is Verdict.UNUSABLE
    assert prepared.mouth_rois.shape[0] == 0
    assert prepared.detection_rate == 0.0
    assert prepared.duration_seconds == 0.0


def test_save_rois_writes_a_readable_array(faceless_video, tmp_path):
    target = tmp_path / "rois.npy"
    main([str(faceless_video), "--check-only", "--save-rois", str(target)])
    assert np.load(target).shape[1:] == (96, 96)
