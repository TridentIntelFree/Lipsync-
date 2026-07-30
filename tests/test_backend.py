"""The contract between preprocessing and the model.

These are the tests that catch the failure this project most wants to avoid: a
tensor that the model accepts but that means something different from what it
was trained on. They run without checkpoints — shapes and wiring are checkable
even when the weights are not available.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="recognition backend not installed")

from lipsync.backend import _use_vendored_espnet  # noqa: E402
from lipsync.constants import NETWORK_CROP  # noqa: E402
from lipsync.pipeline import to_model_tensor  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def vendored():
    _use_vendored_espnet()


def test_espnet_resolves_to_the_vendored_copy():
    """A pip-installed espnet must not win: it has no conv3d front end."""
    import espnet.nets.pytorch_backend.e2e_asr_transformer as module

    assert "lipsync/vendor" in module.__file__.replace("\\", "/")


def test_the_vendored_encoder_offers_the_conv3d_input_layer():
    import inspect
    import re

    from espnet.nets.pytorch_backend.transformer.encoder import Encoder

    options = set(re.findall(r'input_layer == "(\w+)"', inspect.getsource(Encoder.__init__)))
    assert "conv3d" in options, "this build cannot load the visual checkpoints"


def test_every_import_the_recognizer_needs_resolves():
    from espnet.asr.asr_utils import (  # noqa: F401
        add_results_to_json,
        get_model_conf,
        torch_load,
    )
    from espnet.nets.batch_beam_search import BatchBeamSearch  # noqa: F401
    from espnet.nets.lm_interface import dynamic_import_lm  # noqa: F401
    from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E  # noqa: F401
    from espnet.nets.scorers.length_bonus import LengthBonus  # noqa: F401


def test_visual_frontend_accepts_our_tensor_and_preserves_time():
    """(B, C, T, 88, 88) in, one feature vector per frame out."""
    from espnet.nets.pytorch_backend.backbones.conv3d_extractor import Conv3dResNet

    rois = np.random.randint(0, 255, (12, 96, 96), dtype=np.uint8)
    tensor = torch.from_numpy(to_model_tensor(rois))
    assert tensor.shape == (1, 12, NETWORK_CROP, NETWORK_CROP)

    frontend = Conv3dResNet(relu_type="swish").eval()
    with torch.no_grad():
        # E2E.encode() adds the batch dimension itself; mirror that here.
        features = frontend(tensor.unsqueeze(0))

    assert features.shape[:2] == (1, 12), "one feature vector per input frame"
    assert features.shape[2] == 512


def test_full_encoder_stack_preserves_time_steps():
    from espnet.nets.pytorch_backend.transformer.encoder import Encoder

    rois = np.random.randint(0, 255, (9, 96, 96), dtype=np.uint8)
    tensor = torch.from_numpy(to_model_tensor(rois))

    encoder = Encoder(
        idim=-1,
        attention_dim=256,
        attention_heads=4,
        linear_units=256,
        num_blocks=1,
        input_layer="conv3d",
        macaron_style=True,
        encoder_attn_layer_type="rel_mha",
        use_cnn_module=True,
        cnn_module_kernel=31,
    ).eval()

    with torch.no_grad():
        output, _ = encoder(tensor.unsqueeze(0), None)

    assert output.shape[0] == 1
    assert output.shape[1] == 9, "encoder must emit one step per video frame"


def test_compat_shims_behave_like_distutils():
    from lipsync.vendor._compat import LooseVersion, strtobool

    assert strtobool("yes") == 1 and strtobool("off") == 0
    with pytest.raises(ValueError):
        strtobool("perhaps")

    # The comparisons the vendored code actually performs against torch versions.
    assert LooseVersion("2.1.0+cu121") >= LooseVersion("1.7.0")
    assert LooseVersion("1.2.0") >= LooseVersion("1.2")
    assert LooseVersion("1.3") > LooseVersion("1.2.9")
    assert not LooseVersion("1.1.0") >= LooseVersion("1.2.0")


def test_vendored_tree_does_not_import_distutils():
    """Guards Python 3.12+ compatibility, where distutils no longer exists."""
    import pathlib

    vendor = pathlib.Path(__file__).parent.parent / "lipsync" / "vendor"
    offenders = [
        str(p.relative_to(vendor))
        for p in vendor.rglob("*.py")
        if "import distutils" in p.read_text(encoding="utf-8", errors="ignore")
        or "from distutils" in p.read_text(encoding="utf-8", errors="ignore")
    ]
    assert offenders == []
