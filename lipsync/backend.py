"""Auto-AVSR checkpoint loading and joint CTC/attention beam search.

A thin, faithful wrapper over the released ESPnet checkpoints. The structure
here mirrors the reference implementation from Imperial College London
(Apache-2.0) because the checkpoint format, token vocabulary and decoder weights
are all fixed by how the model was trained.

Loading is expensive — roughly 900 MB of weights plus a language model — so a
recogniser is worth constructing once and reusing across videos.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .assets import cache_dir
from .recognize import BEAM_SIZE, CTC_WEIGHT, LENGTH_PENALTY, LM_WEIGHT, WEIGHTS

_TOKENS = Path(__file__).parent / "tokens" / "unigram5000_units.txt"
_VENDOR = Path(__file__).parent / "vendor"


def _use_vendored_espnet() -> None:
    """Put the vendored espnet subset ahead of anything installed.

    The released checkpoints need a 3D-convolution visual front end that stock
    espnet does not have — its encoder only offers audio input layers, so a
    pip-installed espnet cannot load these weights at all. The vendored subset
    is the modified build that can. See VENDOR.md.

    It is placed first on the path deliberately: if both are importable, ours
    has to win, or model construction fails on an unknown ``conv3d`` layer.
    """
    path = str(_VENDOR)
    if path not in sys.path:
        sys.path.insert(0, path)
    elif sys.path[0] != path:
        sys.path.remove(path)
        sys.path.insert(0, path)


def _load_token_list(train_args) -> list[str]:
    """Build the output vocabulary the checkpoint was trained against."""
    labels_type = getattr(train_args, "labels_type", "char")
    if labels_type == "char":
        return list(train_args.char_list)
    if labels_type == "unigram5000":
        units = _TOKENS.read_text(encoding="utf-8").splitlines()
        return ["<blank>"] + [line.split()[0] for line in units if line] + ["<eos>"]
    raise ValueError(f"Unsupported labels_type in checkpoint: {labels_type!r}")


class AutoAVSRRecognizer:
    """Visual speech recogniser over preprocessed mouth ROIs."""

    def __init__(self, device: str | None = None):
        _use_vendored_espnet()

        import torch

        # Beam search over a 5000-token vocabulary is the expensive part: minutes
        # on CPU, seconds on a GPU. Use one when it is there.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        from espnet.asr.asr_utils import get_model_conf, torch_load
        from espnet.nets.batch_beam_search import BatchBeamSearch
        from espnet.nets.lm_interface import dynamic_import_lm
        from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
        from espnet.nets.scorers.length_bonus import LengthBonus

        self._torch = torch
        self.device = device

        cache = cache_dir()
        model_path = cache / WEIGHTS["vsr_model"][1]
        config_path = cache / WEIGHTS["vsr_config"][1]
        lm_path = cache / WEIGHTS["lm_model"][1]
        lm_config_path = cache / WEIGHTS["lm_config"][1]

        missing = [p for p in (model_path, config_path, lm_path, lm_config_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing checkpoints: "
                + ", ".join(str(p) for p in missing)
                + "\nRun: python -m lipsync.recognize --download"
            )

        with config_path.open("rb") as handle:
            confs = json.load(handle)
        args = confs if isinstance(confs, dict) else confs[2]
        self.train_args = argparse.Namespace(**args)

        self.token_list = _load_token_list(self.train_args)
        odim = len(self.token_list)

        self.model = E2E(odim, self.train_args)
        self.model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=False)
        )
        self.model.to(device=device).eval()

        lm_args = get_model_conf(str(lm_path), str(lm_config_path))
        lm_class = dynamic_import_lm(
            getattr(lm_args, "model_module", "default"), lm_args.backend
        )
        language_model = lm_class(odim, lm_args)
        torch_load(str(lm_path), language_model)
        language_model.eval()

        scorers = self.model.scorers()
        scorers["lm"] = language_model
        scorers["length_bonus"] = LengthBonus(odim)

        self.beam_search = BatchBeamSearch(
            beam_size=BEAM_SIZE,
            vocab_size=odim,
            weights=dict(
                decoder=1.0 - CTC_WEIGHT,
                ctc=CTC_WEIGHT,
                lm=LM_WEIGHT,
                length_bonus=LENGTH_PENALTY,
            ),
            scorers=scorers,
            sos=odim - 1,
            eos=odim - 1,
            token_list=self.token_list,
            pre_beam_score_key=None if CTC_WEIGHT == 1.0 else "decoder",
        )
        self.beam_search.to(device=device).eval()

    def transcribe(self, tensor) -> str:
        """Decode a ``(1, T, 88, 88)`` normalised ROI tensor into text."""
        from espnet.asr.asr_utils import add_results_to_json

        with self._torch.no_grad():
            features = self.model.encode(tensor.to(self.device))
            hypotheses = self.beam_search(features)
            best = [h.asdict() for h in hypotheses[: min(len(hypotheses), 1)]]
            text = add_results_to_json(best, self.token_list)

        return text.replace("▁", " ").replace("<eos>", "").strip()
