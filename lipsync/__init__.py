"""Infer speech from silent video of a speaking face."""

from .pipeline import PreparedVideo, prepare
from .quality import QualityReport, Verdict

__version__ = "0.1.0"

__all__ = ["prepare", "PreparedVideo", "QualityReport", "Verdict", "__version__"]
