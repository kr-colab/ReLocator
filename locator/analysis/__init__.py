"""Analysis subpackage for locator."""

from ._helpers import HelpersMixin
from .holdout import HoldoutMixin
from .resampling import ResamplingMixin
from .windowed import WindowedMixin


class AnalysisMixin(HelpersMixin, ResamplingMixin, HoldoutMixin, WindowedMixin):
    """Mixin class providing analysis functionality for Locator."""
