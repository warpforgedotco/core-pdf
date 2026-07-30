import sys
import logging

sys.modules.setdefault("fontTools", sys.modules[__name__])

from .misc.loggingTools import configLogger

log = logging.getLogger(__name__)

version = __version__ = "4.63.0"

__all__ = ["version", "log", "configLogger"]
