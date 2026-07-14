import logging

from core_pdf.impl.third_party.fontTools.misc.loggingTools import configLogger

log = logging.getLogger(__name__)

version = __version__ = "4.63.0"

__all__ = ["version", "log", "configLogger"]
