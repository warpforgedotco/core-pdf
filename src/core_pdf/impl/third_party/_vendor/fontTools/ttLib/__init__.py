"""fontTools.ttLib -- a package for dealing with TrueType fonts."""

from core_pdf.impl.third_party._vendor.fontTools.config import OPTIONS
from core_pdf.impl.third_party._vendor.fontTools.misc.loggingTools import deprecateFunction
import logging


log = logging.getLogger(__name__)


OPTIMIZE_FONT_SPEED = OPTIONS["fontTools.ttLib:OPTIMIZE_FONT_SPEED"]


class TTLibError(Exception):
    pass


class TTLibFileIsCollectionError(TTLibError):
    pass


@deprecateFunction("use logging instead", category=DeprecationWarning)
def debugmsg(msg):
    import time

    print(msg + time.strftime("  (%H:%M:%S)", time.localtime(time.time())))


from core_pdf.impl.third_party._vendor.fontTools.ttLib.ttFont import *
from core_pdf.impl.third_party._vendor.fontTools.ttLib.ttCollection import TTCollection
