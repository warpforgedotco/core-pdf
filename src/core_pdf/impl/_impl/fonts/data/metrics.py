"""Application aliases for the fourteen PDF standard font metric records."""

from core_pdf_spec.s_09_fonts.data.core14 import FONT_DATA as PDF_FONT_DATA
from core_pdf_spec.s_09_fonts.data.core14 import Core14FontMetrics

# Every base font name that resolves to a Standard-14 metric record. Arial,
# CourierNew and TimesNewRoman are metric clones of Helvetica, Courier and
# Times respectively -- that mapping is what lets a PDF naming Arial lay out
# with Helvetica's advances, as an application font-substitution policy.
internal_METRIC_RECORD_NAMES: dict[str, str] = {
    "Arial": "Helvetica",
    "Arial,Bold": "Helvetica-Bold",
    "Arial,BoldItalic": "Helvetica-BoldOblique",
    "Arial,Italic": "Helvetica-Oblique",
    "Courier": "Courier",
    "Courier-Bold": "Courier-Bold",
    "Courier-BoldOblique": "Courier-BoldOblique",
    "Courier-Oblique": "Courier-Oblique",
    "CourierNew": "Courier",
    "CourierNew,Bold": "Courier-Bold",
    "CourierNew,BoldItalic": "Courier-BoldOblique",
    "CourierNew,Italic": "Courier-Oblique",
    "Helvetica": "Helvetica",
    "Helvetica-Bold": "Helvetica-Bold",
    "Helvetica-BoldOblique": "Helvetica-BoldOblique",
    "Helvetica-Oblique": "Helvetica-Oblique",
    "Symbol": "Symbol",
    "Times-Bold": "Times-Bold",
    "Times-BoldItalic": "Times-BoldItalic",
    "Times-Italic": "Times-Italic",
    "Times-Roman": "Times-Roman",
    "TimesNewRoman": "Times-Roman",
    "TimesNewRoman,Bold": "Times-Bold",
    "TimesNewRoman,BoldItalic": "Times-BoldItalic",
    "TimesNewRoman,Italic": "Times-Italic",
    "ZapfDingbats": "ZapfDingbats",
}

FONT_DATA: dict[str, Core14FontMetrics] = {
    name: PDF_FONT_DATA[record] for name, record in internal_METRIC_RECORD_NAMES.items()
}
