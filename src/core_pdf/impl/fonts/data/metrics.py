from core_adobe_fonts.afm.core14 import FONT_DATA as PDF_FONT_DATA
from core_adobe_fonts.afm.core14 import Core14FontMetrics

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
