class PdfminerException(Exception):
    pass


class MalformedPDFException(Exception):
    pass


__all__ = ("MalformedPDFException", "PdfminerException")
