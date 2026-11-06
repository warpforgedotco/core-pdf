from .pdfexceptions import PDFException
from .pdftypes import PDFObjRef


class PDFSyntaxError(PDFException):
    pass


__all__ = ("PDFObjRef", "PDFSyntaxError")
