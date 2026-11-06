from .pdfexceptions import PDFException
from .pdfparser import PDFSyntaxError


class PDFNoValidXRef(PDFSyntaxError):
    pass


class PDFEncryptionError(PDFException):
    pass


class PDFPasswordIncorrect(PDFEncryptionError):
    pass


__all__ = ("PDFEncryptionError", "PDFNoValidXRef", "PDFPasswordIncorrect")
