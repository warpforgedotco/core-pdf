"""Public package for the vendored pdfminer.six compatibility implementation."""

from pathlib import Path

from core_pdf.integrations.pdfminer.six._vendor.pdfminer import __version__

# Make the immutable upstream package available through the public integration
# namespace without copying wrapper modules for every pdfminer submodule.
__path__.append(str(Path(__file__).parent / "_vendor" / "pdfminer"))

__all__ = ("__version__",)
