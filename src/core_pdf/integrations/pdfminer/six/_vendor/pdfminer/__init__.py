# Keep this synchronized with the pinned release in ../vendor.txt. Vendored
# distributions do not retain their .dist-info metadata, so querying
# importlib.metadata would otherwise report an unrelated external installation
# or fall back to "0.0".
__version__ = "20260107"

if __name__ == "__main__":
    print(__version__)
