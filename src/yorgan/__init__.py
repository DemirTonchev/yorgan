import importlib.metadata

try:
    __version__ = importlib.metadata.version(__package__)
except importlib.metadata.PackageNotFoundError:
    # This can happen if the package is not installed (e.g., during development
    # or if running directly without proper installation).
    __version__ = "0.0.0+dev"
