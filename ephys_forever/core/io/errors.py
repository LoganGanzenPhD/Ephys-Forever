"""Exceptions raised by file readers."""


class ReaderError(ValueError):
    """A file could not be read as the format its reader expects."""


class UnsupportedFeatureError(ReaderError):
    """The file is valid but uses a feature this reader does not implement yet."""
