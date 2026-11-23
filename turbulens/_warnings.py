"""Warning categories, kept free of heavy imports so the CLI can refer to them without loading PyTorch."""


class ClippedPixelsWarning(UserWarning):
    """Many input pixels hit the normalization clip boundary: the input may be out of the training distribution."""
