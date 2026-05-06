"""
Patch extraction and aggregation utilities for large-image inference.

These functions are used at inference time when a 256×256 or 512×512
image must be processed by the 128×128 native-resolution model.
"""

import numpy as np


def extract_patches(image: np.ndarray, patch_size: int = 128) -> np.ndarray:
    """
    Slice a 2-D (H, W) image into non-overlapping *patch_size × patch_size* tiles.

    If H or W are not exact multiples of *patch_size*, the image is
    centre-cropped before slicing.

    Args:
        image:      2-D numpy array of shape (H, W).
        patch_size: Side length of each output tile in pixels.

    Returns:
        np.ndarray of shape (n_patches, patch_size, patch_size).

    Examples:
        >>> img = np.zeros((512, 512), dtype=np.float32)
        >>> patches = extract_patches(img, patch_size=128)
        >>> patches.shape
        (16, 128, 128)
    """
    h, w = image.shape[:2]
    h_crop = (h // patch_size) * patch_size
    w_crop = (w // patch_size) * patch_size
    r0 = (h - h_crop) // 2
    c0 = (w - w_crop) // 2
    image = image[r0:r0 + h_crop, c0:c0 + w_crop]

    n_rows = h_crop // patch_size
    n_cols = w_crop // patch_size

    patches = (
        image
        .reshape(n_rows, patch_size, n_cols, patch_size)
        .transpose(0, 2, 1, 3)
        .reshape(-1, patch_size, patch_size)
    )
    return patches.astype(np.float32)


def aggregate_predictions(predictions: np.ndarray) -> float:
    """
    Aggregate per-patch k_min predictions into a single image-level estimate.

    Current strategy: arithmetic mean.  Override this function to implement
    alternatives (median, weighted mean, etc.).

    Args:
        predictions: 1-D array of per-patch predictions.

    Returns:
        Scalar float representing the aggregated k_min estimate.
    """
    return float(np.mean(predictions))


def predict_image(
    model,
    image: np.ndarray,
    patch_size: int = 128,
) -> float:
    """
    Predict k_min for a single 2-D image of any supported resolution.

    Args:
        model:      A trained Keras model with input shape (H, W, 1).
        image:      2-D float32 numpy array (H, W).
        patch_size: Native model input size.

    Returns:
        Predicted k_min as a scalar float.
    """
    patches = extract_patches(image, patch_size)          # (N, P, P)
    patches_4d = patches[:, :, :, np.newaxis]             # (N, P, P, 1)
    preds = model.predict(patches_4d, verbose=0).flatten()
    return aggregate_predictions(preds)
