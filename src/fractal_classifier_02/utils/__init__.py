from .patches import extract_patches, aggregate_predictions, predict_image

try:
    from .gpu import setup_gpu
    __all__ = ['setup_gpu', 'extract_patches', 'aggregate_predictions', 'predict_image']
except ImportError:
    # TensorFlow not available (e.g. during data-only workflows)
    __all__ = ['extract_patches', 'aggregate_predictions', 'predict_image']

