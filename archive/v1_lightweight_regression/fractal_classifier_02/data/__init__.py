from .loader import DataIndexer

try:
    from .pipeline import DataPipeline, _extract_patches
    __all__ = ['DataIndexer', 'DataPipeline', '_extract_patches']
except ImportError:
    __all__ = ['DataIndexer']

