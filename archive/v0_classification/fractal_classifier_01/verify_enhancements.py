#!/usr/bin/env python3
"""
Verification script for Fractal Classifier v0.1 enhancements.

This script tests that all new features can be imported and are functional.
Run after installing the package: pip install -e .
"""

def test_imports():
    """Test that all new modules can be imported."""
    print("Testing imports...")
    
    try:
        # Core imports
        from fractal_classifier import __version__
        print(f"✓ Package version: {__version__}")
        
        # Kernel utilities
        from fractal_classifier.utils import (
            extract_conv_kernels,
            save_kernels_hdf5,
            load_kernels_hdf5,
            compute_kernel_statistics,
            compute_kernel_similarity,
            export_model_kernels
        )
        print("✓ Kernel utilities imported")
        
        # Training callbacks
        from fractal_classifier.training import (
            KernelExportCallback,
            KernelStatisticsCallback
        )
        print("✓ Training callbacks imported")
        
        # Analysis utilities
        from fractal_classifier.analysis import (
            plot_kernel_convergence_trajectory,
            compare_kernels_with_reference,
            analyze_kernel_convergence
        )
        print("✓ Analysis utilities imported")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        return False


def test_config():
    """Test that enhanced configuration can be loaded."""
    print("\nTesting configuration...")
    
    try:
        from fractal_classifier import Config
        from pathlib import Path
        
        # Try loading default config
        config_path = Path(__file__).parent / 'config' / 'default_config.yaml'
        config = Config.from_yaml(str(config_path))
        
        # Check for new configuration sections
        assert hasattr(config.training, 'regularization'), "Missing regularization config"
        assert hasattr(config.training, 'batch_norm'), "Missing batch_norm config"
        assert hasattr(config.training, 'lr_schedule'), "Missing lr_schedule config"
        assert hasattr(config.training.callbacks, 'tensorboard'), "Missing tensorboard config"
        assert hasattr(config.training.callbacks, 'kernel_export'), "Missing kernel_export config"
        assert hasattr(config.visualization, 'kernel_analysis'), "Missing kernel_analysis config"
        
        print("✓ Default configuration loaded")
        print(f"✓ New regularization options: {list(config.training.regularization.keys())}")
        print(f"✓ Kernel export enabled: {config.training.callbacks.kernel_export.enabled}")
        
        print("\n✅ Configuration test successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Configuration test failed: {e}")
        return False


def test_kernel_utilities():
    """Test basic kernel utility functions."""
    print("\nTesting kernel utilities...")
    
    try:
        import numpy as np
        import tempfile
        from pathlib import Path
        from fractal_classifier.utils.kernels import (
            save_kernels_hdf5,
            load_kernels_hdf5,
            compute_kernel_statistics
        )
        
        # Create dummy kernels
        dummy_kernels = {
            'conv2d': np.random.randn(3, 3, 1, 32).astype(np.float32),
            'conv2d_1': np.random.randn(3, 3, 32, 64).astype(np.float32)
        }
        
        # Test save/load
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test_kernels.h5'
            
            # Save
            save_kernels_hdf5(
                dummy_kernels,
                filepath,
                metadata={'epoch': 1, 'test': True}
            )
            print(f"✓ Saved kernels to {filepath}")
            
            # Load
            loaded_kernels, metadata = load_kernels_hdf5(filepath)
            print(f"✓ Loaded kernels: {list(loaded_kernels.keys())}")
            print(f"✓ Metadata: {metadata}")
            
            # Verify
            assert set(loaded_kernels.keys()) == set(dummy_kernels.keys())
            for key in dummy_kernels:
                assert np.allclose(dummy_kernels[key], loaded_kernels[key])
            print("✓ Saved and loaded kernels match")
            
            # Compute statistics
            stats = compute_kernel_statistics(loaded_kernels)
            print(f"✓ Computed statistics for {len(stats)} layers")
            
        print("\n✅ Kernel utilities test successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Kernel utilities test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_callbacks():
    """Test that callbacks can be instantiated."""
    print("\nTesting callbacks...")
    
    try:
        import tempfile
        from pathlib import Path
        from fractal_classifier.training.callbacks import (
            KernelExportCallback,
            KernelStatisticsCallback
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test KernelExportCallback
            callback1 = KernelExportCallback(
                output_dir=tmpdir,
                frequency='epoch',
                format='hdf5',
                verbose=0
            )
            print("✓ KernelExportCallback instantiated")
            
            # Test KernelStatisticsCallback
            log_file = Path(tmpdir) / 'kernel_stats.csv'
            callback2 = KernelStatisticsCallback(
                log_file=log_file,
                frequency=1,
                verbose=0
            )
            print("✓ KernelStatisticsCallback instantiated")
        
        print("\n✅ Callbacks test successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Callbacks test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    print("="*60)
    print("Fractal Classifier v0.1 - Enhancement Verification")
    print("="*60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("Kernel Utilities", test_kernel_utilities()))
    results.append(("Callbacks", test_callbacks()))
    
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:.<40} {status}")
    
    print("="*60)
    
    if all(passed for _, passed in results):
        print("\n🎉 All verification tests passed!")
        print("\nYou can now:")
        print("1. Train with kernel export: see examples/advanced_kernel_tracking.yaml")
        print("2. Analyze convergence: see KERNEL_ANALYSIS_GUIDE.md")
        print("3. Compare with reference kernels: see documentation")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
