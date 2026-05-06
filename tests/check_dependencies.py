#!/usr/bin/env python3
"""
Check if all required dependencies for dataset_generator and tests are available
"""

import sys

def check_dependencies():
    """Check if all required modules are available"""
    
    required_modules = {
        'numpy': 'NumPy (numerical computing)',
        'h5py': 'HDF5 for Python (data storage)',
        'psutil': 'Process and system utilities',
        'configparser': 'Configuration file parser (standard library)',
        'pyFC': 'Fractal Cube library (in pyFC_lib/)',
    }
    
    missing = []
    available = []
    
    print("Checking dependencies...")
    print("=" * 60)
    
    for module, description in required_modules.items():
        try:
            __import__(module)
            available.append(f"✓ {module:15s} - {description}")
        except ImportError:
            missing.append(f"✗ {module:15s} - {description}")
    
    # Print results
    if available:
        print("\nAvailable:")
        for item in available:
            print(f"  {item}")
    
    if missing:
        print("\nMissing:")
        for item in missing:
            print(f"  {item}")
        
        print("\n" + "=" * 60)
        print("Installation instructions:")
        print("=" * 60)
        
        if any('h5py' in m for m in missing):
            print("\nInstall h5py:")
            print("  pip install h5py")
        
        if any('psutil' in m for m in missing):
            print("\nInstall psutil:")
            print("  pip install psutil")
        
        if any('pyFC' in m for m in missing):
            print("\nInstall pyFC:")
            print("  cd pyFC_lib/")
            print("  pip install -e .")
        
        return False
    
    print("\n" + "=" * 60)
    print("✓ All dependencies are available!")
    print("=" * 60)
    return True


if __name__ == '__main__':
    success = check_dependencies()
    sys.exit(0 if success else 1)
