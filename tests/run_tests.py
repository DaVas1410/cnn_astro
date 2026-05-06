#!/usr/bin/env python3
"""
Quick test runner script for dataset generator

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py -v          # Verbose mode
    python run_tests.py TestConfigParsing  # Run specific test class
"""

import sys
import os
import unittest
import argparse

# Ensure test dir and parent dir are in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Import test module
from test_dataset_generator import (
    TestConfigParsing,
    TestParameterCombinations,
    TestSingleImageGeneration,
    TestWorkerFunction,
    TestHDF5Operations,
    TestDatasetPopulation,
    TestUtilityFunctions,
    TestMemoryManagement,
    TestMergeAppend,
    run_test_suite
)


def main():
    parser = argparse.ArgumentParser(description='Run dataset generator tests')
    parser.add_argument('test_class', nargs='?', default=None,
                       help='Specific test class to run (optional)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('-f', '--failfast', action='store_true',
                       help='Stop on first failure')
    
    args = parser.parse_args()
    
    # Determine verbosity
    verbosity = 2 if args.verbose else 1
    
    if args.test_class:
        # Run specific test class
        try:
            test_class = globals()[args.test_class]
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
            runner = unittest.TextTestRunner(verbosity=verbosity, failfast=args.failfast)
            result = runner.run(suite)
        except KeyError:
            print(f"Error: Test class '{args.test_class}' not found")
            print("\nAvailable test classes:")
            print("  - TestConfigParsing")
            print("  - TestParameterCombinations")
            print("  - TestSingleImageGeneration")
            print("  - TestWorkerFunction")
            print("  - TestHDF5Operations")
            print("  - TestDatasetPopulation")
            print("  - TestUtilityFunctions")
            print("  - TestMemoryManagement")
            print("  - TestMergeAppend")
            sys.exit(1)
    else:
        # Run all tests
        result = run_test_suite()
    
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
