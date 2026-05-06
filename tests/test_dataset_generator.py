"""
Comprehensive Test Suite for Dataset Generator

Tests all functionalities of the DatasetGen class including:
- Configuration loading and parsing
- Parameter combination generation
- Single image generation
- HDF5 file operations
- Batch processing
- Validation and error handling
- k_min and k_max value parsing

Author: Test Suite
Date: February 2026
"""

import unittest
import os
import sys
import tempfile
import shutil
import numpy as np
import h5py
from pathlib import Path
import configparser

# Import the module to test
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dataset_generator import DatasetGen, _generate_image_worker, validation
import dataset_generator as dg


class TestConfigParsing(unittest.TestCase):
    """Test configuration file parsing and validation"""
    
    def setUp(self):
        """Create temporary directory for test files"""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'test_config.ini')
        
    def tearDown(self):
        """Clean up temporary files"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_minimal_config(self, **kwargs):
        """Create a minimal valid configuration file with optional overrides"""
        config = configparser.ConfigParser()
        
        config['dataset'] = {
            'output_file': kwargs.get('output_file', 'test.h5'),
            'batch_size': str(kwargs.get('batch_size', 2)),
            'num_images': str(kwargs.get('num_images', 4)),
            'image_width': str(kwargs.get('image_width', 64)),
            'image_height': str(kwargs.get('image_height', 64)),
            'image_depth': str(kwargs.get('image_depth', 1)),
            'k_values': kwargs.get('k_values', '2,4'),
        }
        
        # Add k_max_values if provided
        if 'k_max_values' in kwargs:
            config['dataset']['k_max_values'] = kwargs['k_max_values']
        
        config['distribution'] = {
            'mean': str(kwargs.get('mean', 1.0)),
            'sigma': str(kwargs.get('sigma', 2.0)),
            'beta': str(kwargs.get('beta', -1.5)),
        }
        
        config['fractal'] = {
            'ni': str(kwargs.get('ni', 64)),
            'nj': str(kwargs.get('nj', 64)),
            'nk': str(kwargs.get('nk', 1)),
            'verbose': str(kwargs.get('verbose', False)),
            'random_seed': str(kwargs.get('random_seed', True)),
        }
        
        config['hdf5'] = {
            'compression': kwargs.get('compression', 'gzip'),
            'compression_opts': str(kwargs.get('compression_opts', 1)),
            'chunks_auto': str(kwargs.get('chunks_auto', True)),
            'chunk_size_images': kwargs.get('chunk_size_images', 'auto'),
            'chunk_size_labels': kwargs.get('chunk_size_labels', 'auto'),
        }
        
        config['processing'] = {
            'memory_monitoring': str(kwargs.get('memory_monitoring', True)),
            'progress_reporting': str(kwargs.get('progress_reporting', False)),
            'progress_report_interval': str(kwargs.get('progress_report_interval', 1)),
            'clear_memory_after_batch': str(kwargs.get('clear_memory_after_batch', True)),
            'force_garbage_collection': str(kwargs.get('force_garbage_collection', True)),
            'num_workers': str(kwargs.get('num_workers', 1)),
        }
        
        config['metadata'] = {
            'method': kwargs.get('method', 'test'),
            'store_creation_date': str(kwargs.get('store_creation_date', True)),
            'store_parameters': str(kwargs.get('store_parameters', True)),
        }
        
        config['validation'] = {
            'validate_after_creation': str(kwargs.get('validate_after_creation', False)),
            'check_finite_values': str(kwargs.get('check_finite_values', True)),
            'log_statistics': str(kwargs.get('log_statistics', False)),
        }
        
        config['logging'] = {
            'enable_logging': str(kwargs.get('enable_logging', False)),
            'log_file': kwargs.get('log_file', 'test.log'),
            'log_level': kwargs.get('log_level', 'INFO'),
            'log_format': kwargs.get('log_format', '%%(asctime)s - %%(levelname)s - %%(message)s'),
            'console_output': str(kwargs.get('console_output', False)),
            'file_output': str(kwargs.get('file_output', False)),
        }
        
        with open(self.config_file, 'w') as f:
            config.write(f)
    
    def test_basic_config_loading(self):
        """Test loading a basic configuration file"""
        self.create_minimal_config()
        gen = DatasetGen(config_file=self.config_file)
        
        self.assertEqual(gen.batch_size, 2)
        self.assertEqual(gen.num_images, 4)
        self.assertEqual(gen.image_width, 64)
        self.assertEqual(gen.image_height, 64)
        self.assertEqual(gen.k_values, [2, 4])
    
    def test_k_max_auto_parsing(self):
        """Test parsing k_max_values = auto"""
        self.create_minimal_config(k_max_values='auto')
        gen = DatasetGen(config_file=self.config_file)
        
        # Should have None for each k_value (auto mode)
        self.assertEqual(len(gen.k_max_values), len(gen.k_values))
        self.assertTrue(all(k is None for k in gen.k_max_values))
    
    def test_k_max_specific_values(self):
        """Test parsing specific k_max values"""
        self.create_minimal_config(k_values='2,4,8', k_max_values='32,64,128')
        gen = DatasetGen(config_file=self.config_file)
        
        self.assertEqual(gen.k_values, [2, 4, 8])
        self.assertEqual(gen.k_max_values, [32, 64, 128])
    
    def test_k_max_mixed_values(self):
        """Test parsing mixed auto and specific k_max values"""
        self.create_minimal_config(k_values='2,4,8', k_max_values='auto,64,none')
        gen = DatasetGen(config_file=self.config_file)
        
        self.assertEqual(gen.k_max_values, [None, 64, None])
    
    def test_safe_eval_simple_number(self):
        """Test _safe_eval with simple number"""
        self.create_minimal_config()
        gen = DatasetGen(config_file=self.config_file)
        
        result = gen._safe_eval('5.0')
        self.assertEqual(result, 5.0)
    
    def test_safe_eval_expression(self):
        """Test _safe_eval with mathematical expression"""
        self.create_minimal_config()
        gen = DatasetGen(config_file=self.config_file)
        
        result = gen._safe_eval('np.sqrt(5.)')
        self.assertAlmostEqual(result, np.sqrt(5.0), places=6)
    
    def test_safe_eval_invalid_expression(self):
        """Test _safe_eval with invalid expression"""
        self.create_minimal_config()
        gen = DatasetGen(config_file=self.config_file)
        
        with self.assertRaises(ValueError):
            gen._safe_eval('invalid_function(5)')
    
    def test_parameter_range_single_value(self):
        """Test parsing single parameter value"""
        self.create_minimal_config(mean='1.5')
        gen = DatasetGen(config_file=self.config_file)
        
        self.assertEqual(gen.mean_values, [1.5])
    
    def test_parameter_range_multiple_values(self):
        """Test parsing multiple parameter values"""
        # Create config with beta_values
        config = configparser.ConfigParser()
        config['dataset'] = {
            'output_file': 'test.h5',
            'batch_size': '2',
            'num_images': '4',
            'image_width': '64',
            'image_height': '64',
            'image_depth': '1',
            'k_values': '2,4',
        }
        config['distribution'] = {
            'mean': '1.0',
            'sigma': '2.0',
            'beta_values': '-1.5,-2.0,-2.5',
        }
        config['fractal'] = {
            'ni': '64', 'nj': '64', 'nk': '1',
            'verbose': 'False', 'random_seed': 'True',
        }
        config['hdf5'] = {
            'compression': 'gzip', 'compression_opts': '1',
            'chunks_auto': 'True', 'chunk_size_images': 'auto',
            'chunk_size_labels': 'auto',
        }
        config['processing'] = {
            'memory_monitoring': 'True', 'progress_reporting': 'False',
            'progress_report_interval': '1', 'clear_memory_after_batch': 'True',
            'force_garbage_collection': 'True', 'num_workers': '1',
        }
        config['metadata'] = {
            'method': 'test', 'store_creation_date': 'True', 'store_parameters': 'True',
        }
        config['validation'] = {
            'validate_after_creation': 'False', 'check_finite_values': 'True',
            'log_statistics': 'False',
        }
        config['logging'] = {
            'enable_logging': 'False', 'log_file': 'test.log', 'log_level': 'INFO',
            'log_format': '%%(asctime)s - %%(levelname)s - %%(message)s',
            'console_output': 'False', 'file_output': 'False',
        }
        
        with open(self.config_file, 'w') as f:
            config.write(f)
        
        gen = DatasetGen(config_file=self.config_file)
        self.assertEqual(gen.beta_values, [-1.5, -2.0, -2.5])
    
    def test_invalid_config_file(self):
        """Test handling of missing config file"""
        with self.assertRaises(FileNotFoundError):
            DatasetGen(config_file='nonexistent.ini')
    
    def test_validation_negative_dimensions(self):
        """Test validation catches negative dimensions"""
        self.create_minimal_config(image_width=-10)
        
        with self.assertRaises(ValueError):
            DatasetGen(config_file=self.config_file)
    
    def test_validation_dimension_mismatch(self):
        """Test validation catches dimension mismatch"""
        self.create_minimal_config(image_width=64, ni=128)
        
        with self.assertRaises(ValueError):
            DatasetGen(config_file=self.config_file)


class TestParameterCombinations(unittest.TestCase):
    """Test parameter combination generation"""
    
    def setUp(self):
        """Create test configuration"""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'test_config.ini')
        
        # Use the real test_config.ini for these tests
        if os.path.exists('test_config.ini'):
            self.config_file = 'test_config.ini'
        else:
            # Create minimal config
            config = configparser.ConfigParser()
            config['dataset'] = {
                'output_file': 'test.h5', 'batch_size': '2', 'num_images': '4',
                'image_width': '64', 'image_height': '64', 'image_depth': '1',
                'k_values': '2,4,8', 'k_max_values': 'auto,32,64'
            }
            config['distribution'] = {
                'mean_values': '1.0,2.0',
                'sigma': '2.0',
                'beta_values': '-1.5,-2.0'
            }
            config['fractal'] = {
                'ni': '64', 'nj': '64', 'nk': '1',
                'verbose': 'False', 'random_seed': 'True'
            }
            config['hdf5'] = {
                'compression': 'gzip', 'compression_opts': '1',
                'chunks_auto': 'True', 'chunk_size_images': 'auto',
                'chunk_size_labels': 'auto'
            }
            config['processing'] = {
                'memory_monitoring': 'True', 'progress_reporting': 'False',
                'progress_report_interval': '1', 'clear_memory_after_batch': 'True',
                'force_garbage_collection': 'True', 'num_workers': '1'
            }
            config['metadata'] = {
                'method': 'test', 'store_creation_date': 'True', 'store_parameters': 'True'
            }
            config['validation'] = {
                'validate_after_creation': 'False', 'check_finite_values': 'True',
                'log_statistics': 'False'
            }
            config['logging'] = {
                'enable_logging': 'False', 'log_file': 'test.log', 'log_level': 'INFO',
                'log_format': '%%(asctime)s - %%(levelname)s - %%(message)s',
                'console_output': 'False', 'file_output': 'False'
            }
            
            with open(self.config_file, 'w') as f:
                config.write(f)
    
    def tearDown(self):
        """Clean up"""
        if hasattr(self, 'test_dir') and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_combination_count(self):
        """Test correct number of parameter combinations"""
        gen = DatasetGen(config_file=self.config_file)
        
        # k_values (3) x mean_values (2) x sigma_values (1) x beta_values (2) = 12
        expected_combinations = 3 * 2 * 1 * 2
        self.assertEqual(len(gen.param_combinations), expected_combinations)
    
    def test_combination_structure(self):
        """Test structure of parameter combinations"""
        gen = DatasetGen(config_file=self.config_file)
        
        for combo in gen.param_combinations:
            self.assertIn('k_min', combo)
            self.assertIn('k_max', combo)
            self.assertIn('beta', combo)
            self.assertIn('mean', combo)
            self.assertIn('sigma', combo)
    
    def test_k_min_k_max_pairing(self):
        """Test that k_min and k_max are correctly paired"""
        gen = DatasetGen(config_file=self.config_file)
        
        # Check that k_min values match k_values
        k_min_values = [combo['k_min'] for combo in gen.param_combinations]
        self.assertTrue(all(k in gen.k_values for k in k_min_values))
        
        # Check k_max pairing
        k_pairs = list(zip(gen.k_values, gen.k_max_values))
        for combo in gen.param_combinations:
            k_min = combo['k_min']
            k_max = combo['k_max']
            # Find the corresponding k_max for this k_min
            expected_k_max = gen.k_max_values[gen.k_values.index(k_min)]
            self.assertEqual(k_max, expected_k_max)


class TestSingleImageGeneration(unittest.TestCase):
    """Test single image generation functionality"""
    
    def setUp(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'test_config.ini')
        
        config = configparser.ConfigParser()
        config['dataset'] = {
            'output_file': 'test.h5', 'batch_size': '2', 'num_images': '2',
            'image_width': '32', 'image_height': '32', 'image_depth': '1',
            'k_values': '2,4'
        }
        config['distribution'] = {
            'mean': '1.0', 'sigma': '2.0', 'beta': '-1.5'
        }
        config['fractal'] = {
            'ni': '32', 'nj': '32', 'nk': '1',
            'verbose': 'False', 'random_seed': 'True'
        }
        config['hdf5'] = {
            'compression': 'gzip', 'compression_opts': '1',
            'chunks_auto': 'True', 'chunk_size_images': 'auto',
            'chunk_size_labels': 'auto'
        }
        config['processing'] = {
            'memory_monitoring': 'False', 'progress_reporting': 'False',
            'progress_report_interval': '1', 'clear_memory_after_batch': 'True',
            'force_garbage_collection': 'False', 'num_workers': '1'
        }
        config['metadata'] = {'method': 'test', 'store_creation_date': 'True', 'store_parameters': 'True'}
        config['validation'] = {'validate_after_creation': 'False', 'check_finite_values': 'True', 'log_statistics': 'False'}
        config['logging'] = {'enable_logging': 'False', 'log_file': 'test.log', 'log_level': 'INFO', 'log_format': '%%(asctime)s - %%(levelname)s - %%(message)s', 'console_output': 'False', 'file_output': 'False'}
        
        with open(self.config_file, 'w') as f:
            config.write(f)
        
        self.gen = DatasetGen(config_file=self.config_file)
    
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_single_image_shape(self):
        """Test that generated image has correct shape"""
        img, params = self.gen.single_image(k_min=2, k_max=None)
        
        self.assertEqual(img.shape, (32, 32))
    
    def test_single_image_type(self):
        """Test that generated image is numpy array"""
        img, params = self.gen.single_image(k_min=2, k_max=None)
        
        self.assertIsInstance(img, np.ndarray)
    
    def test_single_image_parameters(self):
        """Test that parameters are returned correctly"""
        k_min, k_max = 2, 16
        beta, mean, sigma = -1.5, 1.0, 2.0
        
        img, params = self.gen.single_image(
            k_min=k_min, k_max=k_max, 
            beta=beta, mean=mean, sigma=sigma
        )
        
        self.assertEqual(params['k_min'], k_min)
        self.assertEqual(params['k_max'], k_max)
        self.assertEqual(params['beta'], beta)
        self.assertEqual(params['mean'], mean)
        self.assertEqual(params['sigma'], sigma)
    
    def test_single_image_auto_kmax(self):
        """Test image generation with auto k_max"""
        img, params = self.gen.single_image(k_min=2, k_max=None)
        
        # k_max should be set to Nyquist limit
        self.assertIsNotNone(params['k_max'])
        # For 32x32, Nyquist limit is 16
        self.assertEqual(params['k_max'], 16.0)
    
    def test_single_image_with_kmax(self):
        """Test image generation with specific k_max"""
        img, params = self.gen.single_image(k_min=2, k_max=8)
        
        self.assertEqual(params['k_max'], 8)


class TestWorkerFunction(unittest.TestCase):
    """Test the multiprocessing worker function"""
    
    def test_worker_basic_execution(self):
        """Test that worker function executes successfully"""
        img, params = _generate_image_worker(
            k_min=2, k_max=None,
            ni=32, nj=32, nk=1,
            mean=1.0, sigma=2.0, beta=-1.5,
            verbose=False, seed=42
        )
        
        self.assertEqual(img.shape, (32, 32))
        self.assertIsInstance(params, dict)
        self.assertEqual(params['seed'], 42)
    
    def test_worker_with_specific_kmax(self):
        """Test worker with specific k_max value"""
        img, params = _generate_image_worker(
            k_min=2, k_max=8,
            ni=32, nj=32, nk=1,
            mean=1.0, sigma=2.0, beta=-1.5,
            verbose=False, seed=123
        )
        
        self.assertEqual(params['k_max'], 8)
        self.assertEqual(params['k_min'], 2)
        self.assertEqual(params['seed'], 123)


class TestHDF5Operations(unittest.TestCase):
    """Test HDF5 file creation and operations"""
    
    def setUp(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.output_file = os.path.join(self.test_dir, 'test_dataset.h5')
        self.config_file = os.path.join(self.test_dir, 'test_config.ini')
        
        config = configparser.ConfigParser()
        config['dataset'] = {
            'output_file': self.output_file, 'batch_size': '2', 'num_images': '2',
            'image_width': '32', 'image_height': '32', 'image_depth': '1',
            'k_values': '2'
        }
        config['distribution'] = {
            'mean': '1.0', 'sigma': '2.0', 'beta': '-1.5'
        }
        config['fractal'] = {
            'ni': '32', 'nj': '32', 'nk': '1',
            'verbose': 'False', 'random_seed': 'True'
        }
        config['hdf5'] = {
            'compression': 'gzip', 'compression_opts': '1',
            'chunks_auto': 'True', 'chunk_size_images': 'auto',
            'chunk_size_labels': 'auto'
        }
        config['processing'] = {
            'memory_monitoring': 'False', 'progress_reporting': 'False',
            'progress_report_interval': '1', 'clear_memory_after_batch': 'True',
            'force_garbage_collection': 'False', 'num_workers': '1'
        }
        config['metadata'] = {'method': 'test', 'store_creation_date': 'True', 'store_parameters': 'True'}
        config['validation'] = {'validate_after_creation': 'False', 'check_finite_values': 'True', 'log_statistics': 'False'}
        config['logging'] = {'enable_logging': 'False', 'log_file': 'test.log', 'log_level': 'INFO', 'log_format': '%%(asctime)s - %%(levelname)s - %%(message)s', 'console_output': 'False', 'file_output': 'False'}
        
        with open(self.config_file, 'w') as f:
            config.write(f)
        
        self.gen = DatasetGen(config_file=self.config_file)
    
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_hdf5_creation(self):
        """Test HDF5 file is created"""
        total_images = 4
        self.gen.create_hdf5(total_images)
        
        self.assertTrue(os.path.exists(self.output_file))
    
    def test_hdf5_structure(self):
        """Test HDF5 file has correct structure"""
        total_images = 4
        self.gen.create_hdf5(total_images)
        
        with h5py.File(self.output_file, 'r') as f:
            # Check main datasets
            self.assertIn('images', f)
            self.assertIn('parameters', f)
            
            # Check images shape
            self.assertEqual(f['images'].shape, (4, 32, 32))
            
            # Check parameter datasets
            params_group = f['parameters']
            required_params = ['k_min', 'k_max', 'beta', 'mean', 'sigma', 'ni', 'nj', 'nk']
            for param in required_params:
                self.assertIn(param, params_group)
                self.assertEqual(len(params_group[param]), total_images)
    
    def test_hdf5_compression(self):
        """Test HDF5 compression is applied"""
        total_images = 4
        self.gen.create_hdf5(total_images)
        
        with h5py.File(self.output_file, 'r') as f:
            self.assertEqual(f['images'].compression, 'gzip')


class TestDatasetPopulation(unittest.TestCase):
    """Test full dataset generation"""
    
    def setUp(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.output_file = os.path.join(self.test_dir, 'test_dataset.h5')
        self.config_file = os.path.join(self.test_dir, 'test_config.ini')
        
        config = configparser.ConfigParser()
        config['dataset'] = {
            'output_file': self.output_file, 
            'batch_size': '2', 
            'num_images': '2',
            'image_width': '32', 
            'image_height': '32', 
            'image_depth': '1',
            'k_values': '2,4',
            'k_max_values': 'auto,16'
        }
        config['distribution'] = {
            'mean': '1.0', 'sigma': '2.0', 'beta': '-1.5'
        }
        config['fractal'] = {
            'ni': '32', 'nj': '32', 'nk': '1',
            'verbose': 'False', 'random_seed': 'True'
        }
        config['hdf5'] = {
            'compression': 'gzip', 'compression_opts': '1',
            'chunks_auto': 'True', 'chunk_size_images': 'auto',
            'chunk_size_labels': 'auto'
        }
        config['processing'] = {
            'memory_monitoring': 'False', 
            'progress_reporting': 'False',
            'progress_report_interval': '1', 
            'clear_memory_after_batch': 'True',
            'force_garbage_collection': 'False', 
            'num_workers': '1'
        }
        config['metadata'] = {
            'method': 'test', 'store_creation_date': 'True', 'store_parameters': 'True'
        }
        config['validation'] = {
            'validate_after_creation': 'False', 'check_finite_values': 'True',
            'log_statistics': 'False'
        }
        config['logging'] = {
            'enable_logging': 'False', 'log_file': 'test.log', 'log_level': 'INFO',
            'log_format': '%%(asctime)s - %%(levelname)s - %%(message)s',
            'console_output': 'False', 'file_output': 'False'
        }
        
        with open(self.config_file, 'w') as f:
            config.write(f)
    
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_populate_dataset_serial(self):
        """Test serial dataset population"""
        gen = DatasetGen(config_file=self.config_file, num_workers=1)
        gen.populate_dataset()
        
        # Check file exists
        self.assertTrue(os.path.exists(self.output_file))
        
        # Check content
        with h5py.File(self.output_file, 'r') as f:
            # 2 k_values x 2 images each = 4 total images
            self.assertEqual(f['images'].shape[0], 4)
            self.assertIn('parameters', f)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility and helper functions"""
    
    def setUp(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, 'test_dataset.h5')
        
        # Create a simple test dataset
        with h5py.File(self.test_file, 'w') as f:
            f.create_dataset('images', data=np.random.rand(10, 32, 32), dtype='float32')
            params = f.create_group('parameters')
            params.create_dataset('k_min', data=np.array([2]*10))
            params.create_dataset('k_max', data=np.array([16]*10))
            params.create_dataset('beta', data=np.array([-1.5]*10))
            params.create_dataset('mean', data=np.array([1.0]*10))
            params.create_dataset('sigma', data=np.array([2.0]*10))
            params.create_dataset('ni', data=np.array([32]*10))
            params.create_dataset('nj', data=np.array([32]*10))
            params.create_dataset('nk', data=np.array([1]*10))
            params.create_dataset('seed', data=np.array([0]*10, dtype=np.int64))
            
            # Add metadata
            f.attrs['dataset_name'] = 'Test Dataset'
            f.attrs['num_images'] = 10
    
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_load_dataset(self):
        """Test loading entire dataset"""
        data = DatasetGen.load_dataset(self.test_file)
        
        self.assertIn('images', data)
        self.assertIn('k_min', data)
        self.assertEqual(len(data['images']), 10)
    
    def test_load_subset(self):
        """Test loading dataset subset"""
        indices = np.array([0, 2, 4])
        data = DatasetGen.load_subset(self.test_file, indices)
        
        self.assertEqual(len(data['images']), 3)
    
    def test_get_dataset_info(self):
        """Test getting dataset information"""
        info = DatasetGen.get_dataset_info(self.test_file)
        
        self.assertIn('total_images', info)
        self.assertIn('image_shape', info)
        self.assertEqual(info['total_images'], 10)
        self.assertEqual(info['image_shape'], (32, 32))
    
    def test_validation(self):
        """Test dataset validation"""
        # Should pass validation
        is_valid = dg.validation(self.test_file)
        self.assertTrue(is_valid)


class TestMemoryManagement(unittest.TestCase):
    """Test memory monitoring and management"""
    
    def setUp(self):
        """Setup test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.test_dir, 'test_config.ini')
        
        config = configparser.ConfigParser()
        config['dataset'] = {
            'output_file': 'test.h5', 'batch_size': '2', 'num_images': '2',
            'image_width': '32', 'image_height': '32', 'image_depth': '1',
            'k_values': '2'
        }
        config['distribution'] = {
            'mean': '1.0', 'sigma': '2.0', 'beta': '-1.5'
        }
        config['fractal'] = {
            'ni': '32', 'nj': '32', 'nk': '1',
            'verbose': 'False', 'random_seed': 'True'
        }
        config['hdf5'] = {
            'compression': 'gzip', 'compression_opts': '1',
            'chunks_auto': 'True', 'chunk_size_images': 'auto',
            'chunk_size_labels': 'auto'
        }
        config['processing'] = {
            'memory_monitoring': 'True', 
            'progress_reporting': 'False',
            'progress_report_interval': '1', 
            'clear_memory_after_batch': 'True',
            'force_garbage_collection': 'True', 
            'num_workers': '1'
        }
        config['metadata'] = {
            'method': 'test', 'store_creation_date': 'True', 'store_parameters': 'True'
        }
        config['validation'] = {
            'validate_after_creation': 'False', 'check_finite_values': 'True',
            'log_statistics': 'False'
        }
        config['logging'] = {
            'enable_logging': 'False', 'log_file': 'test.log', 'log_level': 'INFO',
            'log_format': '%%(asctime)s - %%(levelname)s - %%(message)s',
            'console_output': 'False', 'file_output': 'False'
        }
        
        with open(self.config_file, 'w') as f:
            config.write(f)
    
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_memory_monitoring_enabled(self):
        """Test memory monitoring when enabled"""
        gen = DatasetGen(config_file=self.config_file)
        memory_usage = gen.get_memory_usage()
        
        self.assertIsInstance(memory_usage, float)
        self.assertGreater(memory_usage, 0)
    
    def test_memory_monitoring_disabled(self):
        """Test memory monitoring when disabled"""
        # Recreate config with memory monitoring off
        config = configparser.ConfigParser()
        config['dataset'] = {
            'output_file': 'test.h5', 'batch_size': '2', 'num_images': '2',
            'image_width': '32', 'image_height': '32', 'image_depth': '1',
            'k_values': '2'
        }
        config['distribution'] = {
            'mean': '1.0', 'sigma': '2.0', 'beta': '-1.5'
        }
        config['fractal'] = {
            'ni': '32', 'nj': '32', 'nk': '1',
            'verbose': 'False', 'random_seed': 'True'
        }
        config['hdf5'] = {
            'compression': 'gzip', 'compression_opts': '1',
            'chunks_auto': 'True', 'chunk_size_images': 'auto',
            'chunk_size_labels': 'auto'
        }
        config['processing'] = {
            'memory_monitoring': 'False', 
            'progress_reporting': 'False',
            'progress_report_interval': '1', 
            'clear_memory_after_batch': 'True',
            'force_garbage_collection': 'False', 
            'num_workers': '1'
        }
        config['metadata'] = {
            'method': 'test', 'store_creation_date': 'True', 'store_parameters': 'True'
        }
        config['validation'] = {
            'validate_after_creation': 'False', 'check_finite_values': 'True',
            'log_statistics': 'False'
        }
        config['logging'] = {
            'enable_logging': 'False', 'log_file': 'test.log', 'log_level': 'INFO',
            'log_format': '%%(asctime)s - %%(levelname)s - %%(message)s',
            'console_output': 'False', 'file_output': 'False'
        }
        
        config_file = os.path.join(self.test_dir, 'test_config2.ini')
        with open(config_file, 'w') as f:
            config.write(f)
        
        gen = DatasetGen(config_file=config_file)
        memory_usage = gen.get_memory_usage()
        
        self.assertEqual(memory_usage, 0.0)


class TestMergeAppend(unittest.TestCase):
    """Tests for merge/append HDF5 functionality"""

    def setUp(self):
        """Create temp directory and two small datasets"""
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, 'merge_test.ini')

        config_content = """[dataset]
output_file = {dir}/ds_a.hdf5
batch_size = 2
num_images = 2
image_width = 64
image_height = 64
image_depth = 1
k_values = 2,4
k_max_values = auto

[distribution]
mean = 1.0
sigma = 2.236067977
beta = -1.666666667

[fractal]
ni = 64
nj = 64
nk = 1
verbose = false
random_seed = true

[hdf5]
compression = gzip
compression_opts = 1
chunks_auto = true
chunk_size_images = auto
chunk_size_labels = auto

[processing]
memory_monitoring = false
progress_reporting = false
progress_report_interval = 1
clear_memory_after_batch = true
force_garbage_collection = true
num_workers = 1

[metadata]
method = test
store_creation_date = true
store_parameters = true

[validation]
validate_after_creation = false
check_finite_values = true
log_statistics = false

[logging]
enable_logging = false
log_file = {dir}/merge_test.log
log_level = WARNING
log_format = %%(asctime)s - %%(levelname)s - %%(message)s
console_output = false
file_output = false
""".format(dir=self.test_dir)

        with open(self.config_path, 'w') as f:
            f.write(config_content)

        # Generate dataset A
        self.dg_a = DatasetGen(self.config_path)
        self.dg_a.populate_dataset()

        # Generate dataset B (different k_values)
        config_b = config_content.replace('ds_a.hdf5', 'ds_b.hdf5').replace(
            'k_values = 2,4', 'k_values = 8,16')
        self.config_b_path = os.path.join(self.test_dir, 'merge_test_b.ini')
        with open(self.config_b_path, 'w') as f:
            f.write(config_b)
        self.dg_b = DatasetGen(self.config_b_path)
        self.dg_b.populate_dataset()

        self.file_a = os.path.join(self.test_dir, 'ds_a.hdf5')
        self.file_b = os.path.join(self.test_dir, 'ds_b.hdf5')

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_detect_layout_flat(self):
        layout = DatasetGen._detect_layout(self.file_a)
        self.assertEqual(layout, 'flat')

    def test_merge_same_dimension(self):
        out = os.path.join(self.test_dir, 'merged.hdf5')
        DatasetGen.merge_datasets([self.file_a, self.file_b], out)
        self.assertTrue(os.path.exists(out))
        layout = DatasetGen._detect_layout(out)
        self.assertEqual(layout, 'grouped')
        with h5py.File(out, 'r') as f:
            self.assertIn('64x64', f)

    def test_merge_image_count(self):
        out = os.path.join(self.test_dir, 'merged.hdf5')
        DatasetGen.merge_datasets([self.file_a, self.file_b], out, dup_check=False)
        with h5py.File(out, 'r') as f:
            n_a = h5py.File(self.file_a, 'r')['images'].shape[0]
            n_b = h5py.File(self.file_b, 'r')['images'].shape[0]
            n_merged = f['64x64']['images'].shape[0]
        self.assertEqual(n_merged, n_a + n_b)

    def test_merge_duplicate_detection(self):
        out = os.path.join(self.test_dir, 'merged_dup.hdf5')
        DatasetGen.merge_datasets([self.file_a, self.file_a], out, dup_check=True)
        with h5py.File(out, 'r') as f:
            n_merged = f['64x64']['images'].shape[0]
        # With dup_check, only unique param tuples survive
        # k_values=2,4 → 2 combos, dedup keeps first occurrence of each
        with h5py.File(self.file_a, 'r') as f_a:
            n_a = f_a['images'].shape[0]
        # Merging same file twice with dup_check should give fewer than 2*n_a
        self.assertLess(n_merged, n_a * 2)

    def test_merge_no_dup_check(self):
        out = os.path.join(self.test_dir, 'merged_nodup.hdf5')
        DatasetGen.merge_datasets([self.file_a, self.file_a], out, dup_check=False)
        with h5py.File(out, 'r') as f:
            n_merged = f['64x64']['images'].shape[0]
        n_a = h5py.File(self.file_a, 'r')['images'].shape[0]
        self.assertEqual(n_merged, n_a * 2)

    def test_merge_different_dimensions(self):
        config_c = """[dataset]
output_file = {dir}/ds_c.hdf5
batch_size = 2
num_images = 2
image_width = 32
image_height = 32
image_depth = 1
k_values = 2,4
k_max_values = auto

[distribution]
mean = 1.0
sigma = 2.236067977
beta = -1.666666667

[fractal]
ni = 32
nj = 32
nk = 1
verbose = false
random_seed = true

[hdf5]
compression = gzip
compression_opts = 1
chunks_auto = true
chunk_size_images = auto
chunk_size_labels = auto

[processing]
memory_monitoring = false
progress_reporting = false
progress_report_interval = 1
clear_memory_after_batch = true
force_garbage_collection = true
num_workers = 1

[metadata]
method = test
store_creation_date = true
store_parameters = true

[validation]
validate_after_creation = false
check_finite_values = true
log_statistics = false

[logging]
enable_logging = false
log_file = {dir}/merge_test_c.log
log_level = WARNING
log_format = %%(asctime)s - %%(levelname)s - %%(message)s
console_output = false
file_output = false
""".format(dir=self.test_dir)
        config_c_path = os.path.join(self.test_dir, 'merge_test_c.ini')
        with open(config_c_path, 'w') as f:
            f.write(config_c)
        dg_c = DatasetGen(config_c_path)
        dg_c.populate_dataset()
        file_c = os.path.join(self.test_dir, 'ds_c.hdf5')

        out = os.path.join(self.test_dir, 'merged_multi.hdf5')
        DatasetGen.merge_datasets([self.file_a, file_c], out)
        with h5py.File(out, 'r') as f:
            self.assertIn('64x64', f)
            self.assertIn('32x32', f)

    def test_detect_layout_grouped(self):
        out = os.path.join(self.test_dir, 'merged.hdf5')
        DatasetGen.merge_datasets([self.file_a, self.file_b], out)
        layout = DatasetGen._detect_layout(out)
        self.assertEqual(layout, 'grouped')

    def test_append_flat_to_flat(self):
        self.dg_a.append_dataset(self.file_b)
        layout = DatasetGen._detect_layout(self.file_a)
        self.assertEqual(layout, 'grouped')

    def test_load_dataset_flat(self):
        data = DatasetGen.load_dataset(self.file_a)
        self.assertIn('images', data)
        self.assertIn('k_min', data)

    def test_load_dataset_grouped(self):
        out = os.path.join(self.test_dir, 'merged.hdf5')
        DatasetGen.merge_datasets([self.file_a, self.file_b], out)
        data = DatasetGen.load_dataset(out, dimension='64x64')
        self.assertIn('images', data)

    def test_get_dataset_info_grouped(self):
        out = os.path.join(self.test_dir, 'merged.hdf5')
        DatasetGen.merge_datasets([self.file_a, self.file_b], out)
        info = DatasetGen.get_dataset_info(out)
        self.assertEqual(info['layout'], 'grouped')
        self.assertIn('dimension_groups', info)
        self.assertIn('64x64', info['dimension_groups'])

    def test_validation_both_layouts(self):
        self.assertTrue(validation(self.file_a))
        out = os.path.join(self.test_dir, 'merged.hdf5')
        DatasetGen.merge_datasets([self.file_a, self.file_b], out)
        self.assertTrue(validation(out))


def run_test_suite():
    """Run all tests and generate report"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestConfigParsing))
    suite.addTests(loader.loadTestsFromTestCase(TestParameterCombinations))
    suite.addTests(loader.loadTestsFromTestCase(TestSingleImageGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestWorkerFunction))
    suite.addTests(loader.loadTestsFromTestCase(TestHDF5Operations))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasetPopulation))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryManagement))
    suite.addTests(loader.loadTestsFromTestCase(TestMergeAppend))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {100 * (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun:.1f}%")
    print("="*70)
    
    return result


if __name__ == '__main__':
    result = run_test_suite()
    sys.exit(0 if result.wasSuccessful() else 1)
