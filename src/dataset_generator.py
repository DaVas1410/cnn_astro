"""
Dataset Generator for Fractal Cube Images

A modular and configurable class for generating synthetic datasets from fractal 
cube simulations using the pyFC library. All parameters are loaded from an external 
configuration file to ensure reusability and maintainability.

Author: J.D.V.V
Date: February 2026
"""

import numpy as np
import pyFC
import h5py
import os
import gc
import psutil
from datetime import datetime
import configparser
from typing import List, Optional, Tuple, Dict, Any
import logging
import multiprocessing as mp
from functools import partial


# --------------------------------------------------------------------------------
# MULTIPROCESSING WORKER FUNCTION
# --------------------------------------------------------------------------------

def _gen_cube_compat(fc, verbose: bool, seed: Optional[int]) -> None:
    """
    Call pyFC gen_cube across API variants.

    Some pyFC versions accept `random_seed`, others `seed`, and older ones
    accept no seed kwarg at all.
    """
    if seed is None:
        try:
            fc.gen_cube(verbose=verbose, random_seed=False)
            return
        except TypeError:
            try:
                fc.gen_cube(verbose=verbose, seed=False)
                return
            except TypeError:
                fc.gen_cube(verbose=verbose)
                return

    try:
        fc.gen_cube(verbose=verbose, random_seed=int(seed))
        return
    except TypeError:
        pass

    try:
        fc.gen_cube(verbose=verbose, seed=int(seed))
        return
    except TypeError:
        pass

    np.random.seed(int(seed))
    fc.gen_cube(verbose=verbose)

def _generate_image_worker(k_min: int, k_max: Optional[int], ni: int, nj: int, nk: int, 
                           mean: float, sigma: float, beta: float,
                           verbose: bool = False, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
    """
    Worker function for parallel image generation.
    Must be at module level to be picklable by multiprocessing.
    
    Args:
        k_min: Minimum wavenumber
        k_max: Maximum wavenumber (None for auto)
        ni, nj, nk: Grid dimensions
        mean: Log-normal distribution mean
        sigma: Log-normal distribution sigma
        beta: Power spectrum slope
        verbose: Verbosity flag
        seed: Integer seed for reproducibility (None uses pyFC default seed 123)
        
    Returns:
        Tuple of (image_array, parameters_dict)
    """
    try:
        # Generate fractal cube
        fc = pyFC.LogNormalFractalCube(
            ni=ni, nj=nj, nk=nk,
            kmin=k_min,  # Note: pyFC uses 'kmin' not 'k_min'
            kmax=k_max,  # Use specified k_max or None for auto
            mean=mean,
            sigma=sigma,
            beta=beta
        )
        
        # Generate the cube with pyFC API compatibility
        _gen_cube_compat(fc, verbose=verbose, seed=seed)
        
        # Extract 2D slice
        fcs = pyFC.FCSlicer()
        slice_data = fcs.slice(fc, ax=2)
        
        # Apply log10 transformation
        slice_log = np.log10(slice_data)
        
        # Collect parameters (including actual k_max from pyFC)
        params = {
            'k_min': k_min,
            'k_max': fc.kmax,  # Actual k_max set by pyFC (note: it's 'kmax' not 'k_max')
            'beta': beta,
            'mean': mean,
            'sigma': sigma,
            'ni': ni,
            'nj': nj,
            'nk': nk,
            'seed': seed if seed is not None else 0
        }
        
        # Clean up to save memory in worker process
        del fc, fcs, slice_data
        
        return slice_log, params
        
    except Exception as e:
        # Return error information
        k_max_str = 'auto' if k_max is None else str(k_max)
        raise RuntimeError(f"Worker failed for k_min={k_min}, k_max={k_max_str}: {str(e)}") from e


# --------------------------------------------------------------------------------
# DATASET GENERATOR CLASS
# --------------------------------------------------------------------------------

class DatasetGen:
    """
    Modular Dataset Generator for Fractal Cube Images
    
    A configurable class for generating synthetic datasets from fractal cube simulations
    using the pyFC library. All parameters are loaded from an external configuration file
    to make the class highly modular and reusable.
    
    Parameters can be configured via an INI file with sections for:
    - dataset: output settings, batch size, image dimensions
    - distribution: log normal distribution parameters  
    - fractal: fractal cube generation settings
    - hdf5: file compression and chunking options
    - processing: memory management and progress reporting
    - metadata: information to store in output file
    - validation: data validation options
    - logging: log file configuration and output settings
    
    Example:
        >>> generator = DatasetGen(config_file='dataset_config.ini')
        >>> generator.populate_dataset()
    """
    
    def __init__(self, config_file: str = 'configs/dataset_config.ini', **overrides):
        """
        Initialize DatasetGen with configuration from file
        
        Args:
            config_file: Path to configuration INI file
            **overrides: Optional parameter overrides for specific values
                        (e.g., output_file='custom.h5', num_images=500)
        """
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        
        # Load configuration
        self._load_config()
        
        # Apply any overrides
        for key, value in overrides.items():
            if hasattr(self, key):
                # Special handling for k_values list
                if key == 'k_values' and isinstance(value, list):
                    setattr(self, key, value)
                else:
                    setattr(self, key, value)
        
        # Validate configuration
        self._validate_config()
        
        # Print configuration summary
        self._print_config_summary()
    
    def _safe_eval(self, value_str: str) -> float:
        """
        Safely evaluate mathematical expressions in config values
        
        Args:
            value_str: String containing a mathematical expression or number
            
        Returns:
            Evaluated float value
            
        Raises:
            ValueError: If expression cannot be safely evaluated
        """
        import numpy as np
        import math
        
        # If it's already a number, return it directly
        try:
            return float(value_str)
        except ValueError:
            pass
        
        # Define safe mathematical functions and constants
        safe_dict = {
            'np': np,
            'math': math,
            'sqrt': math.sqrt,
            'pi': math.pi,
            'e': math.e,
            '__builtins__': {}
        }
        
        try:
            # Safely evaluate the expression
            result = eval(value_str, safe_dict)
            return float(result)
        except Exception as e:
            raise ValueError(f"Could not evaluate expression '{value_str}': {e}")
    
    def _parse_parameter_range(self, section: str, param: str) -> list:
        """
        Parse parameter from config, supporting both single values and comma-separated lists
        
        Args:
            section: Config section name
            param: Parameter name (e.g., 'beta', 'mean', 'sigma')
            
        Returns:
            List of parameter values
        """
        # First try to get the _values version (e.g., 'beta_values')
        values_key = f'{param}_values'
        if self.config.has_option(section, values_key):
            values_str = self.config.get(section, values_key)
            return [self._safe_eval(v.strip()) for v in values_str.split(',')]
        # Otherwise get single value and return as list
        elif self.config.has_option(section, param):
            return [self._safe_eval(self.config.get(section, param))]
        else:
            raise ValueError(f"Neither '{param}' nor '{values_key}' found in [{section}]")
    
    def _generate_parameter_combinations(self):
        """
        Generate all combinations of k_min, k_max, beta, mean, and sigma values
        
        Creates self.param_combinations as a list of dictionaries, each containing
        one combination of parameters to generate images for.
        """
        from itertools import product
        
        # Create k_min, k_max pairs
        k_pairs = list(zip(self.k_values, self.k_max_values))
        
        # Generate all combinations
        self.param_combinations = []
        for (k_min, k_max), beta, mean, sigma in product(
            k_pairs,
            self.beta_values, 
            self.mean_values, 
            self.sigma_values
        ):
            self.param_combinations.append({
                'k_min': k_min,
                'k_max': k_max,
                'beta': beta,
                'mean': mean,
                'sigma': sigma
            })
    
    def _load_config(self):
        """
        Load and parse configuration from INI file
        
        Raises:
            FileNotFoundError: If configuration file doesn't exist
        """
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        
        self.config.read(self.config_file)
        
        # Dataset parameters
        self.output_file = self.config.get('dataset', 'output_file')
        self.batch_size = self.config.getint('dataset', 'batch_size')
        self.num_images = self.config.getint('dataset', 'num_images')
        self.image_width = self.config.getint('dataset', 'image_width')
        self.image_height = self.config.getint('dataset', 'image_height')
        self.image_depth = self.config.getint('dataset', 'image_depth')
        
        # Parse k_values from comma-separated string
        k_values_str = self.config.get('dataset', 'k_values')
        self.k_values = [int(k.strip()) for k in k_values_str.split(',')]
        
        # Parse k_max_values (support 'auto', 'none', or comma-separated integers)
        if self.config.has_option('dataset', 'k_max_values'):
            k_max_str = self.config.get('dataset', 'k_max_values').strip().lower()
            if k_max_str in ['auto', 'none']:
                # Single 'auto' or 'none' applies to all k_values
                self.k_max_values = [None] * len(self.k_values)
            else:
                # Parse comma-separated k_max values
                k_max_list = [k.strip() for k in k_max_str.split(',')]
                self.k_max_values = []
                for k in k_max_list:
                    if k.lower() in ['auto', 'none']:
                        self.k_max_values.append(None)
                    else:
                        self.k_max_values.append(int(k))
        else:
            # Default: auto for all k_values
            self.k_max_values = [None] * len(self.k_values)
        
        # Distribution parameters - support both single values and ranges
        # Try to get _values (list) versions first, otherwise use single value
        self.beta_values = self._parse_parameter_range('distribution', 'beta')
        self.mean_values = self._parse_parameter_range('distribution', 'mean')
        self.sigma_values = self._parse_parameter_range('distribution', 'sigma')
        
        # Keep backward compatibility - set single values to first in list
        self.beta = self.beta_values[0]
        self.mean = self.mean_values[0]
        self.sigma = self.sigma_values[0]
        
        # Generate all parameter combinations
        self._generate_parameter_combinations()
        
        # Fractal parameters
        self.ni = self.config.getint('fractal', 'ni')
        self.nj = self.config.getint('fractal', 'nj')
        self.nk = self.config.getint('fractal', 'nk')
        self.verbose = self.config.getboolean('fractal', 'verbose')
        
        # Parse random_seed: 'random' | integer | 'true'/'false' (backward compat)
        seed_raw = self.config.get('fractal', 'random_seed').strip().lower()
        if seed_raw == 'random':
            self.seed_mode = 'random'
        elif seed_raw in ('true', 'yes', '1'):
            self.seed_mode = 'random'  # backward compat: true → random
        elif seed_raw in ('false', 'no', '0'):
            self.seed_mode = False  # fixed seed (pyFC default: 123)
        else:
            try:
                self.seed_mode = int(seed_raw)
            except ValueError:
                raise ValueError(f"Invalid random_seed value: '{seed_raw}'. Use 'random', an integer, 'true', or 'false'.")
        
        # HDF5 parameters
        self.compression = self.config.get('hdf5', 'compression')
        self.compression_opts = self.config.getint('hdf5', 'compression_opts')
        self.chunks_auto = self.config.getboolean('hdf5', 'chunks_auto')
        
        # Parse chunk sizes
        chunk_images_str = self.config.get('hdf5', 'chunk_size_images')
        if chunk_images_str.lower() != 'auto':
            self.chunk_size_images = tuple(int(x.strip()) for x in chunk_images_str.split(','))
        else:
            self.chunk_size_images = None
            
        chunk_labels_str = self.config.get('hdf5', 'chunk_size_labels')
        if chunk_labels_str.lower() != 'auto':
            self.chunk_size_labels = tuple(int(x.strip()) for x in chunk_labels_str.split(','))
        else:
            self.chunk_size_labels = None
        
        # Processing parameters
        self.memory_monitoring = self.config.getboolean('processing', 'memory_monitoring')
        self.progress_reporting = self.config.getboolean('processing', 'progress_reporting')
        self.progress_report_interval = self.config.getint('processing', 'progress_report_interval')
        self.clear_memory_after_batch = self.config.getboolean('processing', 'clear_memory_after_batch')
        self.force_garbage_collection = self.config.getboolean('processing', 'force_garbage_collection')
        self.num_workers = self.config.getint('processing', 'num_workers', fallback=1)
        
        # Metadata parameters
        self.method = self.config.get('metadata', 'method')
        self.store_creation_date = self.config.getboolean('metadata', 'store_creation_date')
        self.store_parameters = self.config.getboolean('metadata', 'store_parameters')
        
        # Validation parameters
        self.validate_after_creation = self.config.getboolean('validation', 'validate_after_creation')
        self.check_finite_values = self.config.getboolean('validation', 'check_finite_values')
        self.log_statistics = self.config.getboolean('validation', 'log_statistics')
        
        # Logging parameters
        self.enable_logging = self.config.getboolean('logging', 'enable_logging')
        self.log_file = self.config.get('logging', 'log_file')
        self.log_level = self.config.get('logging', 'log_level')
        self.log_format = self.config.get('logging', 'log_format')
        self.console_output = self.config.getboolean('logging', 'console_output')
        self.file_output = self.config.getboolean('logging', 'file_output')
        
        # Initialize logging if enabled
        if self.enable_logging:
            self._setup_logging()
    
    def _validate_config(self):
        """
        Validate loaded configuration parameters
        
        Raises:
            ValueError: If any configuration parameters are invalid
        """
        errors = []
        
        # Validate dimensions
        if self.image_width <= 0 or self.image_height <= 0:
            errors.append("Image dimensions must be positive")
        
        if self.batch_size <= 0:
            errors.append("Batch size must be positive")
            
        if self.num_images <= 0:
            errors.append("Number of images must be positive")
        
        # Validate k_values
        if not self.k_values or any(k <= 0 for k in self.k_values):
            errors.append("K-values must be positive integers")
        
        # Validate distribution parameters
        if self.sigma <= 0:
            errors.append("Sigma must be positive")
            
        # Validate fractal dimensions match image dimensions
        if self.ni != self.image_width or self.nj != self.image_height:
            errors.append("Fractal dimensions (ni, nj) must match image dimensions")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
    
    def _setup_logging(self):
        """Setup logging configuration"""
        # Create logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, self.log_level.upper()))
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(self.log_format)
        
        # Add file handler if enabled
        if self.file_output:
            file_handler = logging.FileHandler(self.log_file, mode='w')
            file_handler.setLevel(getattr(logging, self.log_level.upper()))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
        # Prevent propagation to avoid duplicate messages
        self.logger.propagate = False
    
    def log(self, message: str, level: str = 'INFO'):
        """
        Log a message if logging is enabled (only to file)
        
        Args:
            message: Message to log
            level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        """
        if self.enable_logging and hasattr(self, 'logger'):
            log_func = getattr(self.logger, level.lower(), self.logger.info)
            log_func(message)
    
    def _print_config_summary(self):
        """Print clean configuration summary to console only"""
        print("=" * 60)
        print("DATASET GENERATOR CONFIGURATION")
        print("=" * 60)
        print(f"Config File: {self.config_file}")
        print(f"Output File: {self.output_file}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Image Dimensions: {self.image_width} x {self.image_height} x {self.image_depth}")
        print(f"Images per parameter combination: {self.num_images}")
        print(f"K-min values: {self.k_values}")
        
        # Format k_max display
        k_max_display = ['auto' if k is None else str(k) for k in self.k_max_values]
        print(f"K-max values: {k_max_display}")
        
        # Show parameter ranges
        if len(self.beta_values) > 1:
            print(f"Beta values: {self.beta_values}")
        else:
            print(f"Beta: {self.beta_values[0]:.3f}")
        
        if len(self.mean_values) > 1:
            print(f"Mean values: {self.mean_values}")
        else:
            print(f"Mean: {self.mean_values[0]}")
        
        if len(self.sigma_values) > 1:
            print(f"Sigma values: {self.sigma_values}")
        else:
            print(f"Sigma: {self.sigma_values[0]:.3f}")
        
        print(f"Parameter combinations: {len(self.param_combinations)}")
        print(f"Total images: {len(self.param_combinations) * self.num_images}")
        print(f"Compression: {self.compression} (level {self.compression_opts})")
        print(f"Memory Monitoring: {self.memory_monitoring}")
        print(f"Logging: {'Enabled' if self.enable_logging else 'Disabled'}")
        if self.enable_logging:
            print(f"Log File: {self.log_file}")
        print("=" * 60)
        
        # Log configuration to file if logging is enabled
        if self.enable_logging:
            self.log("DatasetGen initialized with configuration:", 'INFO')
            self.log(f"Config File: {self.config_file}", 'INFO')
            self.log(f"Output File: {self.output_file}", 'INFO')
            self.log(f"Batch Size: {self.batch_size}, Images per combination: {self.num_images}", 'INFO')
            self.log(f"Parameter combinations: {len(self.param_combinations)}", 'INFO')

    def get_memory_usage(self) -> float:
        """
        Get current RAM usage in MB
        
        Returns:
            Current memory usage in megabytes, or 0.0 if monitoring is disabled
        """
        if not self.memory_monitoring:
            return 0.0
        
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def create_hdf5(self, total_images: int):
        """
        Create the HDF5 file structure with flat layout for comprehensive parameter tracking
        
        Creates a single images dataset and separate parameter datasets for each generation parameter.
        This allows for flexible multi-label learning and turbulence analysis.
        
        Args:
            total_images: Total number of images that will be stored
        """
        self.log("Creating HDF5 file structure (flat layout)", 'INFO')
        
        with h5py.File(self.output_file, 'w') as f:
            # Determine chunk sizes
            image_chunks = self.chunk_size_images if not self.chunks_auto else (1, self.image_height, self.image_width)
            param_chunks = True  # Auto-chunking for parameters
            
            # Main images dataset - all images in one tensor
            f.create_dataset(
                'images',
                shape=(total_images, self.image_height, self.image_width),
                maxshape=(None, self.image_height, self.image_width),
                dtype=np.float32,
                compression=self.compression,
                compression_opts=self.compression_opts,
                chunks=image_chunks
            )
            
            # Create parameters group for organization
            params_group = f.create_group('parameters')
            
            # Fractal generation parameters (can vary per image)
            params_group.create_dataset(
                'k_min',
                shape=(total_images,),
                maxshape=(None,),
                dtype=np.float32,
                compression=self.compression,
                compression_opts=self.compression_opts,
                chunks=param_chunks
            )
            
            params_group.create_dataset(
                'k_max',
                shape=(total_images,),
                maxshape=(None,),
                dtype=np.float32,
                compression=self.compression,
                compression_opts=self.compression_opts,
                chunks=param_chunks
            )
            
            params_group.create_dataset(
                'beta',
                shape=(total_images,),
                maxshape=(None,),
                dtype=np.float32,
                compression=self.compression,
                compression_opts=self.compression_opts,
                chunks=param_chunks
            )
            
            # Distribution parameters
            params_group.create_dataset(
                'mean',
                shape=(total_images,),
                maxshape=(None,),
                dtype=np.float32,
                compression=self.compression,
                compression_opts=self.compression_opts,
                chunks=param_chunks
            )
            
            params_group.create_dataset(
                'sigma',
                shape=(total_images,),
                maxshape=(None,),
                dtype=np.float32,
                compression=self.compression,
                compression_opts=self.compression_opts,
                chunks=param_chunks
            )
            
            # Grid dimensions
            params_group.create_dataset(
                'ni',
                shape=(total_images,),
                maxshape=(None,),
                dtype=np.int32,
                compression=self.compression,
                compression_opts=self.compression_opts,
                chunks=param_chunks
            )
            
            params_group.create_dataset(
                'nj',
                shape=(total_images,),
                maxshape=(None,),
                dtype=np.int32,
                compression=self.compression,
                compression_opts=self.compression_opts,
                chunks=param_chunks
            )
            
            params_group.create_dataset(
                'nk',
                shape=(total_images,),
                maxshape=(None,),
                dtype=np.int32,
                compression=self.compression,
                compression_opts=self.compression_opts,
                chunks=param_chunks
            )
            
            # Per-image random seed (for reproducibility)
            params_group.create_dataset(
                'seed',
                shape=(total_images,),
                maxshape=(None,),
                dtype=np.int64,
                compression=self.compression,
                compression_opts=self.compression_opts,
                chunks=param_chunks
            )
            
            # Store global metadata as attributes
            if self.store_parameters:
                f.attrs['method'] = self.method
                f.attrs['image_dimensions'] = (self.image_width, self.image_height, self.image_depth)
                f.attrs['batch_size'] = self.batch_size
                f.attrs['total_images'] = total_images
                
            if self.store_creation_date:
                f.attrs['creation_date'] = datetime.now().isoformat()
                f.attrs['config_file'] = self.config_file
            
            # Add descriptions for clarity
            params_group.attrs['description'] = 'Per-image generation parameters for turbulence analysis'
            f['images'].attrs['description'] = 'Log10-transformed fractal density slices'
            f['images'].attrs['transform'] = 'log10(density_slice)'

        self.log(f'HDF5 structure created: {self.output_file} ({total_images} images)', 'INFO')
    
    def single_image(self, k_min: float, k_max: Optional[float] = None, 
                    beta: Optional[float] = None, mean: Optional[float] = None,
                    sigma: Optional[float] = None, ni: Optional[int] = None,
                    nj: Optional[int] = None, nk: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Generate a single fractal image with specified parameters
        
        Args:
            k_min: Lower cutoff wavenumber (kmin)
            k_max: Upper cutoff wavenumber (optional, defaults to Nyquist limit)
            beta: Power spectrum slope (optional, defaults to class value)
            mean: Mean of lognormal distribution (optional, defaults to class value)
            sigma: Std of lognormal distribution (optional, defaults to class value)
            ni, nj, nk: Grid dimensions (optional, default to class values)
            
        Returns:
            Tuple of (image_array, parameters_dict)
            - image_array: 2D numpy array containing log-transformed fractal density slice
            - parameters_dict: Dictionary of actual parameters used for generation
            
        Raises:
            RuntimeError: If fractal cube generation fails
        """
        # Use defaults if not specified
        if beta is None: beta = self.beta
        if mean is None: mean = self.mean
        if sigma is None: sigma = self.sigma
        if ni is None: ni = self.ni
        if nj is None: nj = self.nj
        if nk is None: nk = self.nk
        
        try:
            # Create the Log Normal Fractal Cube
            fc = pyFC.LogNormalFractalCube(
                ni=ni, 
                nj=nj, 
                nk=nk,
                kmin=k_min,
                kmax=k_max,
                mean=mean,
                sigma=sigma,
                beta=beta
            )
            
            # Generate the cube with per-image seed
            if self.seed_mode == 'random':
                seed = int(np.random.randint(0, 2**31))
            elif isinstance(self.seed_mode, int):
                seed = int(self.seed_mode)
            else:
                seed = 0

            _gen_cube_compat(fc, verbose=self.verbose, seed=(seed if seed != 0 else None))
            
            # Extract density slice
            fcs = pyFC.FCSlicer()
            slice_data = fcs.slice(fc, ax=2)
            
            # Apply log10 transformation
            slice_log = np.log10(slice_data)
            
            # Validate if configured
            if self.check_finite_values:
                finite_count = np.isfinite(slice_log).sum()
                if finite_count != slice_log.size:
                    warning_msg = f"Warning: {slice_log.size - finite_count} non-finite values in image for k_min={k_min}"
                    self.log(warning_msg, 'WARNING')
            
            # Store actual parameters used (including k_max which may be auto-set)
            actual_k_max = fc.kmax
            actual_seed = seed if self.seed_mode == 'random' else (self.seed_mode if isinstance(self.seed_mode, int) else 0)
            params = {
                'k_min': float(k_min),
                'k_max': float(actual_k_max),
                'beta': float(beta),
                'mean': float(mean),
                'sigma': float(sigma),
                'ni': int(ni),
                'nj': int(nj),
                'nk': int(nk),
                'seed': int(actual_seed)
            }
            
            # Clear memory if configured
            if self.clear_memory_after_batch:
                del fc, slice_data
                if self.force_garbage_collection:
                    gc.collect()
            
            return slice_log, params
            
        except Exception as e:
            error_msg = f"Failed to generate image for k_min={k_min}: {str(e)}"
            self.log(error_msg, 'ERROR')
            raise RuntimeError(error_msg) from e
    
    
    def populate_dataset(self):
        """
        Generate the complete dataset with parameter tracking
        
        Main method that orchestrates the entire dataset generation process.
        Supports both serial and parallel processing modes with memory management.
        
        Raises:
            RuntimeError: If dataset population fails
        """
        if self.num_workers > 1:
            return self._populate_dataset_parallel()
        else:
            return self._populate_dataset_serial()
    
    def _populate_dataset_serial(self):
        """
        Serial dataset population (single process)
        Optimized for memory management with batch processing.
        """
        try:
            if self.progress_reporting:
                print(f'Initializing dataset population (SERIAL mode) into {self.output_file}')
                if self.memory_monitoring:
                    print(f'Initial RAM Usage: {self.get_memory_usage():.2f} MB')
            
            self.log(f'Starting dataset population (serial): {self.output_file}', 'INFO')
            if self.memory_monitoring:
                self.log(f'Initial RAM: {self.get_memory_usage():.2f} MB', 'INFO')
            
            # Calculate total images (combinations * images per combination)
            total_images = len(self.param_combinations) * self.num_images
            
            # Create HDF5 structure
            self.create_hdf5(total_images)
            
            start_time = datetime.now()
            global_image_idx = 0
            
            for combo_idx, params in enumerate(self.param_combinations):
                k_min_value = params['k_min']
                k_max_value = params['k_max']
                beta = params['beta']
                mean = params['mean']
                sigma = params['sigma']
                
                k_max_str = 'auto' if k_max_value is None else str(k_max_value)
                if self.progress_reporting:
                    print(f'\nProcessing combination {combo_idx+1}/{len(self.param_combinations)}: '
                          f'k_min={k_min_value}, k_max={k_max_str}, β={beta:.3f}, μ={mean:.2f}, σ={sigma:.3f}')
                
                self.log(f'Processing combination {combo_idx+1}: k_min={k_min_value}, k_max={k_max_str}, β={beta:.3f}, μ={mean}, σ={sigma}', 'INFO')
                
                # Calculate batches for memory management
                num_batches = (self.num_images + self.batch_size - 1) // self.batch_size
                
                with h5py.File(self.output_file, 'a') as f:
                    images_ds = f['images']
                    params_group = f['parameters']
                    
                    for batch_idx in range(num_batches):
                        start_idx = batch_idx * self.batch_size
                        end_idx = min(start_idx + self.batch_size, self.num_images)
                        current_batch_size = end_idx - start_idx
                        
                        if self.progress_reporting and self.verbose:
                            print(f'  Batch {batch_idx+1}/{num_batches} | Images {start_idx} to {end_idx-1}')
                        
                        # Generate batch
                        batch_images = []
                        batch_params = {
                            'k_min': [], 'k_max': [], 'beta': [], 
                            'mean': [], 'sigma': [],
                            'ni': [], 'nj': [], 'nk': [], 'seed': []
                        }
                        
                        for i in range(current_batch_size):
                            # Generate image with current parameter combination
                            img, img_params = self.single_image(
                                k_min=k_min_value,
                                k_max=k_max_value,
                                beta=beta, 
                                mean=mean, 
                                sigma=sigma
                            )
                            batch_images.append(img)
                            
                            # Store parameters
                            for key in batch_params.keys():
                                batch_params[key].append(img_params[key])
                            
                            # Clear memory per image if needed
                            if self.clear_memory_after_batch:
                                del img
                                if self.force_garbage_collection:
                                    gc.collect()
                        
                        # Convert to arrays
                        batch_images = np.array(batch_images, dtype=np.float32)
                        
                        # Write to HDF5
                        global_end = global_image_idx + current_batch_size
                        images_ds[global_image_idx:global_end] = batch_images
                        
                        # Write parameters
                        for key, values in batch_params.items():
                            params_group[key][global_image_idx:global_end] = values
                        
                        global_image_idx += current_batch_size
                        
                        # Clear memory after batch write
                        if self.clear_memory_after_batch:
                            del batch_images, batch_params
                            if self.force_garbage_collection:
                                gc.collect()
                        
                        # Progress reporting
                        if self.progress_reporting and (batch_idx % self.progress_report_interval == 0 or batch_idx == num_batches - 1):
                            elapsed = datetime.now() - start_time
                            rate = global_image_idx / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
                            eta = (total_images - global_image_idx) / rate if rate > 0 else 0
                            
                            progress_msg = f'Progress: {global_image_idx}/{total_images} ({100*global_image_idx/total_images:.1f}%) | {rate:.1f} img/s | ETA: {eta:.0f}s'
                            if self.memory_monitoring:
                                mem_current = self.get_memory_usage()
                                progress_msg += f' | RAM: {mem_current:.2f} MB'
                            print(progress_msg)
                        
                        # Log batch completion
                        self.log(f'Batch {batch_idx+1}/{num_batches} completed for k_min={k_min_value}, k_max={k_max_str}', 'DEBUG')
            
            # Final summary
            total_time = datetime.now() - start_time
            file_size = os.path.getsize(self.output_file) / 1024**2
            
            if self.progress_reporting:
                print(f'\n{"="*70}')
                print(f'Dataset generation completed in {total_time}')
                print(f'Total Images: {global_image_idx}')
                print(f'Output File Size: {file_size:.2f} MB ({file_size/1024:.3f} GB)')
                print(f'Average Speed: {global_image_idx / total_time.total_seconds():.2f} img/s')
                if self.memory_monitoring:
                    print(f'Final RAM Usage: {self.get_memory_usage():.2f} MB')
                print(f'{"="*70}')
            
            # Log completion
            self.log(f"Dataset generation completed in {total_time}", 'INFO')
            self.log(f"Total images: {global_image_idx}, File size: {file_size:.2f} MB", 'INFO')
            self.log(f"Average speed: {global_image_idx / total_time.total_seconds():.2f} img/s", 'INFO')
            if self.memory_monitoring:
                self.log(f"Final RAM: {self.get_memory_usage():.2f} MB", 'INFO')
                
        except Exception as e:
            error_msg = f"Dataset population failed: {str(e)}"
            self.log(error_msg, 'ERROR')
            raise RuntimeError(error_msg) from e
    
    def _populate_dataset_parallel(self):
        """
        Parallel dataset population using multiprocessing
        Generates images in parallel, writes to HDF5 sequentially in batches.
        """
        try:
            if self.progress_reporting:
                print(f'Initializing dataset population (PARALLEL mode: {self.num_workers} workers) into {self.output_file}')
                if self.memory_monitoring:
                    print(f'Initial RAM Usage: {self.get_memory_usage():.2f} MB')
            
            self.log(f'Starting dataset population (parallel, {self.num_workers} workers): {self.output_file}', 'INFO')
            if self.memory_monitoring:
                self.log(f'Initial RAM: {self.get_memory_usage():.2f} MB', 'INFO')
            
            # Calculate total images (combinations * images per combination)
            total_images = len(self.param_combinations) * self.num_images
            
            # Create HDF5 structure
            self.create_hdf5(total_images)
            
            start_time = datetime.now()
            global_image_idx = 0
            
            for combo_idx, params in enumerate(self.param_combinations):
                k_min_value = params['k_min']
                k_max_value = params['k_max']
                beta = params['beta']
                mean = params['mean']
                sigma = params['sigma']
                
                k_max_str = 'auto' if k_max_value is None else str(k_max_value)
                if self.progress_reporting:
                    print(f'\nProcessing combination {combo_idx+1}/{len(self.param_combinations)}: '
                          f'k_min={k_min_value}, k_max={k_max_str}, β={beta:.3f}, μ={mean:.2f}, σ={sigma:.3f}')
                
                self.log(f'Processing combination {combo_idx+1}: k_min={k_min_value}, k_max={k_max_str}, β={beta:.3f}, μ={mean}, σ={sigma}', 'INFO')
                
                # Calculate batches for memory management
                num_batches = (self.num_images + self.batch_size - 1) // self.batch_size
                
                with h5py.File(self.output_file, 'a') as f:
                    images_ds = f['images']
                    params_group = f['parameters']
                    
                    # Use multiprocessing pool
                    with mp.Pool(processes=self.num_workers) as pool:
                        for batch_idx in range(num_batches):
                            start_idx = batch_idx * self.batch_size
                            end_idx = min(start_idx + self.batch_size, self.num_images)
                            current_batch_size = end_idx - start_idx
                            
                            if self.progress_reporting and self.verbose:
                                print(f'  Batch {batch_idx+1}/{num_batches} | Images {start_idx} to {end_idx-1}')
                            
                            # Generate per-image seeds
                            if self.seed_mode == 'random':
                                seeds = [int(np.random.randint(0, 2**31)) for _ in range(current_batch_size)]
                            elif isinstance(self.seed_mode, int):
                                seeds = [self.seed_mode] * current_batch_size
                            else:
                                seeds = [None] * current_batch_size
                            
                            # Build argument tuples for starmap
                            work_args = [
                                (k_min_value, k_max_value, self.ni, self.nj, self.nk,
                                 mean, sigma, beta, False, seeds[i])
                                for i in range(current_batch_size)
                            ]
                            results = pool.starmap(_generate_image_worker, work_args)
                            
                            # Unpack results
                            batch_images = []
                            batch_params = {
                                'k_min': [], 'k_max': [], 'beta': [], 
                                'mean': [], 'sigma': [],
                                'ni': [], 'nj': [], 'nk': [], 'seed': []
                            }
                            
                            for img, params in results:
                                batch_images.append(img)
                                for key in batch_params.keys():
                                    batch_params[key].append(params[key])
                            
                            # Convert to arrays
                            batch_images = np.array(batch_images, dtype=np.float32)
                            
                            # Write to HDF5 (sequential, HDF5 is not thread-safe)
                            global_end = global_image_idx + current_batch_size
                            images_ds[global_image_idx:global_end] = batch_images
                            
                            # Write parameters
                            for key, values in batch_params.items():
                                params_group[key][global_image_idx:global_end] = values
                            
                            global_image_idx += current_batch_size
                            
                            # Clear memory after batch
                            if self.clear_memory_after_batch:
                                del batch_images, batch_params, results
                                if self.force_garbage_collection:
                                    gc.collect()
                            
                            # Progress reporting
                            if self.progress_reporting and (batch_idx % self.progress_report_interval == 0 or batch_idx == num_batches - 1):
                                elapsed = datetime.now() - start_time
                                rate = global_image_idx / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
                                eta = (total_images - global_image_idx) / rate if rate > 0 else 0
                                
                                progress_msg = f'Progress: {global_image_idx}/{total_images} ({100*global_image_idx/total_images:.1f}%) | {rate:.1f} img/s | ETA: {eta:.0f}s'
                                if self.memory_monitoring:
                                    mem_current = self.get_memory_usage()
                                    progress_msg += f' | RAM: {mem_current:.2f} MB'
                                print(progress_msg)
                            
                            # Log batch completion
                            self.log(f'Batch {batch_idx+1}/{num_batches} completed for k_min={k_min_value}, k_max={k_max_str}', 'DEBUG')
            
            # Final summary
            total_time = datetime.now() - start_time
            file_size = os.path.getsize(self.output_file) / 1024**2
            
            if self.progress_reporting:
                print(f'\n{"="*70}')
                print(f'Dataset generation completed in {total_time}')
                print(f'Total Images: {global_image_idx}')
                print(f'Output File Size: {file_size:.2f} MB ({file_size/1024:.3f} GB)')
                print(f'Average Speed: {global_image_idx / total_time.total_seconds():.2f} img/s')
                print(f'Parallel Speedup: ~{self.num_workers}x theoretical')
                if self.memory_monitoring:
                    print(f'Final RAM Usage: {self.get_memory_usage():.2f} MB')
                print(f'{"="*70}')
            
            # Log completion
            self.log(f"Dataset generation completed in {total_time}", 'INFO')
            self.log(f"Total images: {global_image_idx}, File size: {file_size:.2f} MB", 'INFO')
            self.log(f"Average speed: {global_image_idx / total_time.total_seconds():.2f} img/s", 'INFO')
            if self.memory_monitoring:
                self.log(f"Final RAM: {self.get_memory_usage():.2f} MB", 'INFO')
                
        except Exception as e:
            error_msg = f"Dataset population failed (parallel): {str(e)}"
            self.log(error_msg, 'ERROR')
            raise RuntimeError(error_msg) from e
    
    # --------------------------------------------------------------------------------
    # MERGE / APPEND METHODS
    # --------------------------------------------------------------------------------
    
    @staticmethod
    def _detect_layout(filepath: str) -> str:
        """
        Detect the HDF5 layout of a dataset file.
        
        Args:
            filepath: Path to HDF5 dataset file
            
        Returns:
            'flat' if root contains /images dataset,
            'grouped' if root contains dimension-named groups (e.g., /512x512/)
            
        Raises:
            ValueError: If layout cannot be determined
        """
        with h5py.File(filepath, 'r') as f:
            if 'images' in f and isinstance(f['images'], h5py.Dataset):
                return 'flat'
            for key in f.keys():
                if 'x' in key:
                    parts = key.split('x')
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        return 'grouped'
            raise ValueError(f"Cannot determine HDF5 layout for: {filepath}")
    
    @staticmethod
    def _detect_duplicates(source_params: Dict[str, np.ndarray],
                           target_params: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Detect duplicate images between source and target based on parameter tuples.
        
        Args:
            source_params: Dict of parameter arrays from the source file
            target_params: Dict of parameter arrays already in the target
            
        Returns:
            Boolean mask for source (True = unique, should be kept)
        """
        compare_keys = ['k_min', 'k_max', 'beta', 'mean', 'sigma', 'seed']
        
        n_source = len(source_params[compare_keys[0]])
        if n_source == 0:
            return np.array([], dtype=bool)
        
        n_target = len(target_params[compare_keys[0]])
        if n_target == 0:
            return np.ones(n_source, dtype=bool)
        
        target_tuples = set()
        for i in range(n_target):
            t = tuple(round(float(target_params[k][i]), 6) for k in compare_keys)
            target_tuples.add(t)
        
        mask = np.ones(n_source, dtype=bool)
        for i in range(n_source):
            s = tuple(round(float(source_params[k][i]), 6) for k in compare_keys)
            if s in target_tuples:
                mask[i] = False
        
        return mask
    
    @staticmethod
    def _read_file_data(filepath: str) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Read all data from an HDF5 file, organized by dimension group.
        
        Args:
            filepath: Path to HDF5 file
            
        Returns:
            Dict keyed by dimension string (e.g., '512x512'), each containing
            'images' array and parameter arrays.
        """
        layout = DatasetGen._detect_layout(filepath)
        result = {}
        param_names = ['k_min', 'k_max', 'beta', 'mean', 'sigma', 'ni', 'nj', 'nk', 'seed']
        
        with h5py.File(filepath, 'r') as f:
            if layout == 'flat':
                images = f['images'][:]
                h, w = images.shape[1], images.shape[2]
                dim_key = f'{h}x{w}'
                group_data = {'images': images}
                for p in param_names:
                    if p in f['parameters']:
                        group_data[p] = f['parameters'][p][:]
                    else:
                        # Backward compat: old files without seed
                        group_data[p] = np.zeros(images.shape[0], dtype=np.int64)
                result[dim_key] = group_data
            else:
                for key in f.keys():
                    if 'x' in key:
                        parts = key.split('x')
                        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                            grp = f[key]
                            group_data = {'images': grp['images'][:]}
                            for p in param_names:
                                if p in grp['parameters']:
                                    group_data[p] = grp['parameters'][p][:]
                                else:
                                    group_data[p] = np.zeros(group_data['images'].shape[0], dtype=np.int64)
                            result[key] = group_data
        
        return result
    
    @staticmethod
    def merge_datasets(file_list: List[str], output_file: str, 
                       dup_check: bool = True, compression: str = 'gzip',
                       compression_opts: int = 1) -> str:
        """
        Merge multiple HDF5 dataset files into one file with group-based layout.
        
        Each unique image dimension gets its own group (e.g., /512x512/images).
        Supports merging files with different image dimensions.
        
        Args:
            file_list: List of paths to HDF5 dataset files to merge
            output_file: Path for the merged output file
            dup_check: If True, skip duplicate images (same parameter tuple)
            compression: Compression algorithm for output file
            compression_opts: Compression level
            
        Returns:
            Path to the merged output file
            
        Raises:
            FileNotFoundError: If any input file doesn't exist
            ValueError: If any input file has an unrecognizable layout
        """
        for fp in file_list:
            if not os.path.exists(fp):
                raise FileNotFoundError(f"Input file not found: {fp}")
        
        all_data = {}
        param_names = ['k_min', 'k_max', 'beta', 'mean', 'sigma', 'ni', 'nj', 'nk', 'seed']
        
        for fp in file_list:
            file_data = DatasetGen._read_file_data(fp)
            for dim_key, group_data in file_data.items():
                if dim_key not in all_data:
                    all_data[dim_key] = {
                        'images': [],
                        **{p: [] for p in param_names}
                    }
                all_data[dim_key]['images'].append(group_data['images'])
                for p in param_names:
                    all_data[dim_key][p].append(group_data[p])
        
        # Concatenate arrays per dimension group
        for dim_key in all_data:
            all_data[dim_key]['images'] = np.concatenate(all_data[dim_key]['images'], axis=0)
            for p in param_names:
                all_data[dim_key][p] = np.concatenate(all_data[dim_key][p], axis=0)
        
        # Duplicate detection per dimension group
        if dup_check:
            for dim_key in all_data:
                group = all_data[dim_key]
                n_before = group['images'].shape[0]
                
                compare_keys = ['k_min', 'k_max', 'beta', 'mean', 'sigma', 'seed']
                seen = set()
                mask = np.ones(n_before, dtype=bool)
                for i in range(n_before):
                    t = tuple(round(float(group[k][i]), 6) for k in compare_keys)
                    if t in seen:
                        mask[i] = False
                    else:
                        seen.add(t)
                
                n_after = mask.sum()
                if n_after < n_before:
                    print(f"[{dim_key}] Removed {n_before - n_after} duplicates "
                          f"({n_before} -> {n_after} images)")
                    group['images'] = group['images'][mask]
                    for p in param_names:
                        group[p] = group[p][mask]
        
        # Write merged output with group-based layout
        with h5py.File(output_file, 'w') as f:
            total_all = 0
            for dim_key, group_data in all_data.items():
                dim_group = f.create_group(dim_key)
                n_images = group_data['images'].shape[0]
                h, w = group_data['images'].shape[1], group_data['images'].shape[2]
                total_all += n_images
                
                dim_group.create_dataset(
                    'images',
                    data=group_data['images'],
                    maxshape=(None, h, w),
                    dtype=np.float32,
                    compression=compression,
                    compression_opts=compression_opts,
                    chunks=(1, h, w)
                )
                
                params_grp = dim_group.create_group('parameters')
                float_params = ['k_min', 'k_max', 'beta', 'mean', 'sigma']
                int_params = ['ni', 'nj', 'nk']
                long_params = ['seed']
                
                for p in float_params:
                    params_grp.create_dataset(
                        p, data=group_data[p],
                        maxshape=(None,),
                        dtype=np.float32,
                        compression=compression,
                        compression_opts=compression_opts
                    )
                for p in int_params:
                    params_grp.create_dataset(
                        p, data=group_data[p],
                        maxshape=(None,),
                        dtype=np.int32,
                        compression=compression,
                        compression_opts=compression_opts
                    )
                for p in long_params:
                    params_grp.create_dataset(
                        p, data=group_data[p],
                        maxshape=(None,),
                        dtype=np.int64,
                        compression=compression,
                        compression_opts=compression_opts
                    )
                
                dim_group.attrs['num_images'] = n_images
                dim_group.attrs['image_shape'] = (h, w)
                dim_group['images'].attrs['description'] = 'Log10-transformed fractal density slices'
                params_grp.attrs['description'] = 'Per-image generation parameters'
            
            # Global metadata
            f.attrs['layout'] = 'grouped'
            f.attrs['merge_date'] = datetime.now().isoformat()
            f.attrs['source_files'] = [os.path.basename(fp) for fp in file_list]
            f.attrs['total_images'] = total_all
            f.attrs['dimension_groups'] = list(all_data.keys())
        
        print(f"Merged {len(file_list)} files -> {output_file}")
        for dim_key, group_data in all_data.items():
            print(f"  [{dim_key}]: {group_data['images'].shape[0]} images")
        print(f"  Total: {total_all} images")
        
        return output_file
    
    def append_dataset(self, other_file: str, dup_check: bool = True) -> None:
        """
        Append data from another HDF5 file into this instance's output file.
        
        Converts target to group-based layout if it currently uses flat layout.
        Creates new dimension groups if the source has dimensions not in the target.
        
        Args:
            other_file: Path to HDF5 file to append from
            dup_check: If True, skip duplicate images (same parameter tuple)
            
        Raises:
            FileNotFoundError: If other_file or self.output_file doesn't exist
        """
        if not os.path.exists(other_file):
            raise FileNotFoundError(f"Source file not found: {other_file}")
        if not os.path.exists(self.output_file):
            raise FileNotFoundError(f"Target file not found: {self.output_file}")
        
        param_names = ['k_min', 'k_max', 'beta', 'mean', 'sigma', 'ni', 'nj', 'nk', 'seed']
        float_params = ['k_min', 'k_max', 'beta', 'mean', 'sigma']
        int_params = ['ni', 'nj', 'nk']
        long_params = ['seed']
        
        source_data = DatasetGen._read_file_data(other_file)
        target_layout = DatasetGen._detect_layout(self.output_file)
        
        if target_layout == 'flat':
            # Convert flat to grouped by merging both files
            self.log("Converting target from flat to grouped layout", 'INFO')
            print(f"Converting {self.output_file} from flat to grouped layout...")
            
            all_files = [self.output_file, other_file]
            tmp_output = self.output_file + '.tmp'
            DatasetGen.merge_datasets(all_files, tmp_output, dup_check=dup_check,
                                      compression=self.compression,
                                      compression_opts=self.compression_opts)
            os.replace(tmp_output, self.output_file)
            self.log(f"Appended {other_file} (converted to grouped layout)", 'INFO')
            return
        
        # Target is already grouped — append in-place
        with h5py.File(self.output_file, 'a') as f:
            for dim_key, src_group in source_data.items():
                src_images = src_group['images']
                src_params = {p: src_group[p] for p in param_names}
                
                if dim_key in f:
                    grp = f[dim_key]
                    existing_n = grp['images'].shape[0]
                    
                    if dup_check:
                        target_params = {}
                        for p in param_names:
                            if p in grp['parameters']:
                                target_params[p] = grp['parameters'][p][:]
                            else:
                                target_params[p] = np.zeros(existing_n, dtype=np.int64)
                        mask = DatasetGen._detect_duplicates(src_params, target_params)
                        n_dupes = (~mask).sum()
                        if n_dupes > 0:
                            print(f"[{dim_key}] Skipping {n_dupes} duplicate images")
                            self.log(f"[{dim_key}] Skipping {n_dupes} duplicates", 'INFO')
                        src_images = src_images[mask]
                        src_params = {p: src_params[p][mask] for p in param_names}
                    
                    new_n = src_images.shape[0]
                    if new_n == 0:
                        print(f"[{dim_key}] No new images to append")
                        continue
                    
                    total_n = existing_n + new_n
                    grp['images'].resize(total_n, axis=0)
                    grp['images'][existing_n:total_n] = src_images
                    
                    for p in param_names:
                        if p in grp['parameters']:
                            grp['parameters'][p].resize(total_n, axis=0)
                            grp['parameters'][p][existing_n:total_n] = src_params[p]
                        else:
                            # Add missing param (e.g., seed on old files)
                            grp['parameters'].create_dataset(
                                p, data=np.concatenate([np.zeros(existing_n, dtype=np.int64), src_params[p]]),
                                maxshape=(None,), dtype=np.int64,
                                compression=self.compression, compression_opts=self.compression_opts
                            )
                    
                    grp.attrs['num_images'] = total_n
                    print(f"[{dim_key}] Appended {new_n} images ({existing_n} -> {total_n})")
                    self.log(f"[{dim_key}] Appended {new_n} images", 'INFO')
                    
                else:
                    # Deduplicate within source for new group
                    if dup_check:
                        compare_keys = ['k_min', 'k_max', 'beta', 'mean', 'sigma', 'seed']
                        seen = set()
                        mask = np.ones(src_images.shape[0], dtype=bool)
                        for i in range(src_images.shape[0]):
                            t = tuple(round(float(src_params[k][i]), 6) for k in compare_keys)
                            if t in seen:
                                mask[i] = False
                            else:
                                seen.add(t)
                        src_images = src_images[mask]
                        src_params = {p: src_params[p][mask] for p in param_names}
                    
                    n_images = src_images.shape[0]
                    h, w = src_images.shape[1], src_images.shape[2]
                    
                    dim_group = f.create_group(dim_key)
                    dim_group.create_dataset(
                        'images', data=src_images,
                        maxshape=(None, h, w),
                        dtype=np.float32,
                        compression=self.compression,
                        compression_opts=self.compression_opts,
                        chunks=(1, h, w)
                    )
                    
                    params_grp = dim_group.create_group('parameters')
                    for p in float_params:
                        params_grp.create_dataset(
                            p, data=src_params[p],
                            maxshape=(None,),
                            dtype=np.float32,
                            compression=self.compression,
                            compression_opts=self.compression_opts
                        )
                    for p in int_params:
                        params_grp.create_dataset(
                            p, data=src_params[p],
                            maxshape=(None,),
                            dtype=np.int32,
                            compression=self.compression,
                            compression_opts=self.compression_opts
                        )
                    for p in long_params:
                        params_grp.create_dataset(
                            p, data=src_params[p],
                            maxshape=(None,),
                            dtype=np.int64,
                            compression=self.compression,
                            compression_opts=self.compression_opts
                        )
                    
                    dim_group.attrs['num_images'] = n_images
                    dim_group.attrs['image_shape'] = (h, w)
                    dim_group['images'].attrs['description'] = 'Log10-transformed fractal density slices'
                    params_grp.attrs['description'] = 'Per-image generation parameters'
                    
                    print(f"[{dim_key}] Created new group with {n_images} images")
                    self.log(f"[{dim_key}] Created new group with {n_images} images", 'INFO')
            
            f.attrs['layout'] = 'grouped'
            f.attrs['last_append_date'] = datetime.now().isoformat()
        
        self.log(f"Append from {other_file} completed", 'INFO')
    
    @staticmethod
    def load_dataset(filepath: str, dimension: Optional[str] = None) -> Dict[str, Any]:
        """
        Load complete dataset from HDF5 file (auto-detects flat or grouped layout)
        
        Args:
            filepath: Path to HDF5 dataset file
            dimension: For grouped layout, load only this dimension group (e.g., '512x512').
                       If None and grouped, returns dict keyed by dimension.
            
        Returns:
            Flat layout: {'images': array, 'k_min': array, ...}
            Grouped layout (no dimension): {'512x512': {'images': ..., 'k_min': ...}, ...}
            Grouped layout (with dimension): {'images': array, 'k_min': array, ...}
        """
        layout = DatasetGen._detect_layout(filepath)
        
        if layout == 'flat':
            data = {}
            with h5py.File(filepath, 'r') as f:
                data['images'] = f['images'][:]
                for param_name in f['parameters'].keys():
                    data[param_name] = f['parameters'][param_name][:]
            return data
        else:
            with h5py.File(filepath, 'r') as f:
                if dimension is not None:
                    if dimension not in f:
                        raise KeyError(f"Dimension group '{dimension}' not found. "
                                       f"Available: {[k for k in f.keys() if 'x' in k]}")
                    grp = f[dimension]
                    data = {'images': grp['images'][:]}
                    for param_name in grp['parameters'].keys():
                        data[param_name] = grp['parameters'][param_name][:]
                    return data
                else:
                    all_data = {}
                    for key in f.keys():
                        if 'x' in key:
                            parts = key.split('x')
                            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                                grp = f[key]
                                group_data = {'images': grp['images'][:]}
                                for param_name in grp['parameters'].keys():
                                    group_data[param_name] = grp['parameters'][param_name][:]
                                all_data[key] = group_data
                    return all_data
    
    @staticmethod
    def load_subset(filepath: str, indices: np.ndarray, 
                    dimension: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Load a subset of the dataset by indices
        
        Args:
            filepath: Path to HDF5 dataset file
            indices: Array of indices to load
            dimension: Required for grouped layout — specifies which dimension group
            
        Returns:
            Dictionary containing subset of images and parameters
        """
        layout = DatasetGen._detect_layout(filepath)
        
        if layout == 'flat':
            data = {}
            with h5py.File(filepath, 'r') as f:
                data['images'] = f['images'][indices]
                for param_name in f['parameters'].keys():
                    data[param_name] = f['parameters'][param_name][indices]
            return data
        else:
            if dimension is None:
                raise ValueError("'dimension' parameter is required for grouped layout "
                                 "(e.g., dimension='512x512')")
            with h5py.File(filepath, 'r') as f:
                if dimension not in f:
                    raise KeyError(f"Dimension group '{dimension}' not found. "
                                   f"Available: {[k for k in f.keys() if 'x' in k]}")
                grp = f[dimension]
                data = {'images': grp['images'][indices]}
                for param_name in grp['parameters'].keys():
                    data[param_name] = grp['parameters'][param_name][indices]
                return data
    
    @staticmethod
    def get_dataset_info(filepath: str) -> Dict[str, Any]:
        """
        Get comprehensive information about dataset structure and contents.
        Auto-detects flat or grouped layout.
        
        Args:
            filepath: Path to HDF5 dataset file
            
        Returns:
            Dictionary with dataset statistics and metadata
        """
        layout = DatasetGen._detect_layout(filepath)
        info = {'layout': layout}
        
        file_size = os.path.getsize(filepath)
        info['file_size_mb'] = file_size / (1024**2)
        info['file_size_gb'] = file_size / (1024**3)
        
        with h5py.File(filepath, 'r') as f:
            info['metadata'] = dict(f.attrs)
            
            if layout == 'flat':
                info['total_images'] = f['images'].shape[0]
                info['image_shape'] = f['images'].shape[1:]
                info['dtype'] = str(f['images'].dtype)
                
                info['parameters'] = {}
                for param_name in f['parameters'].keys():
                    param_data = f['parameters'][param_name][:]
                    info['parameters'][param_name] = {
                        'min': float(np.min(param_data)),
                        'max': float(np.max(param_data)),
                        'mean': float(np.mean(param_data)),
                        'std': float(np.std(param_data)),
                        'unique_values': int(len(np.unique(param_data)))
                    }
            else:
                info['dimension_groups'] = {}
                total = 0
                for key in f.keys():
                    if 'x' in key:
                        parts = key.split('x')
                        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                            grp = f[key]
                            grp_info = {
                                'num_images': grp['images'].shape[0],
                                'image_shape': grp['images'].shape[1:],
                                'dtype': str(grp['images'].dtype),
                                'parameters': {}
                            }
                            total += grp['images'].shape[0]
                            for param_name in grp['parameters'].keys():
                                param_data = grp['parameters'][param_name][:]
                                grp_info['parameters'][param_name] = {
                                    'min': float(np.min(param_data)),
                                    'max': float(np.max(param_data)),
                                    'mean': float(np.mean(param_data)),
                                    'std': float(np.std(param_data)),
                                    'unique_values': int(len(np.unique(param_data)))
                                }
                            info['dimension_groups'][key] = grp_info
                info['total_images'] = total
        
        return info
    
    @staticmethod
    def print_dataset_summary(filepath: str):
        """
        Print a human-readable summary of the dataset.
        Auto-detects flat or grouped layout.
        
        Args:
            filepath: Path to HDF5 dataset file
        """
        info = DatasetGen.get_dataset_info(filepath)
        
        print("=" * 70)
        print("FRACTAL DATASET SUMMARY")
        print("=" * 70)
        print(f"File: {filepath}")
        print(f"Layout: {info['layout']}")
        print(f"Size: {info['file_size_mb']:.2f} MB ({info['file_size_gb']:.3f} GB)")
        print(f"Total Images: {info['total_images']:,}")
        
        print(f"\nGlobal Metadata:")
        for key, value in info['metadata'].items():
            print(f"  {key}: {value}")
        
        if info['layout'] == 'flat':
            print(f"\nShape: {info['image_shape']}")
            print(f"Dtype: {info['dtype']}")
            print(f"\nParameter Statistics:")
            print("-" * 70)
            print(f"{'Parameter':<12} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} {'Unique':>8}")
            print("-" * 70)
            for param, stats in info['parameters'].items():
                print(f"{param:<12} {stats['min']:>10.4f} {stats['max']:>10.4f} "
                      f"{stats['mean']:>10.4f} {stats['std']:>10.4f} {stats['unique_values']:>8}")
        else:
            for dim_key, grp_info in info['dimension_groups'].items():
                print(f"\n--- [{dim_key}] {grp_info['num_images']} images ---")
                print(f"Shape: {grp_info['image_shape']}, Dtype: {grp_info['dtype']}")
                print(f"{'Parameter':<12} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} {'Unique':>8}")
                print("-" * 70)
                for param, stats in grp_info['parameters'].items():
                    print(f"{param:<12} {stats['min']:>10.4f} {stats['max']:>10.4f} "
                          f"{stats['mean']:>10.4f} {stats['std']:>10.4f} {stats['unique_values']:>8}")
        print("=" * 70)
    
    def get_config_dict(self) -> Dict[str, Any]:
        """
        Return current configuration as dictionary
        
        Returns:
            Dictionary containing all configuration parameters organized by section
        """
        return {
            'dataset': {
                'output_file': self.output_file,
                'batch_size': self.batch_size,
                'num_images': self.num_images,
                'image_dimensions': (self.image_width, self.image_height, self.image_depth),
                'k_values': self.k_values
            },
            'distribution': {
                'mean': self.mean,
                'sigma': self.sigma,
                'beta': self.beta
            },
            'fractal': {
                'ni': self.ni, 'nj': self.nj, 'nk': self.nk,
                'verbose': self.verbose,
                'random_seed': str(self.seed_mode)
            },
            'processing': {
                'memory_monitoring': self.memory_monitoring,
                'progress_reporting': self.progress_reporting,
                'clear_memory_after_batch': self.clear_memory_after_batch
            },
            'logging': {
                'enable_logging': self.enable_logging,
                'log_file': self.log_file,
                'log_level': self.log_level
            }
        }
    
    def save_config(self, output_path: str):
        """
        Save current configuration to file
        
        Args:
            output_path: Path where configuration file will be saved
        """
        with open(output_path, 'w') as f:
            self.config.write(f)
        self.log(f"Configuration saved to: {output_path}", 'INFO')


# --------------------------------------------------------------------------------
# HELPER FUNCTIONS FOR DATASET ANALYSIS AND VALIDATION
# --------------------------------------------------------------------------------

def _validate_flat(f) -> bool:
    """Validate a flat-layout HDF5 file (internal helper)."""
    if 'images' not in f:
        print('ERROR: /images dataset not found!')
        return False
    if 'parameters' not in f:
        print('ERROR: /parameters group not found!')
        return False
    
    images = f['images']
    params_group = f['parameters']
    
    print(f'\nImages dataset:')
    print(f'  Shape: {images.shape}')
    print(f'  Dtype: {images.dtype}')
    print(f'  Compression: {images.compression}')
    print(f'  Chunks: {images.chunks}')
    
    total_images = images.shape[0]
    
    expected_params = ['k_min', 'k_max', 'beta', 'mean', 'sigma', 'ni', 'nj', 'nk', 'seed']
    print(f'\nParameters:')
    for param_name in expected_params:
        if param_name not in params_group:
            print(f'  ERROR: Parameter {param_name} not found!')
            return False
        param_array = params_group[param_name]
        if param_array.shape[0] != total_images:
            print(f'  ERROR: Parameter {param_name} has wrong length: {param_array.shape[0]} vs {total_images}')
            return False
        print(f'  {param_name}: shape={param_array.shape}, dtype={param_array.dtype}')
    
    sample_indices = np.random.choice(total_images, min(10, total_images), replace=False)
    sample_indices = sorted(sample_indices.tolist())
    sample_images = np.array([images[i] for i in sample_indices])
    
    print(f'\nImage statistics (sample of {len(sample_images)}):')
    print(f'  Min: {np.nanmin(sample_images):.4f}')
    print(f'  Max: {np.nanmax(sample_images):.4f}')
    print(f'  Mean: {np.nanmean(sample_images):.4f}')
    print(f'  Std: {np.nanstd(sample_images):.4f}')
    print(f'  Finite count: {np.isfinite(sample_images).sum()}/{sample_images.size}')
    
    print(f'\nParameter statistics:')
    for param_name in expected_params:
        param_data = params_group[param_name][:]
        unique_vals = len(np.unique(param_data))
        print(f'  {param_name}: min={param_data.min():.2f}, max={param_data.max():.2f}, unique={unique_vals}')
    
    print(f'\nTotal Images in Dataset: {total_images}')
    return True


def validation(filepath: str) -> bool:
    """
    Validates the dataset structure and content.
    Auto-detects flat or grouped layout.
    
    Args:
        filepath: Path to HDF5 dataset file
        
    Returns:
        True if validation passes, False otherwise
    """
    if not os.path.exists(filepath):
        print(f'Filepath {filepath} does not exist.')
        return False
    
    try:
        layout = DatasetGen._detect_layout(filepath)
        
        with h5py.File(filepath, 'r') as f:
            print(f'Layout: {layout}')
            print(f'Root keys: {list(f.keys())}')
            
            if layout == 'flat':
                valid = _validate_flat(f)
            else:
                valid = True
                dim_groups_found = False
                for key in f.keys():
                    if 'x' in key:
                        parts = key.split('x')
                        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                            dim_groups_found = True
                            print(f'\n=== Dimension group: [{key}] ===')
                            grp = f[key]
                            if 'images' not in grp or 'parameters' not in grp:
                                print(f'  ERROR: Missing images or parameters in /{key}/')
                                valid = False
                                continue
                            valid = _validate_flat(grp) and valid
                
                if not dim_groups_found:
                    print('ERROR: No dimension groups found in grouped layout!')
                    valid = False
            
            if valid:
                print('\nDataset validation completed successfully.')
            else:
                print('\nDataset validation FAILED.')
            return valid
            
    except Exception as e:
        print(f'Error reading HDF5 file: {e}')
        import traceback
        traceback.print_exc()
        return False


def preview(filepath: str, samples_per_k: int = 2, max_k_values: int = 6):
    """
    Previews random samples from the dataset, grouped by k_min values
    
    Args:
        filepath: Path to HDF5 dataset file
        samples_per_k: Number of samples to show per k-value
        max_k_values: Maximum number of different k-values to show
    """
    import matplotlib.pyplot as plt
    
    with h5py.File(filepath, 'r') as f:
        images = f['images']
        k_min_values = f['parameters/k_min'][:]
        
        # Get unique k values
        unique_k = np.unique(k_min_values)
        unique_k = unique_k[:max_k_values]  # Limit number of k-values shown
        
        print(f'Found k_min values: {unique_k}')
        print(f'Total images: {images.shape[0]}')
        
        fig, axes = plt.subplots(len(unique_k), samples_per_k, 
                                figsize=(4*samples_per_k, 4*len(unique_k)))
        
        # Handle single row/column cases
        if len(unique_k) == 1 and samples_per_k == 1:
            axes = np.array([[axes]])
        elif len(unique_k) == 1:
            axes = axes.reshape(1, -1)
        elif samples_per_k == 1:
            axes = axes.reshape(-1, 1)
        
        for k_idx, k_val in enumerate(unique_k):
            # Find all images with this k_min
            k_indices = np.where(k_min_values == k_val)[0]
            
            for sample_idx in range(samples_per_k):
                ax = axes[k_idx, sample_idx]
                
                # Random sample from this k-value
                if len(k_indices) > 0:
                    img_idx = np.random.choice(k_indices)
                    img = images[img_idx]
                    
                    # Get parameters for this image
                    beta = f['parameters/beta'][img_idx]
                    k_max = f['parameters/k_max'][img_idx]
                    
                    # Display
                    im = ax.imshow(img, cmap='copper', 
                                vmin=np.nanpercentile(img, 1), 
                                vmax=np.nanpercentile(img, 99))
                    ax.set_title(f'k_min={int(k_val)}, k_max={int(k_max)}\nβ={beta:.2f}, idx={img_idx}')
                    ax.axis('off')
                else:
                    ax.text(0.5, 0.5, 'No images', ha='center', va='center')
                    ax.axis('off')
        
        plt.tight_layout()
        plt.show()


def analyze_dataset_diversity(filepath: str, n_samples: int = 50, by_k: bool = True):
    """
    Analyzes dataset diversity using various metrics
    
    Args:
        filepath: Path to HDF5 dataset file
        n_samples: Number of samples to analyze per k-value (if by_k=True) or total
        by_k: If True, analyze separately for each k-value; if False, analyze whole dataset
    """
    from sklearn.metrics.pairwise import cosine_similarity
    import hashlib
    import scipy.stats
    
    diversity_results = {}
    
    with h5py.File(filepath, 'r') as f:
        images_ds = f['images']
        k_min_values = f['parameters/k_min'][:]
        
        if by_k:
            unique_k = np.unique(k_min_values)
            
            for k_val in unique_k:
                k_indices = np.where(k_min_values == k_val)[0]
                total_images = len(k_indices)
                
                # Sample indices
                sample_size = min(n_samples, total_images)
                sample_idx = np.random.choice(k_indices, sample_size, replace=False)
                sample_idx = sorted(sample_idx.tolist())  # HDF5 requires sorted indices
                sample_images = np.array([images_ds[i] for i in sample_idx])
                
                # Hash uniqueness
                hashes = {hashlib.md5(img.tobytes()).hexdigest() for img in sample_images}
                hash_uniqueness = len(hashes) / len(sample_images)
                
                # Correlations
                correlations = [
                    abs(np.corrcoef(sample_images[i].flatten(), sample_images[j].flatten())[0, 1])
                    for i in range(min(20, len(sample_images)))
                    for j in range(i+1, min(20, len(sample_images)))
                ]
                avg_correlation = np.mean(correlations) if correlations else 0
                
                # Cosine similarity
                flat_images = sample_images.reshape(len(sample_images), -1)
                cos_sim = cosine_similarity(flat_images[:20])
                mask = np.triu(np.ones_like(cos_sim, dtype=bool), k=1)
                avg_cosine_sim = np.mean(np.abs(cos_sim[mask]))
                
                # Pixel differences
                pixel_diffs = [
                    np.mean(np.abs(sample_images[i] - sample_images[j]))
                    for i in range(min(10, len(sample_images)))
                    for j in range(i+1, min(10, len(sample_images)))
                ]
                avg_pixel_diff = np.mean(pixel_diffs) if pixel_diffs else 0
                
                diversity_results[int(k_val)] = {
                    'hash_uniqueness': hash_uniqueness,
                    'avg_correlation': avg_correlation,
                    'avg_cosine_similarity': avg_cosine_sim,
                    'avg_pixel_difference': avg_pixel_diff,
                    'n_samples': len(sample_images)
                }
                print(f"k={int(k_val)}: unique={hash_uniqueness:.2f}, correlation={avg_correlation:.3f}, "
                      f"cosine={avg_cosine_sim:.3f}, pixel_diff={avg_pixel_diff:.2f}")
        else:
            # Analyze whole dataset
            total_images = images_ds.shape[0]
            sample_size = min(n_samples, total_images)
            sample_idx = np.random.choice(total_images, sample_size, replace=False)
            sample_idx = sorted(sample_idx.tolist())  # HDF5 requires sorted indices
            sample_images = np.array([images_ds[i] for i in sample_idx])
            
            hashes = {hashlib.md5(img.tobytes()).hexdigest() for img in sample_images}
            hash_uniqueness = len(hashes) / len(sample_images)
            
            correlations = [
                abs(np.corrcoef(sample_images[i].flatten(), sample_images[j].flatten())[0, 1])
                for i in range(min(20, len(sample_images)))
                for j in range(i+1, min(20, len(sample_images)))
            ]
            avg_correlation = np.mean(correlations) if correlations else 0
            
            flat_images = sample_images.reshape(len(sample_images), -1)
            cos_sim = cosine_similarity(flat_images[:20])
            mask = np.triu(np.ones_like(cos_sim, dtype=bool), k=1)
            avg_cosine_sim = np.mean(np.abs(cos_sim[mask]))
            
            pixel_diffs = [
                np.mean(np.abs(sample_images[i] - sample_images[j]))
                for i in range(min(10, len(sample_images)))
                for j in range(i+1, min(10, len(sample_images)))
            ]
            avg_pixel_diff = np.mean(pixel_diffs) if pixel_diffs else 0
            
            diversity_results['all'] = {
                'hash_uniqueness': hash_uniqueness,
                'avg_correlation': avg_correlation,
                'avg_cosine_similarity': avg_cosine_sim,
                'avg_pixel_difference': avg_pixel_diff,
                'n_samples': len(sample_images)
            }
            print(f"Dataset: unique={hash_uniqueness:.2f}, correlation={avg_correlation:.3f}, "
                  f"cosine={avg_cosine_sim:.3f}, pixel_diff={avg_pixel_diff:.2f}")
    
    return diversity_results


if __name__ == "__main__":
    # Example usage when run as script
    print("DatasetGen module loaded successfully")
    print("Usage: from dataset_generator import DatasetGen")
    print("\nHelper functions available:")
    print("  - validation(filepath)")
    print("  - preview(filepath, samples_per_k, max_k_values)")
    print("  - analyze_dataset_diversity(filepath, n_samples, by_k)")
    print("  - DatasetGen.load_dataset(filepath)")
    print("  - DatasetGen.load_subset(filepath, indices)")
    print("  - DatasetGen.get_dataset_info(filepath)")
    print("  - DatasetGen.print_dataset_summary(filepath)")
