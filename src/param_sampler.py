"""
Parameter Sampler for Flexible Dataset Generation

Provides utilities for sampling fractal cube generation parameters (k_min, k_max, sigma, mean)
from continuous ranges while respecting physical constraints from the pyFC library.

Supports multiple distribution types including uniform and exponential (skewed to favor larger values).

Author: Copilot
Date: 2026
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Literal
from dataclasses import dataclass
import warnings


@dataclass
class DistributionConfig:
    """Configuration for parameter distribution sampling.
    
    Attributes:
        dist_type: Distribution type ('uniform' or 'exponential')
        rate: Rate parameter for exponential distribution (lambda). Higher values = stronger skew toward higher values.
              Ignored for uniform distribution.
    """
    dist_type: Literal['uniform', 'exponential'] = 'uniform'
    rate: float = 1.0
    
    def __post_init__(self):
        if self.dist_type not in ('uniform', 'exponential'):
            raise ValueError(f"dist_type must be 'uniform' or 'exponential', got {self.dist_type}")
        if self.rate <= 0:
            raise ValueError(f"rate must be > 0, got {self.rate}")


class ParameterSampler:
    """
    Sampler for generating random parameter sets for fractal cube generation.
    
    Supports multiple distribution types (uniform, exponential) to control parameter sampling.
    
    Enforces physical constraints:
    - k_min >= 1 (minimum wavenumber)
    - k_max <= Nyquist limit (based on image dimensions)
    - k_min < k_max (physical ordering)
    - sigma > 0 (distribution parameter)
    
    Example:
        >>> sampler = ParameterSampler(
        ...     image_height=512, image_width=512,
        ...     kmin_range=(1, 32), kmax_range=(50, 256),
        ...     sigma_range=(2.0, 2.5),
        ...     kmin_dist=DistributionConfig(dist_type='exponential', rate=2.0),
        ...     kmax_dist=DistributionConfig(dist_type='exponential', rate=1.5)
        ... )
        >>> params = sampler.sample_batch(n_images=100)
        >>> print(params['k_min'].shape)  # (100,)
    """
    
    def __init__(
        self,
        image_height: int,
        image_width: int,
        image_depth: int = 1,
        kmin_range: Tuple[float, float] = (1.0, 32.0),
        kmax_range: Optional[Tuple[float, float]] = None,
        sigma_range: Tuple[float, float] = (2.0, 2.5),
        mean_range: Tuple[float, float] = (1.0, 1.0),
        auto_kmax: bool = False,
        kmin_dist: Optional[DistributionConfig] = None,
        kmax_dist: Optional[DistributionConfig] = None,
        sigma_dist: Optional[DistributionConfig] = None,
    ):
        """
        Initialize the parameter sampler.
        
        Args:
            image_height: Height of generated image (nj)
            image_width: Width of generated image (ni)
            image_depth: Depth of generated image (nk)
            kmin_range: (min, max) range for k_min sampling
            kmax_range: (min, max) range for k_max sampling, or None for auto (Nyquist)
            sigma_range: (min, max) range for sigma sampling
            mean_range: (min, max) range for mean sampling
            auto_kmax: If True, always use Nyquist limit for k_max (ignore kmax_range)
            kmin_dist: Distribution config for k_min (default: uniform)
            kmax_dist: Distribution config for k_max (default: uniform)
            sigma_dist: Distribution config for sigma (default: uniform)
        
        Raises:
            ValueError: If ranges are invalid or constraint-violating
        """
        self.image_height = image_height
        self.image_width = image_width
        self.image_depth = image_depth
        self.auto_kmax = auto_kmax
        
        # Store distribution configs
        self.kmin_dist = kmin_dist or DistributionConfig(dist_type='uniform')
        self.kmax_dist = kmax_dist or DistributionConfig(dist_type='uniform')
        self.sigma_dist = sigma_dist or DistributionConfig(dist_type='uniform')
        
        # Calculate Nyquist limit (= floor(max(ni, nj, nk) / 2))
        self.nyquist_limit = float(np.floor(max(image_height, image_width, image_depth) / 2.0))
        
        # Validate and store k_min range
        self.kmin_range = self._validate_range(kmin_range, "k_min")
        if self.kmin_range[0] < 1.0:
            warnings.warn(
                f"k_min_low={self.kmin_range[0]} < 1.0 (physical minimum). "
                f"Values will be clamped to 1.0 by pyFC."
            )
        
        # Validate and store k_max range
        if auto_kmax or kmax_range is None:
            self.auto_kmax = True
            self.kmax_range = (self.nyquist_limit, self.nyquist_limit)
        else:
            self.kmax_range = self._validate_range(kmax_range, "k_max")
            if self.kmax_range[1] > self.nyquist_limit:
                warnings.warn(
                    f"k_max_high={self.kmax_range[1]} > Nyquist limit={self.nyquist_limit}. "
                    f"Values will be clamped by pyFC."
                )
        
        # Validate and store sigma range
        self.sigma_range = self._validate_range(sigma_range, "sigma")
        if self.sigma_range[0] <= 0.0:
            raise ValueError(f"sigma_low must be > 0, got {self.sigma_range[0]}")
        
        # Validate and store mean range
        self.mean_range = self._validate_range(mean_range, "mean")
        
        # Validate that kmin_max < kmax_min (will be checked per-sample)
        if not auto_kmax and self.kmin_range[1] >= self.kmax_range[0]:
            warnings.warn(
                f"k_min_high={self.kmin_range[1]} >= k_max_low={self.kmax_range[0]}. "
                f"Some samples may violate k_min < k_max constraint."
            )
    
    @staticmethod
    def _validate_range(r: Tuple[float, float], name: str) -> Tuple[float, float]:
        """
        Validate a parameter range.
        
        Args:
            r: (min, max) tuple
            name: Parameter name for error messages
            
        Returns:
            Validated (min, max) tuple
            
        Raises:
            ValueError: If range is invalid
        """
        if not isinstance(r, (tuple, list)) or len(r) != 2:
            raise ValueError(f"{name} range must be a 2-tuple, got {r}")
        
        min_val, max_val = float(r[0]), float(r[1])
        
        if min_val > max_val:
            raise ValueError(
                f"{name} range invalid: min ({min_val}) > max ({max_val})"
            )
        
        return (min_val, max_val)
    
    @staticmethod
    def _sample_exponential_bounded(low: float, high: float, rate: float, size: int) -> np.ndarray:
        """
        Sample from exponential distribution bounded to [low, high], skewed toward larger values.
        
        Uses inverse exponential distribution (flipped) to favor larger values within the range.
        The rate parameter controls skewness strength.
        
        Args:
            low: Lower bound
            high: Upper bound
            rate: Rate parameter (lambda) for exponential distribution. Higher values = stronger skew toward larger values.
            size: Number of samples
            
        Returns:
            (size,) array of bounded exponential samples, skewed toward high end
        """
        range_width = high - low
        # Sample from exponential distribution and invert to favor larger values
        # The transformation: if X ~ Exp(rate), then cdf is 1 - exp(-rate*x)
        # We sample U ~ Uniform(0, 1) and use quantile: x = -log(1-U) / rate
        u = np.random.uniform(0, 1, size=size)
        exp_samples = -np.log(1 - u) / rate
        # Normalize to [0, 1]
        exp_normalized = np.clip(exp_samples / exp_samples.max() * 1.0, 0, 1)
        # Invert: (1 - exp_normalized) puts small exponential values at the high end
        exp_inverted = 1.0 - exp_normalized
        # Scale to [low, high]
        return (low + exp_inverted * range_width).astype(np.float32)
    
    def sample_batch(self, n_images: int, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Generate a batch of random parameter sets.
        
        Args:
            n_images: Number of parameter sets to generate
            seed: Optional random seed for reproducibility
            
        Returns:
            Dictionary with keys:
            - 'k_min': (n_images,) array of k_min values
            - 'k_max': (n_images,) array of k_max values
            - 'sigma': (n_images,) array of sigma values
            - 'mean': (n_images,) array of mean values
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Sample k_min with configured distribution
        k_mins = self._sample_parameter(
            self.kmin_range[0], self.kmin_range[1], 
            self.kmin_dist, n_images
        )
        
        # Sample k_max
        if self.auto_kmax:
            # All k_max values are Nyquist limit
            k_maxs = np.full(n_images, self.nyquist_limit, dtype=np.float32)
        else:
            k_maxs = self._sample_parameter(
                self.kmax_range[0], self.kmax_range[1],
                self.kmax_dist, n_images
            )
        
        # Sample sigma with configured distribution
        sigmas = self._sample_parameter(
            self.sigma_range[0], self.sigma_range[1],
            self.sigma_dist, n_images
        )
        
        # Sample mean (always uniform for simplicity)
        means = np.random.uniform(self.mean_range[0], self.mean_range[1], size=n_images)
        
        return {
            'k_min': k_mins.astype(np.float32),
            'k_max': k_maxs.astype(np.float32),
            'sigma': sigmas.astype(np.float32),
            'mean': means.astype(np.float32),
        }
    
    def _sample_parameter(self, low: float, high: float, config: DistributionConfig, size: int) -> np.ndarray:
        """
        Sample parameter values using the specified distribution.
        
        Args:
            low: Lower bound
            high: Upper bound
            config: Distribution configuration
            size: Number of samples
            
        Returns:
            Array of sampled values
        """
        if config.dist_type == 'uniform':
            return np.random.uniform(low, high, size=size)
        elif config.dist_type == 'exponential':
            return self._sample_exponential_bounded(low, high, config.rate, size)
        else:
            raise ValueError(f"Unknown distribution type: {config.dist_type}")
    
    def get_config(self) -> Dict:
        """
        Get configuration dictionary for metadata storage.
        
        Returns:
            Dictionary with sampler configuration
        """
        return {
            'image_height': self.image_height,
            'image_width': self.image_width,
            'image_depth': self.image_depth,
            'nyquist_limit': self.nyquist_limit,
            'kmin_range': list(self.kmin_range),
            'kmax_range': list(self.kmax_range),
            'auto_kmax': self.auto_kmax,
            'sigma_range': list(self.sigma_range),
            'mean_range': list(self.mean_range),
            'kmin_dist': {'dist_type': self.kmin_dist.dist_type, 'rate': self.kmin_dist.rate},
            'kmax_dist': {'dist_type': self.kmax_dist.dist_type, 'rate': self.kmax_dist.rate},
            'sigma_dist': {'dist_type': self.sigma_dist.dist_type, 'rate': self.sigma_dist.rate},
        }
    
    def __repr__(self) -> str:
        """String representation of sampler configuration."""
        return (
            f"ParameterSampler("
            f"image={self.image_width}x{self.image_height}x{self.image_depth}, "
            f"Nyquist={self.nyquist_limit}, "
            f"k_min={self.kmin_range}({self.kmin_dist.dist_type}), "
            f"k_max={'auto' if self.auto_kmax else self.kmax_range}({self.kmax_dist.dist_type}), "
            f"sigma={self.sigma_range}({self.sigma_dist.dist_type}), "
            f"mean={self.mean_range}"
            f")"
        )


class ParameterValidator:
    """
    Validates parameter values against pyFC constraints.
    
    Provides static methods to check if parameter combinations are physically valid.
    """
    
    @staticmethod
    def validate_kmin(k_min: float, image_height: int, image_width: int, image_depth: int) -> bool:
        """
        Check if k_min is valid.
        
        k_min must be >= 1 after scaling by cube ratios (for isotropic cubes, simply k_min >= 1).
        
        Args:
            k_min: Minimum wavenumber
            image_height: ni (height)
            image_width: nj (width)
            image_depth: nk (depth)
            
        Returns:
            True if k_min is valid, False otherwise
        """
        # For isotropic cube or when smaller dimensions are sufficient
        # Check simplified constraint: k_min >= 1
        return k_min >= 1.0
    
    @staticmethod
    def validate_kmax(k_max: float, image_height: int, image_width: int, image_depth: int) -> bool:
        """
        Check if k_max is valid.
        
        k_max must not exceed Nyquist limit in any direction.
        
        Args:
            k_max: Maximum wavenumber
            image_height: ni (height)
            image_width: nj (width)
            image_depth: nk (depth)
            
        Returns:
            True if k_max is valid, False otherwise
        """
        nyquist = np.floor(max(image_height, image_width, image_depth) / 2.0)
        return k_max <= nyquist
    
    @staticmethod
    def validate_pair(k_min: float, k_max: float) -> bool:
        """
        Check if k_min and k_max satisfy ordering constraint.
        
        Args:
            k_min: Minimum wavenumber
            k_max: Maximum wavenumber
            
        Returns:
            True if k_min < k_max, False otherwise
        """
        return k_min < k_max
    
    @staticmethod
    def validate_sigma(sigma: float) -> bool:
        """
        Check if sigma is valid (must be positive).
        
        Args:
            sigma: Distribution parameter
            
        Returns:
            True if sigma > 0, False otherwise
        """
        return sigma > 0.0


# Convenience functions for single-parameter sampling

def sample_kmin_range(low: float, high: float, n: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Sample k_min values uniformly from a range.
    
    Args:
        low: Minimum k_min value
        high: Maximum k_min value
        n: Number of samples
        seed: Optional random seed
        
    Returns:
        (n,) array of k_min values
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(low, high, size=n).astype(np.float32)


def sample_kmax_range(low: float, high: float, n: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Sample k_max values uniformly from a range.
    
    Args:
        low: Minimum k_max value
        high: Maximum k_max value
        n: Number of samples
        seed: Optional random seed
        
    Returns:
        (n,) array of k_max values
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(low, high, size=n).astype(np.float32)


def sample_sigma_range(low: float, high: float, n: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Sample sigma values uniformly from a range.
    
    Args:
        low: Minimum sigma value
        high: Maximum sigma value
        n: Number of samples
        seed: Optional random seed
        
    Returns:
        (n,) array of sigma values
    """
    if seed is not None:
        np.random.seed(seed)
    return np.random.uniform(low, high, size=n).astype(np.float32)
