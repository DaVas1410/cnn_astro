"""
This module can be used to generate a fractal cube cloud
with the parameters from an INI file.

It uses our slightly modified version of the pyFC library.

Author: Males-Araujo Yorlan
"""
# Libraries
import os
import numpy as np
import configparser
from datetime import datetime
from typing import Optional, Tuple

from pyFC import LogNormalFractalCube

class ParameterReader:
    """
    Extract parameters from a INI file.
    """
    def __init__(self, file_path: str):
        """
        Initializes the class with the given file path.

        Parameters
        ----------
        file_path : str, required
            Path of the .ini file to be read.
        """
        # Store the file path and read the file contents
        self.file_path = file_path
        self.config = self._read_config()

        # Spatial dimensions
        self.dimensions = self.config.getint('Parameters', 'dimensions')

        # Maximum and minimum wavenumbers
        self.kmin = self.config.getfloat('Parameters', 'kmin')
        self.kmax = self._parse_kmax()

        # Dimensions of the grid
        self.ni, self.nj, self.nk = self._parse_grid_dimensions()

        # Gaussian (normal) parameters
        self.mean = self.config.getfloat('Parameters', 'mean')
        self.sigma = self.config.getfloat('Parameters', 'sigma')

        # Beta
        self.beta = self.config.getfloat('Parameters', 'beta')

        # Binary file name and summary
        self.name = self.config.get('Parameters', 'name')
        self.summary = self.config.getboolean('Parameters', 'summary')

        # Random seed
        self.random_seed = self.config.getboolean('Parameters', 'random_seed')

        # Stuff needed in other methods
        self.cloud = None
        self.vtk_filename = None
        self.data = None

    # ________________________________________________
    # The following methods are used within __init__()
    # to help reading some specific parameters.
    # ________________________________________________

    def _read_config(self) -> configparser.ConfigParser:
        """
        Read the file contents.

        Returns
        -------
        config : configparser.ConfigParser
            Stores the parameters.
        """
        # Simply set it
        config = configparser.ConfigParser()

        # And read the file
        config.read(self.file_path)
        return config

    def _parse_kmax(self) -> Optional[int]:
        """
        Read the maximum wavenumber. None would
        correspond to the Nyquist limit.

        Returns
        -------
        kmax_str : None or integer
            Determines the maximum wavenumber.
        """
        # Get it as
        kmax_str = self.config.get('Parameters', 'kmax')

        return None if kmax_str == 'None' else float(kmax_str)

    def _parse_grid_dimensions(self) -> Tuple[int, int, Optional[int]]:
        """
        Get all grid dimensions.

        Returns
        -------
        Tuple with:
        ni, nj : int
            x and y grid dimensions
        nk : int or None
            z grid dimension. None if self.dimensions != 3.
        """
        # Read them
        ni = self.config.getint('Parameters', 'ni')
        nj = self.config.getint('Parameters', 'nj')
        nk = self.config.getint('Parameters', 'nk') if self.dimensions == 3 else None

        return ni, nj, nk


class CubeGenerator(ParameterReader):
    """
    It creates the fractal cube with the parameters from an INI file,
    and it can be used to export it as a binary or VTK file.

    It inherits the ParameterReader class.

    Parameters
    ----------
    params : str, required
        Path to the file with parameters.
    """
    def __init__(self, params: str) -> None:
        """
        Initialize this class with the INI file.
        """
        if not os.path.exists(params):
            raise FileNotFoundError(f"Parameters file not found: {params}")
        
        super().__init__(params)
        self.cube_params = self._cube_params()
        self.data        = None

    # ===============================================================================
    #                                 Public methods                                
    # ===============================================================================

    def getData(self) -> np.ndarray:
        """
        Get the fractal cube data from a binary file.

        Returns
        -------
        data : np.ndarray
            Data from the binary file.
        """
        # Read the binary file
        self._write_binary()
        data = np.fromfile(self.name, dtype = np.float64)

        if self.dimensions == 3:
            shape = (self.ni, self.nj, self.nk)
        else:
            shape = (self.ni, self.nj)

        self.data = data.reshape(shape)

        # Binary no longer needed
        if os.path.exists(self.name):
            os.remove(self.name)

        return self.data
    
    # ===============================================================================
    #                                 Private methods                                
    # ===============================================================================

    def _cube_params(self) -> dict:
        """
        Return a dictionary with the parameters to generate the fractal cube.

        Returns
        -------
        cube_params : dict
            Dictionary with the parameters.
        """
        cube_params = {
            'ni': self.ni,
            'nj': self.nj,
            'kmin': self.kmin,
            'kmax': self.kmax,
            'sigma': self.sigma,
            'mean': self.mean,
            'beta': self.beta
        }

        if self.dimensions == 3:
            cube_params['nk'] = self.nk

        return cube_params

    def _write_binary(self) -> None:
        """
        Generate a fractal cube using the LogNormalFractalCube class from pyFC
        and export it as a binary file.
        
        The cube is generated using LogNormalFractalCube's `gen_cube()` method
        and then exported using `write_cube()` method.
        """
        try: 
            self.cloud = LogNormalFractalCube(**self.cube_params)

            # Generate
            self.cloud.gen_cube(
                verbose = False, 
                summary = self.summary, 
                log_name = f"pyFC-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log", 
                random_seed = self.random_seed
                )

            # Export to binary file
            self.cloud.write_cube(fname = self.name, app = False)

        except Exception as e:
            self.cloud = None
            raise RuntimeError(f"Failed to generate and export fractal cube: {str(e)}")
