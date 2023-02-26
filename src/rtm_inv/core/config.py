'''
A base class for configuring the RTM inversion and trait retrieval

Copyright (C) 2022 Lukas Valentin Graf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np

from pathlib import Path
from typing import Any, Dict, List, Optional

class RTMConfig:
    """
    Radiative transfer model inversion configuration set-up
    """
    def __init__(
            self,
            traits: List[str],
            lut_params: Path,
            rtm: Optional[str] = 'prosail'
        ):
        """
        Class Constructor

        :param traits:
            list of traits to retrieve from the inversion process
            (depending on radiative transfer model chosen)
        :param rtm_params:
            CSV-file specifying RTM specific parameter ranges and their
            distribution
        :param rtm:
            name of the radiative transfer model to use. 'prosail' by
            default
        """
        self.traits = traits
        self.lut_params = lut_params
        self.rtm = rtm

        @property
        def lut_params() -> Path:
            return self.lut_params

        @property
        def traits() -> List[str]:
            return self.traits

        @property
        def rtm() -> str:
            return self.rtm

        def to_dict(self) -> Dict[str, Any]:
            return self.__dict__

class LookupTableBasedInversion(RTMConfig):
    """
    Lookup-table based radiative transfer model inversion set-up
    """
    def __init__(
            self,
            n_solutions: int,
            cost_function: str,
            lut_size: int,
            sampling_method: Optional[str] = 'LHS',
            fpath_srf: Optional[str] = None,
            remove_invalid_green_peaks: Optional[bool] = True,
            **kwargs
        ):
        """
        Class constructor

        :param n_solutions:
            number of solutions to use for lookup-table based inversion. Can
            be provided as relative share of spectra in the lookup-table (then
            provide a number between 0 and 1) or in absolute numbers.
        :param cost_function:
            name of the cost-function to evaluate similarity between
            observed and simulated spectra
        :param lut_size:
            size of the lookup-table (number of spectra to simulate)
        :param sampling_method:
            method to use for sampling the lookup-table
        :param fpath_srf:
            path to Spreadsheet with spectral response functions to use
            for resampling PROSAIL spectra to the spectral resolution of a
            given sensor. If not provided (default) uses central wavelength and
            FWHMs of the sensor assuming a Gaussian spectral response function.
        :param remove_invalid_green_peaks:
            if True (default) removes PROSAIL spectra with a green reflectance
            peak shifted towards unrealistically short wavelengths (blue region
            of the spectrum) as suggested by Wocher et al. (2020).
        :param kwargs:
            keyword arguments to pass to super-class
        """
        # call constructor of super-class
        super().__init__(**kwargs)
        self.cost_function = cost_function
        self.sampling_method = sampling_method
        self.lut_size = lut_size
        self.fpath_srf = fpath_srf
        self.remove_invalid_green_peaks = remove_invalid_green_peaks

        # adopt number of solutions if given in relative numbers [0,1[
        if 0 < n_solutions < 1:
            self.n_solutions = int(np.round(n_solutions * lut_size, 0))
        elif n_solutions < 0:
            raise ValueError('The number of solutions must not be negative')
        elif n_solutions > self.lut_size:
            raise ValueError(
                'The number of solutions must not be greater than the lookup-table size'
            )
        else:
            self.n_solutions = n_solutions

        @property
        def lut_size() -> int:
            return self.lut_size

        @property
        def n_solutions() -> int:
            return self.n_solutions

        @property
        def cost_function() -> str:
            return self.cost_function

        @property
        def sampling_method() -> str:
            return self.method
