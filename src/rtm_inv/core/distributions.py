'''
Distributions for sampling RTM parameters.

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

from scipy.stats import truncnorm
from typing import List, Optional, Union


class Distributions(object):
    """
    Class with statistical distributions for drawning RTM
    samples from a set of input parameters.

    For each RTM parameter, min, max and type of distribution
    must be passed.
    """

    distributions: List[str] = ['Gaussian', 'Uniform']

    def __init__(
            self,
            min_value: Union[int,float],
            max_value: Union[int,float],
            mean_value: Optional[Union[int,float]] = None,
            std_value: Optional[Union[int,float]] = None
        ):
        """
        Creates a new ``Distributions`` class to use for sampling
        RTM parameter values

        :param min_value:
            minimum parameter value
        :param max_value:
            maximum parameter value
        :param mean_value:
            optional mean value to be used for creating a Gaussian
            distribution. If not provided the mean is calculated as
            min_value + 0.5 * (max_value - min_value)
        :param std_value:
            optional standard deviation value to be used for creating
            a Gaussian distribution. If not provided the standard
            deviation is calculated as 0.5 * (max_value - min_value)
        """
        if min_value > max_value:
            raise ValueError('Minimum cannot be greater than maximum')
        if mean_value is None or np.isnan(mean_value):
            mean_value = min_value + 0.5 * (max_value - min_value)
        if std_value is None or np.isnan(std_value):
            std_value = 0.5 * (max_value - min_value)
        self.min_value = min_value
        self.max_value = max_value
        self.mean_value = mean_value
        self.std_value = std_value

    def sample(self,
               distribution: str,
               n_samples: int
            ):
        """
        Returns a ``numpy.ndarray`` with RTM parameter samples drawn from
        a specific distribution

        :param distribution:
            name of the distribution from which to sample. See
            `~core.distributions.Distributions.distributions` for a list
            of distributions currently implemented
        :param n_samples:
            number of samples to draw.
        """
        if distribution == 'Uniform':
            return np.random.uniform(
                low=self.min_value,
                high=self.max_value,
                size=n_samples
            )
        elif distribution == 'Gaussian':
            a, b = (self.min_value - self.mean_value) / self.std_value, \
                (self.max_value - self.mean_value) / self.std_value
            X = truncnorm(a, b, loc=self.mean_value, scale=self.std_value)
            return X.rvs(n_samples)
