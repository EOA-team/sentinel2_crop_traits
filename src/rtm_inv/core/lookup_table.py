'''
Module to create lookup-tables (LUT) of synthetic spectra

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

from __future__ import annotations

import lhsmdu
import numpy as np
import pandas as pd

from pathlib import Path
from typing import List, Optional

from rtm_inv.core.distributions import Distributions
from rtm_inv.core.rtm_adapter import RTM
from rtm_inv.core.utils import (
    chlorophyll_carotiniod_constraint,
    glai_ccc_constraint,
    transform_lai
)

# sampling methods available
sampling_methods: List[str] = ['LHS', 'FRS']

class LookupTable(object):
    """
    Lookup-table with RTM simulated spectra plus corresponding
    parameterization (leaf and canopy traits)

    :attrib samples:
        RTM trait samples generated using a custom sample strategy
        sampling. RTM-generated spectra are appended as additional
        columns.
    :attrib lai_transformed:
        was LAI transformed using approach proposed by Verhoef et al.
        (2018, https://doi.org/10.1016/j.rse.2017.08.006)?
    """
    def __init__(
            self,
            params: Path | pd.DataFrame
        ):
        """
        creates a new ``Lookup Table`` instance

        :param params:
            csv file with RTM parameters (traits), their min and max
            value and selected distribution
        """
        if isinstance(params, Path):
            self._params_df = pd.read_csv(self.params_csv)
        elif isinstance(params, pd.DataFrame):
            self._params_df = params.copy()
        else:
            raise TypeError('Expected Path-object or DataFrame')
        self.samples = None
        self.lai_transformed = False

    @property
    def samples(self) -> pd.DataFrame | None:
        """
        Trait samples for generating synthetic spectra
        """
        return self._samples

    @samples.setter
    def samples(self, in_df: pd.DataFrame) -> None:
        """
        Trait samples for generating synthetic spectra
        """
        if in_df is not None:
            if not isinstance(in_df, pd.DataFrame):
                raise TypeError(
                    f'Expected a pandas DataFrame instance, got {type(in_df)}'
                )
        self._samples = in_df

    def generate_samples(
            self,
            num_samples: int,
            method: str,
            seed_value: Optional[int] = 0,
            apply_glai_ccc_constraint: Optional[bool] = True,
            apply_chlorophyll_carotiniod_constraint: Optional[bool] = True
        ):
        """
        Sample parameter values using a custom sampling scheme.

        Currently supported sampling schemes are:

        - Latin Hypercube Sampling (LHS)
        - Fully Random Sampling (FRS) using distributions of traits
        - ...

        All parameters (traits) are sampled, whose distribution is not set
        as "constant"

        :param num_samples:
            number of samples to draw (equals the size of the resulting
            lookup-table)
        :param method:
            sampling method to apply
        :param seed_value:
            seed value to set to the pseudo-random-number generator. Default
            is zero.
        :param apply_glai_ccc_constraint:
            whether the apply the GLAI-CCC constraint. Default is True.
        :param apply_glai_ccc_constraint:
            whether the apply the Cab-Car constraint. Default is True.
        """
        # set seed to the random number generator
        np.random.seed(seed_value)

        # determine traits to sample (indicated by a distribution different from
        # "Constant"
        traits = self._params_df[
            self._params_df['Distribution'].isin(Distributions.distributions)
        ]
        trait_names = traits['Parameter'].to_list()
        traits = traits.transpose()
        traits.columns = trait_names
        n_traits = len(trait_names)

        # and those traits/ parameters remaining constant
        constant_traits = self._params_df[
            ~self._params_df['Parameter'].isin(trait_names)
        ]
        constant_trait_names = constant_traits['Parameter'].to_list()
        constant_traits = constant_traits.transpose()
        constant_traits.columns = constant_trait_names

        # select method and conduct sampling
        if method.upper() == 'LHS':
            # create LHS matrix
            lhc = lhsmdu.createRandomStandardUniformMatrix(n_traits, num_samples)
            traits_lhc = pd.DataFrame(lhc).transpose()
            traits_lhc.columns = trait_names
            # replace original values in LHS matrix (scaled between 0 and 1) with
            # trait values scaled between their specific min and max
            for trait_name in trait_names:
                traits_lhc[trait_name] = traits_lhc[trait_name] * \
                    traits[trait_name]['Max'] + traits[trait_name]['Min']
            sample_traits = traits_lhc
        elif method.upper() == 'FRS':
            # fully random sampling within the trait ranges specified
            # drawing a random sample for each trait
            frs_matrix = np.empty((num_samples, n_traits), dtype=np.float32)
            traits_frs = pd.DataFrame(frs_matrix)
            traits_frs.columns = trait_names
            for trait_name in trait_names:
                mode = None
                if 'Mode' in traits[trait_name].index:
                    mode = traits[trait_name]['Mode']
                std = None
                if 'Std' in traits[trait_name].index:
                    std = traits[trait_name]['Std']
                dist = Distributions(
                    min_value=traits[trait_name]['Min'],
                    max_value=traits[trait_name]['Max'],
                    mean_value=mode,
                    std_value=std
                )
                traits_frs[trait_name] = dist.sample(
                    distribution=traits[trait_name]['Distribution'],
                    n_samples=num_samples
                )
            sample_traits = traits_frs
        else:
            raise NotImplementedError(f'{method} not found')

        # combine trait samples and constant values into a single DataFrame
        # so that in can be passed to the RTM
        for constant_trait in constant_trait_names:
            # for constant traits the value in the min column is used
            # (actually min and max should be set to the same value)
            sample_traits[constant_trait] = constant_traits[constant_trait]['Min']

        # implement constraints to make the LUT physiologically more plausible, i.e.,
        # increase the correlation between plant biochemical and biophysical parameters
        # which is observed in plants but not reflected by a random sampling scheme
        if apply_glai_ccc_constraint:
            sample_traits = glai_ccc_constraint(lut_df=sample_traits)
        if apply_chlorophyll_carotiniod_constraint:
            sample_traits = chlorophyll_carotiniod_constraint(lut_df=sample_traits)
        # set samples to instance variable
        self.samples = sample_traits

def _setup(
        lut_params: pd.DataFrame,
        rtm_name: str,
        solar_zenith_angle: Optional[float] = None,
        viewing_zenith_angle: Optional[float] = None,
        solar_azimuth_angle: Optional[float] = None,
        viewing_azimuth_angle: Optional[float] = None,
        relative_azimuth_angle: Optional[float] = None
    ) -> pd.DataFrame:
    """
    Setup LUT for RTM (modification of angles and names if required)
    """
    if rtm_name == 'prosail':
        sol_angle = 'tts'   # solar zenith
        obs_angle = 'tto'   # observer zenith
        rel_angle = 'psi'   # relative azimuth
    elif rtm_name == 'spart':
        sol_angle = 'sol_angle'
        obs_angle = 'obs_angle'
        rel_angle = 'rel_angle'

    # overwrite angles in LUT DataFrame if provided as fixed values
    if solar_zenith_angle is not None:
        lut_params.loc[lut_params['Parameter'] == sol_angle,'Min'] = solar_zenith_angle
        lut_params.loc[lut_params['Parameter'] == sol_angle,'Max'] = solar_zenith_angle
    if viewing_zenith_angle is not None:
        lut_params.loc[lut_params['Parameter'] == obs_angle, 'Min'] = viewing_zenith_angle
        lut_params.loc[lut_params['Parameter'] == obs_angle, 'Max'] = viewing_zenith_angle
    # calculate relative azimuth (psi) if viewing angles are passed
    if viewing_azimuth_angle is not None and solar_azimuth_angle is not None:
        psi = abs(solar_azimuth_angle - viewing_azimuth_angle)
        lut_params.loc[lut_params['Parameter'] == rel_angle, 'Min'] = psi
        lut_params.loc[lut_params['Parameter'] == rel_angle, 'Max'] = psi
    if relative_azimuth_angle is not None:
        psi = relative_azimuth_angle
        lut_params.loc[lut_params['Parameter'] == rel_angle, 'Min'] = psi
        lut_params.loc[lut_params['Parameter'] == rel_angle, 'Max'] = psi

    # 'mode' and 'std' are optional columns
    further_columns = ['Mode', 'Std']
    for further_column in further_columns:
        if further_column in lut_params.columns:
            lut_params.loc[lut_params['Parameter'] == sol_angle, further_column] = solar_zenith_angle
            lut_params.loc[lut_params['Parameter'] == obs_angle, further_column] = viewing_zenith_angle
            lut_params.loc[lut_params['Parameter'] == rel_angle, further_column] = psi

    return lut_params

def generate_lut(
        sensor: str,
        lut_params: Path | pd.DataFrame,
        lut_size: Optional[int] = 50000,
        rtm_name: Optional[str] = 'prosail',
        sampling_method: Optional[str] = 'LHS',
        solar_zenith_angle: Optional[float] = None,
        viewing_zenith_angle: Optional[float] = None,
        solar_azimuth_angle: Optional[float] = None,
        viewing_azimuth_angle: Optional[float] = None,
        relative_azimuth_angle: Optional[float] = None,
        fpath_srf: Optional[Path] = None,
        remove_invalid_green_peaks: Optional[bool] = False,
        linearize_lai: Optional[bool] = False,
        **kwargs
    ) -> pd.DataFrame:
    """
    Generates a Lookup-Table (LUT) based on radiative transfer model input parameters.

    IMPORTANT:
        Depending on the RTM and the size of the LUT the generation of a LUT
        might take a while!

    :param sensor:
        name of the sensor for which the simulated spectra should be resampled.
        See `rtm_inv.core.sensors.Sensors` for a list of sensors currently implemented.
    :param lut_params:
        lookup-table parameters with mininum and maximum range (always required),
        type of distribution (important to indicate which parameters are constant),
        mode and std (for Gaussian distributions).
    :param lut_size:
        number of items (spectra) to simulate in the LUT
    :param rtm_name:
        name of the RTM to call.
    :param sampling_method:
        sampling method for generating the input parameter space of the LUT. 'LHS'
        (latin hypercube sampling) by default.
    :param solar_zenith_angle:
        solar zenith angle as fixed scene-wide value (optional) in degrees.
    :param viewing_zenith_angle:
        viewing (observer) zenith angle as fixed scene-wide value (optional) in degrees.
    :param solar_azimuth_angle:
        solar azimuth angle as fixed scene-wide value (optional) in deg C.
    :param viewing_azimuth_angle:
        viewing (observer) azimuth angle as fixed scene-wide value (optional) in deg C.
    :param relative_azimuth_angle:
        relative azimuth angle (if available, optional) in deg C. If provided, the relative
        azimuth angle is not calculated from solar and observer azimuth angle and also
        not checked against them!
    :param fpath_srf:
        if provided uses actual spectral response functions (SRF) for spectral resampling
        of RTM outputs (usually in 1nm steps) into the spectral resolution of a given sensor
    :param remove_invalid_green_peaks:
        remove simulated spectra with unrealistic green peaks (occuring at wavelengths > 547nm)
        as suggested by Wocher et al. (2020, https://doi.org/10.1016/j.jag.2020.102219).
        NOTE: When this option is used, spectra not fulfilling the green peak criterion 
        are set to NaN.
    :param linearize_lai:
        if True, transforms LAI values to a more linearized representation
        as suggested by Verhoef et al. (2018, https://doi.org/10.1016/j.rse.2017.08.006)
    :param kwargs:
        optional keyword-arguments to pass to `LookupTable.generate_samples`
    :returns:
        input parameters and simulated spectra as `DataFrame`.
    """
    # read parameters from CSV if not provided as a DataFrame
    if isinstance(lut_params, Path):
        lut_params = pd.read_csv(lut_params)

    # prepare LUTs for RTMs
    lut_params = _setup(lut_params, rtm_name, solar_zenith_angle, viewing_zenith_angle,
                        solar_azimuth_angle, viewing_azimuth_angle, relative_azimuth_angle)
    # get input parameter samples first
    lut = LookupTable(params=lut_params)
    lut.generate_samples(num_samples=lut_size, method=sampling_method, **kwargs)

    # and run the RTM in forward mode in the second step
    # outputs get resampled to the spectral resolution of the sensor
    rtm = RTM(lut=lut, rtm=rtm_name)
    lut_simulations = rtm.simulate_spectra(
        sensor=sensor,
        fpath_srf=fpath_srf,
        remove_invalid_green_peaks=remove_invalid_green_peaks
    )
    # linearize LAI as proposed by Verhoef et al. (2018,
    # https://doi.org/10.1016/j.rse.2017.08.006)
    if linearize_lai:
        lut_simulations['lai'] = transform_lai(lut_simulations['lai'], inverse=False)
    return lut_simulations
