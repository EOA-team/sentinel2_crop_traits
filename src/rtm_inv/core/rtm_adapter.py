'''
Adapter to radiative transfer models (RTMs). RTMs currently implemented

    - ProSAIL (4SAIL with either Prospect-5 or Prospect-D as leaf model)
    - SPART (BSM, 4SAIL, SMAC and Prospect-5 or Prospect-PRO)

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
import pandas as pd
import prosail
import SPART as spart

from pathlib import Path
from spectral import BandResampler
from typing import Optional

from rtm_inv.core.sensors import Sensors
from rtm_inv.core.utils import green_is_valid, resample_spectra

class RTMRunTimeError(Exception):
    pass

class SPARTParameters:
    """
    class defining which leaf, canopy, soil and atmosphere parameters
    are required to run SPART simulations.

    This class helps scenes the entries of the CSV with the SPART model
    parameterization to the actual SPART function call so that user do not
    have to take care about the order of parameters.
    """

    __name__ = 'SPART'

    # define the entries required to run the SPART submodels
    SMB = ['B', 'lat', 'lon', 'SMp', 'SMC', 'film']  # soil model
    prospect_5d = ['Cab', 'Cca', 'Cw', 'Cdm', 'Cs', 'Cant', 'N', 'PROT', 'CBC']  # leaf model
    sailh = ['LAI', 'LIDFa', 'LIDFb', 'q']  # canopy model
    SMAC = ['aot550', 'uo3', 'uh2o', 'Pa']  # atmosphere model
    angles = ['sol_angle', 'obs_angle', 'rel_angle']  # sun and observer geometry


class ProSAILParameters:
    """
    class defining which leaf and canopy and soil parameters are required to run
    ProSAIL simulations
    """

    __name__ = 'prosail'

    prospect5 = ['n', 'cab', 'car', 'cbrown', 'cw', 'cm']
    prospectD = ['n', 'cab', 'car', 'cbrown', 'cw', 'cm', 'ant']
    fourSAIL= [
        'lai', 'lidfa', 'lidfb', 'psoil', 'rsoil', \
        'hspot', 'tts', 'tto', 'phi', 'typelidf', \
        'rsoil0', 'soil_spectrum1', 'soil_spectrum2', 'alpha'
    ]

class RTM:
    """
    Class for simulating synthetic vegetation spectra
    """
    def __init__(
            self,
            lut,
            rtm: str,
            n_step: Optional[int] = 500
        ):
        """
        Class constructor

        :param lut:
            lookup-table with vegetation traits and parameters for
            which to simulate spectra
        :param rtm:
            name of the RTM to run ("prosail", "SPART")
        :param n_step:
            step at which to write output to logs when created spectra
        """
        if lut.samples.empty:
            raise ValueError('LUT must not be empty')
        if rtm not in ['prosail', 'spart']:
            raise ValueError('Unknown RTM name')
        if n_step <= 0:
            raise ValueError('Steps must be > 0')

        self._lut = lut
        self._rtm = rtm
        self._nstep = n_step

    def _run_spart(
        self,
        sensor: str,
        output: Optional[str] = 'R_TOC',
        doy: Optional[int] = 100
    ) -> None:
        """
        Runs the SPART RTM.

        :param sensor:
            name of the sensor for which to simulate the spectra
        :param output:
            output of the simulation to use. Top-of-Canopy reflectance (R_TOC)
            by default. Further options are 'R_TOA' (top-of-atmosphere reflectance)
            and 'L_TOA' (top-of-atmosphere radiance)
        :param doy:
            day of year (doy) for which to run the simulation (required for sun-earth
            distance calculate). Default is doy 100.
        """
        # get sensor
        try:
            sensor = eval(f'Sensors.{sensor}()')
        except Exception as e:
            raise Exception(f'No such sensor: {sensor}: {e}')
        # get band names
        sensor_bands = sensor.band_names
        sensor_spart_name = sensor.name
        self._lut.samples[sensor_bands] = np.nan

        # get LUT entries for the sub-models of SPART
        # leaf model (PROSPECT-5 or PROSPECT-PRO)
        leaf_traits = SPARTParameters.prospect_5d
        if 'PROT' not in self._lut.samples.columns:
            leaf_traits.remove('PROT')
        if 'CBC' not in self._lut.samples.columns:
            leaf_traits.remove('CBC')
        if 'Cdm' not in self._lut.samples.columns:
            leaf_traits.remove('Cdm')
        lut_leafbio = self._lut.samples[SPARTParameters.prospect_5d].copy()
        # soil model (SMB)
        lut_soilpar = self._lut.samples[SPARTParameters.SMB].copy()
        # canopy model (SAILH)
        lut_canopy = self._lut.samples[SPARTParameters.sailh].copy()
        # angles
        lut_angles = self._lut.samples[SPARTParameters.angles].copy()
        # atmosphere model (SMAC)
        lut_atm = self._lut.samples[SPARTParameters.SMAC].copy()

        # iterate through LUT and run SPART
        for idx in range(self._lut.samples.shape[0]):
            leafbio = spart.LeafBiology(**lut_leafbio.iloc[idx].to_dict())
            canopy = spart.CanopyStructure(**lut_canopy.iloc[idx].to_dict())
            soilpar = spart.SoilParameters(**lut_soilpar.iloc[idx].to_dict())
            angles = spart.Angles(**lut_angles.iloc[idx].to_dict())
            atm = spart.AtmosphericProperties(**lut_atm.iloc[idx].to_dict())
            spart_model = spart.SPART(
                soilpar,
                leafbio,
                canopy,
                atm,
                angles,
                sensor=sensor_spart_name,
                DOY=doy
            )
            spart_sim = spart_model.run()
            self._lut.samples.loc[idx,sensor_bands] = spart_sim[output].values

    def _run_prosail(
        self,
        sensor: str,
        fpath_srf: Optional[Path] = None,
        remove_invalid_green_peaks: Optional[bool] = False
    ) -> None:
        """
        Runs the ProSAIL RTM.

        :param sensor:
            name of the sensor for which to simulate the spectra
        :param fpath_srf:
            optional path to file with spectral response function of the spectral bands
            of the target `sensor`. The data must contain the wavelengths in nm
            and the SRF of the single bands. If not provided, the central wavelength and
            FWHM of the sensor are used assuming a Gaussian SRF.
        :param remove_invalid_green_peaks:
            remove simulated spectra with unrealistic green peaks (occuring at wavelengths > 547nm)
            as suggested by Wocher et al. (2020, https://doi.org/10.1016/j.jag.2020.102219).
        """
        # check if Prospect version
        if set(ProSAILParameters.prospect5).issubset(set(self._lut.samples.columns)):
            prospect_version = '5'
        elif set(self._lut.samples.columns).issubset(ProSAILParameters.prospectD):
            prospect_version = 'D'
        else:
            raise ValueError('Cannot determine Prospect Version')

        # get sensor
        try:
            sensor = eval(f'Sensors.{sensor}()')
        except Exception as e:
            raise Exception(f'No such sensor: {sensor}: {e}')

        # get band names
        sensor_bands = sensor.band_names
        self._lut.samples[sensor_bands] = np.nan

        # define centers and bandwidth of ProSAIL output
        centers_prosail = np.arange(400,2501,1)
        fwhm_prosail = np.ones(centers_prosail.size)

        # no SRF available
        if fpath_srf is None:
            # get central wavelengths and band width per band
            centers_sensor, fwhm_sensor = sensor.central_wvls, sensor.band_widths
            # convert band withs to FWHM (full-width-half-maximum)
            fwhm_sensor = [x*0.5 for x in fwhm_sensor]
            # initialize spectral sampler object to perform the spectral
            # convolution from 1nm ProSAIL output to the sensor's spectral
            # resolution using a Gaussian spectral response function
            resampler = BandResampler(
                centers1=centers_prosail,
                centers2=centers_sensor,
                fwhm1=fwhm_prosail,
                fwhm2=fwhm_sensor
            )
        else:
            srf_df = sensor.read_srf_from_xls(fpath_srf)

        # iterate through LUT and run ProSAIL
        spectrum = None
        traits = self._lut.samples.columns
        # drop band columns B01, B02, etc.
        traits = [x for x in traits if not x.startswith('B')]
        lut = self._lut.samples[traits].copy()
        for idx, record in lut.iterrows():
            # set the PROSPECT version
            record_inp = record.to_dict()
            record_inp.update({
                'prospect_version': prospect_version
            })
            # run ProSAIL
            try:
                spectrum = prosail.run_prosail(**record_inp)
            except Exception as e:
                raise RTMRunTimeError(f'Simulation of spectrum failed: {e}')
            if (idx+1)%self._nstep == 0:
                print(f'Simulated spectrum {idx+1}/{self._lut.samples.shape[0]}')

            # check if the spectrum has an invalid green peak (optionally, following
            # the approach by Wocher et al., 2020, https://doi.org/10.1016/j.jag.2020.102219)
            if remove_invalid_green_peaks:
                valid = green_is_valid(wvls=centers_prosail, spectrum=spectrum)
                # set invalid spectra to NaN (so they can be filtered out) and continue
                if not valid:
                    self._lut.samples.at[idx,sensor_bands] = np.nan
                    continue

            # resample to the spectral resolution of sensor
            if fpath_srf is None:
                sensor_spectrum = resampler(spectrum)
            else:
                # resample RTM output based on true SRFs
                prosail_df = pd.DataFrame(
                    {'wvl': centers_prosail, 'prosail': spectrum}
                )
                sensor_spectrum = resample_spectra(
                    spectral_df=prosail_df, sat_srf=srf_df, wl_column='wvl'
                )
                sensor_spectrum = sensor_spectrum[0].values

            self._lut.samples.at[idx,sensor_bands] = sensor_spectrum

    def simulate_spectra(self, sensor: str, **kwargs) -> pd.DataFrame:
        """
        Simulation of spectra for all entries in the lookup-table

        :paran sensor:
            name of the sensor for which to generate spectra
        :param kwargs:
            RTM-specific optional keyword arguments
        :returns:
            lookup-table with RTM simulated spectra as `DataFrame`
        """
        # call different RTMs
        if self._rtm == 'prosail':
            self._run_prosail(sensor=sensor, **kwargs)
        elif self._rtm == 'spart':
            self._run_spart(sensor=sensor, **kwargs)
        return self._lut.samples
