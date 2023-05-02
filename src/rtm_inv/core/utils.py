"""
"""

import pandas as pd
import numpy as np

from scipy.stats import poisson
from typing import Optional

green_peak_threshold = 547 # nm
green_region = (500, 600) # nm

def resample_spectra(spectral_df: pd.DataFrame, sat_srf: pd.DataFrame, wl_column: str = "WL"
                     ) -> pd.DataFrame:
    """
    Function to spectrally resample hyperspectral 1nm RTM output to the
    spectral response function of a multi-spectral sensor.

    :param spectral_df:
        data frame containing the original spectra, needs to be in long-format with a
        "wavelength" (WL) column. Currently the WL column must be named the same as
        in the satellite SRF df.
    :param sat_srf:
        spectral response function (SRF) of the satellite. Must be in long-format with
        a WL column.
    :param wl_column:
        Name of the column containing the wavelength
    :returns:
        A data frame with the resampled spectra
    """
    # get satellite bandnames
    sat_bands = sat_srf.drop(wl_column, axis=1).columns

    # get columns of the spectral_df (=individual spectra)
    indiv_spectra = spectral_df.drop(wl_column, axis=1).columns

    # force the satellite SRF onto the same spectra as the spectral DF
    innerdf = spectral_df.merge(sat_srf, on=wl_column)
    sat_srf = innerdf.loc[:, [wl_column, *sat_bands]]

    # iterate over every spectra
    out_list = []
    for spectrum in indiv_spectra:

        spec_df = spectral_df.loc[:, spectrum]
        # iterate over every target (multi-spectral) band
        spectrum_bandlist = []
        for satband in sat_bands:

            # get response of selected band
            sat_response = sat_srf.loc[:, satband]
            # start the resampling process. First, multiply the 1nm RTM
            # output with the spectral response function of the target
            # band
            sim_vec = spec_df * sat_response
            # set zeroes to NA for calculations
            sim_vec[sim_vec == 0] = np.nan

            # second, sum up the rescaled reflectance values and divide
            # them by the integral of the SRF coefficients
            sat_sim_refl = np.nansum(sim_vec)
            sat_sim_refl = sat_sim_refl / np.nansum(sat_response)
            spectrum_bandlist.append(sat_sim_refl)

        # append to DF / dict
        resampled_dict = dict(zip(sat_bands, spectrum_bandlist))
        # append to out_DF
        out_list.append(resampled_dict)

    out_df = pd.DataFrame.from_dict(out_list, orient="columns")
    out_df = out_df.transpose()
    out_df = out_df.reset_index()
    out_df = out_df.rename(columns={"index": "sat_bands"})
    return out_df


def chlorophyll_carotiniod_constraint(lut_df: pd.DataFrame) -> pd.DataFrame:
    """
    Samples leaf carotenoid content based on leaf chlorophyll content
    using a truncated Poisson distribution based on empirical bounds
    from the ANGERS dataset as proposed by Wocher et al. (2020,
    https://doi.org/10.1016/j.jag.2020.102219).

    :param lut_df:
        DataFrame with (uncorrelated) leaf chlorophyll a+b (cab)
        and leaf carotenoid (car) samples for running PROSAIL
    :returns:
        LUT with car values sampled based on cab
    """
    def lower_boundary(cab: float | np.ndarray) -> float | np.ndarray:
        """
        lower boundary of the cab-car relationship reported by
        Wocher et al, 2020 based on the ANGERS dataset
        """
        return 0.223 / 4.684 * 3 * cab

    def upper_boundary(cab: float | np.ndarray) -> float | np.ndarray:
        """
        upper boundary of the cab-car relationship reported by
        Wocher et al, 2020 based on the ANGERS dataset
        """
        return 0.223 * 4.684 / 3 * cab + 2 * 0.986

    def cab_car_regression(cab: float | np.ndarray) -> float | np.ndarray:
        """
        empirical regression line between leaf chlorophyll and carotinoid
        content based on the ANGERS dataset reported by Wocher et al. (2020)
        """
        return 0.223 * cab + 0.986

    def sample_poisson(cab: float | np.ndarray) -> np.ndarray:
        """
        Sample leaf carotenoid values based on leaf chlorophyll
        values
        """
        if isinstance(cab, float):
            cab = [cab]
        car_samples = []
        for cab_val in cab:
            lower = lower_boundary(cab_val)
            upper = upper_boundary(cab_val)
            regression_val = cab_car_regression(cab_val)
            while True:
                car_sample = poisson.rvs(mu=regression_val)
                if lower < car_sample < upper:
                    car_samples.append(car_sample)
                    break
        return np.array(car_samples)

    # sample car based on cab as suggested by Wocher et al. (2020)
    cab = lut_df['cab']
    car_samples = sample_poisson(cab)
    # update the car column in the LUT
    out_df = lut_df.copy()
    out_df['car'] = car_samples

    return out_df


def calc_ccc(glai: float | np.ndarray, cab: float | np.ndarray) -> float | np.ndarray:
    """
    Calculate the Canopy Chlorophyll Content (CCC) in g/m2 from Green LAI and
    leaf chlorophyll a+b content (in ug/cm2).

    CCC = LAI * Cab * 0.01 (0.01 is used to convert ug/m2 into SI units, i.e., g/m2)

    :param glai:
        green leaf area index [m2 m-2]
    :param cab:
        leaf chlorophyll a+b content [ug cm-2]
    :returns:
        canopy chlorophyll content [g m-2]
    """
    return glai * cab * 0.01

def glai_ccc_constraint(lut_df: pd.DataFrame) -> pd.DataFrame:
    """
    Redistribute leaf chlorophyll values based on an empirical relationship between
    canopy chlorophyll content (CCC) and green leaf area index (GLAI) established
    using data from summer and winter wheat and barley and corn from multiple
    years from sites in Germany and Switzerland.

    :param lut_df:
        DataFrame with (uncorrelated) leaf chlorophyll a+b (cab)
        and leaf area index (lai) samples for running PROSAIL
    :returns:
        LUT with leaf and canopy chlorophyll values sampled based on lai
    """
    def lower_boundary(glai: float | np.ndarray) -> float | np.ndarray:
        """
        lower boundary of the empirical GLAI-CCC relationship
        """
        return 2*0.038146440139412485 * (glai / 1.681904541682302)**2 + 0.3336636117799512 * \
            glai / 1.681904541682302 - 0.055147504680912895

    def upper_boundary(glai: float | np.ndarray) -> float | np.ndarray:
        """
        upper boundary of the empirical GLAI-CCC relationship
        """
        return 0.5827304274822582 * 1.4663274265907047 * glai + 2*1.7426271784579451e-16

    def glai_ccc_regression(glai: float | np.ndarray) -> float | np.ndarray:
        """
        empirical linear regression between GLAI and CCC
        """
        return 3.6287601640166543e-29 + 0.5754806937355119 * glai

    def sample_gamma(glai: float | np.ndarray) -> np.ndarray:
        """
        Sample CCC values based on GLAI values using a truncated
        Gamma distribution between lower and upper bounds
        """
        if isinstance(glai, float):
            glai = [glai]
        ccc_samples = []
        for glai_val in glai:
            lower = lower_boundary(glai_val)
            upper = upper_boundary(glai_val)
            regression_val = glai_ccc_regression(glai_val)
            while True:
                ccc_sample = np.random.gamma(regression_val)
                if lower < ccc_sample < upper:
                    ccc_samples.append(ccc_sample)
                    break
        return np.array(ccc_samples)

    # redistribute CCC based on glai-ccc relationship
    glai = lut_df['lai']
    ccc_samples = sample_gamma(glai)
    # update the cab column in the LUT using CCC and LAI
    out_df = lut_df.copy()
    # cab = ccc / glai * 100
    cab_samples = ccc_samples / glai * 100
    out_df['cab'] = cab_samples

    return out_df


def green_is_valid(wvls: np.ndarray, spectrum: np.ndarray) -> bool:
    """
    Checks if a simulated spectrum is valid in terms of the position
    of its green-peak. Green peaks occuring at wavelengths >547nm are
    consider invalid according to Wocher et al. (2020,
    https://doi.org/10.1016/j.jag.2020.102219)

    :param wvls:
        array with wavelengths in nm
    :param spectrum:
        corresponding spectral data
    :returns:
        `True` if the green-peak is valid, else `False`.
    """
    # get array indices of wavelengths in "green" part of the spectrum
    green_wvls_idx = np.where(wvls == green_region[0])[0][0], np.where(wvls == green_region[1])[0][0]
    green_spectrum = spectrum[green_wvls_idx[0]:green_wvls_idx[1]]
    green_wvls = wvls[green_wvls_idx[0]:green_wvls_idx[1]]
    green_peak = green_wvls[np.argmax(green_spectrum)]
    # green peaks smaller than the threshold are considered invalid
    if green_peak < green_peak_threshold:
        return False
    else:
        return True

    # TODO: use this code to produce a figure for the paper
    # import matplotlib.pyplot as plt
    # plt.style.use('seaborn-darkgrid')
    # plt.plot(wvls[green_wvls_idx[0]:green_wvls_idx[1]], green_spectrum)
    # plt.vlines(green_peak_threshold, label='Green Peak Threshold', ymin=0, ymax=0.07,
    #            color='green', linewidth=2)
    # plt.vlines(green_peak, label='Green Peak of Spectrum', ymin=0, ymax=0.07,
    #            color='grey', linewidth=2, linestyle='dashed')
    # plt.xlabel('Wavelength [nm]')
    # plt.ylabel('Reflectance Factors [%]')
    # plt.legend()
    # plt.ylim(0,0.07)
    # plt.show()


def transform_lai(lai: pd.Series | np.ndarray | float, inverse: Optional[bool] = False
                  ) -> pd.Series | np.ndarray | float:
    """
    Apply a transformation of LAI values to linearize the model behavior to increase
    the SWIR band sensitivity towards high high values as suggested by Verhoef et al. (2018,
    https://doi.org/10.1016/j.rse.2017.08.006). The function works forwards and inverse.

    :param lai:
        leaf area index values to transform or transform back
    :param inverse:
        if True transformes transformed values back into LAI values
    :returns:
        transformed LAI values
    """
    # transformed LAI back to actual LAI
    if inverse:
        return -5 * np.log(-lai + 1)
    # LAI to transformed values
    else:
        return 1 - np.exp(-0.2 * lai)
    