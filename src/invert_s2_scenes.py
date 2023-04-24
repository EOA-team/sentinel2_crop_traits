'''
Inversion of Sentinel-2 data for crop trait retrieval using
PROSAIL lookup tables

@author Lukas Valentin Graf
'''

import numpy as np
import pandas as pd
import warnings

from eodal.config import get_settings
from eodal.core.band import Band
from eodal.core.raster import RasterCollection
from pathlib import Path
from typing import Dict, List, Optional
from rtm_inv.core.inversion import inv_img, retrieve_traits

logger = get_settings().logger
warnings.filterwarnings('ignore')

band_selection = [
    'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']


def invert_scenes(
    data_dir: Path,
    farms: List[str],
    n_solutions: Dict[str, int],
    cost_functions: Dict[str, str],
    aggregation_methods: Dict[str, str],
    lut_sizes: Dict[str, str],
    traits: Optional[List[str]] = ['lai', 'cab', 'ccc']
):
    """
    Lookup table based inversion of S2 imagery. The inversion setup can
    be adopted for each phenological macro-stage.

    :param data_dir:
        directory where PROSAIL LUTs and extracted S2 data are located
    :param farms:
        list of farms to process (data is organized by farms)
    :param n_solutions:
        number of solutions of the inversion to use per phenological
        macro-stage
    :param cost_functions:
        cost function per phenological macro-stage
    :param aggregation_methods:
        aggregation methods of the solutions found per phenological macro-stage
    :param lut_sizes:
        LUT sizes (i.e., number of PROSAIL spectra) to use per phenological
        macro-stage to run the inversion
    :param traits:
        list of traits to extract (this is used to find the correct LUT file).
        Defaults to 'lai', 'cab', and 'ccc'.
    """
    # loop over locations
    for farm in farms:
        farm_dir = data_dir.joinpath(farm)
        if not farm_dir.exists():
            continue
        # loop over scenes in farm, find lookup tables and apply the LUT based
        # inversion
        for scene_dir in farm_dir.glob('*.SAFE'):
            # load the Sentinel-2 data
            fpath_s2_raster = scene_dir.joinpath('SRF_S2.tiff')
            s2_ds = RasterCollection.from_multi_band_raster(
                fpath_raster=fpath_s2_raster)
            bands = s2_ds.band_names[:-1]
            s2_spectra = s2_ds.get_values(band_selection=bands)

            logger.info(f'{farm}: Started inversion of {scene_dir.name}')
            # find the LUTs generated and use them for inversion
            for fpath_lut in scene_dir.glob('*lut.pkl'):
                # check if the LUT contains the correct traits,
                # otherwise continue
                fname_lut = fpath_lut.name
                if not all([x in fname_lut for x in traits]):
                    continue
                lut = pd.read_pickle(fpath_lut)
                pheno_phase = fpath_lut.name.split('_')[0]
                if pheno_phase == 'all':
                    pheno_phase = 'all_phases'

                # draw sub-sample from LUT if required
                if lut_sizes[pheno_phase] < lut.shape[0]:
                    lut = lut.sample(lut_sizes[pheno_phase])

                # invert the S2 scene by comparing ProSAIL simulated
                # to S2 observed spectra
                s2_lut_spectra = lut[bands].values

                if isinstance(s2_spectra, np.ma.MaskedArray):
                    mask = s2_spectra.mask[0, :, :]
                    s2_spectra = s2_spectra.data
                else:
                    mask = np.zeros(
                        shape=(
                            s2_spectra.shape[1], s2_spectra.shape[2]
                        ),
                        dtype='uint8')
                    mask = mask.astype('bool')
                    mask[s2_spectra[0, :, :] == 0] = True

                lut_idxs, cost_function_values = inv_img(
                    lut=s2_lut_spectra,
                    img=s2_spectra,
                    mask=mask,
                    cost_function=cost_functions[pheno_phase],
                    n_solutions=n_solutions[pheno_phase],
                )
                trait_img, q05_img, q95_img = retrieve_traits(
                    lut=lut,
                    lut_idxs=lut_idxs,
                    traits=traits,
                    cost_function_values=cost_function_values,
                    measure=aggregation_methods[pheno_phase]
                )

                # save traits to file
                trait_collection = RasterCollection()
                for tdx, trait in enumerate(traits):
                    trait_collection.add_band(
                        Band,
                        geo_info=s2_ds[bands[0]].geo_info,
                        band_name=trait,
                        values=trait_img[tdx, :, :]
                    )
                    trait_collection.add_band(
                        Band,
                        geo_info=s2_ds[bands[0]].geo_info,
                        band_name=f'{trait}_q05',
                        values=q05_img[tdx, :, :]
                    )
                    trait_collection.add_band(
                        Band,
                        geo_info=s2_ds[bands[0]].geo_info,
                        band_name=f'{trait}_q95',
                        values=q95_img[tdx, :, :]
                    )
                # save lowest, median and highest cost function value
                highest_cost_function_vals = cost_function_values[-1, :, :]
                highest_cost_function_vals[np.isnan(trait_img[0, :, :])] = \
                    np.nan
                lowest_cost_function_vals = cost_function_values[0, :, :]
                lowest_cost_function_vals[np.isnan(trait_img[0, :, :])] = \
                    np.nan
                median_cost_function_vals = np.median(
                    cost_function_values[:, :, :], axis=0)
                median_cost_function_vals[np.isnan(trait_img[0, :, :])] = \
                    np.nan
                trait_collection.add_band(
                    Band,
                    geo_info=s2_ds[bands[0]].geo_info,
                    band_name='lowest_error',
                    values=lowest_cost_function_vals
                )
                trait_collection.add_band(
                    Band,
                    geo_info=s2_ds[bands[0]].geo_info,
                    band_name='highest_error',
                    values=highest_cost_function_vals
                )
                trait_collection.add_band(
                    Band,
                    geo_info=s2_ds[bands[0]].geo_info,
                    band_name='median_error',
                    values=median_cost_function_vals
                )

                # save to GeoTiff
                fname = scene_dir.joinpath(f'{pheno_phase}_lutinv_traits.tiff')
                trait_collection.to_rasterio(fpath_raster=fname)

            logger.info(f'{farm}: Finished inversion of {scene_dir.name}')


if __name__ == '__main__':

    farms = [
        'Strickhof', 'SwissFutureFarm', 'Witzwil', 'Arenenberg',
        'SwissFutureFarm_2019']
    data_dir = Path('./results/lut_based_inversion')

    cost_functions = {
        'all_phases': 'mae',
        'germination-endoftillering': 'rmse',
        'stemelongation-endofheading': 'mae',
        'flowering-fruitdevelopment-plantdead': 'mae'
    }
    aggregation_methods = {
        'all_phases': 'median',
        'germination-endoftillering': 'median',
        'stemelongation-endofheading': 'median',
        'flowering-fruitdevelopment-plantdead': 'median'
    }
    n_solutions = {
        'all_phases': 5000,
        'germination-endoftillering': 100,
        'stemelongation-endofheading': 5000,
        'flowering-fruitdevelopment-plantdead': 5000
    }
    lut_sizes = {
        'all_phases': 50000,
        'germination-endoftillering': 10000,
        'stemelongation-endofheading': 50000,
        'flowering-fruitdevelopment-plantdead': 50000
    }

    invert_scenes(
        data_dir,
        farms,
        n_solutions=n_solutions,
        cost_functions=cost_functions,
        aggregation_methods=aggregation_methods,
        lut_sizes=lut_sizes
    )
