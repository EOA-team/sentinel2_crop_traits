'''
Extract Sentinel-2 SRF data for selected field parcels over the growing season
and run PROSAIL simulations
'''

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from copy import deepcopy
from datetime import date
from eodal.config import get_settings
from eodal.operational.mapping.sentinel2 import Sentinel2Mapper
from eodal.utils.sentinel2 import ProcessingLevels, get_S2_platform_from_safe
from eodal.operational.mapping import MapperConfigs
from pathlib import Path
from rtm_inv.core.lookup_table import generate_lut
from typing import Any, Dict, List

from utils import get_farms

logger = get_settings().logger
band_selection = ['B02','B03','B04','B05','B06','B07','B8A','B11','B12']

def extract_s2_spectra(
    s2_mapper_config: Dict[str, Any]
) -> Sentinel2Mapper:
    """
    Extract S2 data using EOdal
    """
    # setup Sentinel-2 mapper to get the relevant scenes
    mapper_configs = MapperConfigs(
        band_names=band_selection,
        resampling_method=s2_mapper_config['resampling_method'],
        spatial_resolution=s2_mapper_config['spatial_resolution']
    )
    mapper_config = deepcopy(s2_mapper_config)
    del mapper_config['spatial_resolution']
    del mapper_config['resampling_method']
    mapper_config.update({'mapper_configs': mapper_configs})
    mapper = Sentinel2Mapper(**mapper_config)
    # query metadata records
    mapper.get_scenes()
    return mapper

def get_s2_spectra(
    output_dir: Path,
    lut_params_dir: Path,
    s2_mapper_config: Dict[str, Any],
    rtm_lut_config: Dict[str, Any],
    traits: List[str]
) -> None:
    """
    Extract S2 SRF for field parcel geometries and run PROSAIL in forward mode
    """
    trait_str = '-'.join(traits)

    mapper = extract_s2_spectra(s2_mapper_config)
    # get spectral data
    s2_data = mapper.get_complete_timeseries()
    # extraction is based on features (1 to N field parcel geometries)
    features = mapper.feature_collection['features']

    # loop over features and perform inversion
    for idx, feature in enumerate(features):
        feature_id = mapper.get_feature_ids()[idx]
        feature_scenes = s2_data[feature_id]
        feature_metadata = mapper.observations[feature_id]
        feature_metadata['sensing_time'] = pd.to_datetime(feature_metadata.sensing_time)
        output_dir_feature = output_dir.joinpath(feature_id)
        output_dir_feature.mkdir(exist_ok=True)
        # loop over mapper
        for feature_scene in feature_scenes:
            # make sure we're looking at the right metadata
            metadata = feature_metadata[
                feature_metadata.sensing_time.dt.date == \
                feature_scene.scene_properties.acquisition_time.date()
            ]
            # to speed model development introduce some calendar checks, i.e., we
            # don't need to run all simulations all the time
            scene_month = pd.to_datetime(metadata.sensing_time.iloc[0]).month
            pheno_phase_selection = None
            if scene_month in [3]:
                pheno_phase_selection = ['all_phases', 'germination-endoftillering', 'stemelongation-endofheading']
            elif scene_month in [4]:
                pheno_phase_selection = ['all_phases', 'germination-endoftillering', 'stemelongation-endofheading']
            elif scene_month in [5, 6]:
                pheno_phase_selection = ['all_phases', 'stemelongation-endofheading', 'flowering-fruitdevelopment-plantdead']

            # get viewing and illumination angles for PROSAIL run
            angle_dict = {
                'solar_zenith_angle': metadata['sun_zenith_angle'].iloc[0],
                'solar_azimuth_angle': metadata['sun_azimuth_angle'].iloc[0],
                'viewing_zenith_angle': metadata['sensor_zenith_angle'].iloc[0],
                'viewing_azimuth_angle': metadata['sensor_azimuth_angle'].iloc[0]
            }
            # get platform
            platform = get_S2_platform_from_safe(
                dot_safe_name=metadata['product_uri'].iloc[0]
            )
            # map to full platform name
            full_names = {'S2A': 'Sentinel2A', 'S2B': 'Sentinel2B'}
            platform = full_names[platform]
            rtm_lut_config.update({'sensor': platform})

            # prepare S2 spectra
            band_names = feature_scene.band_names
            if 'SCL' in band_names: band_names.remove('SCL')
            # mask clouds and shadows if processing level is L2A
            if feature_scene.scene_properties.processing_level == ProcessingLevels.L2A:
                try:
                    feature_scene.mask_clouds_and_shadows(
                        bands_to_mask=band_names, inplace=True
                    )
                except Exception as e:
                    logger.error(e)
                    continue

            # check for black-fill and no-data after masking
            if feature_scene.is_blackfilled:
                logger.info(f'Feature {feature_id}: {metadata.product_uri.iloc[0]} is black-filled -> skipping')
                continue
            if feature_scene['blue'].values.mask.all():
                logger.info(f'Feature {feature_id}: {metadata.product_uri.iloc[0]} is no-data -> skipping')
                continue

            # save spectra and PROSAIL simulations in a sub-directory for each scene
            res_dir_scene = output_dir_feature.joinpath(metadata['product_uri'].iloc[0])
            # if res_dir_scene.exists():
            #     logger.info(f'{res_dir_scene.name} exists already - skipping')
            #     continue
            res_dir_scene.mkdir(exist_ok=True)

            # save S2 spectra to disk for analysis
            feature_scene.to_rasterio(res_dir_scene.joinpath('SRF_S2.tiff'))
            fig_s2_rgb = feature_scene.plot_multiple_bands(['blue', 'green', 'red'])
            fname_fig_s2_rgb = res_dir_scene.joinpath('SRF_S2_VIS.png')
            fig_s2_rgb.savefig(fname_fig_s2_rgb)
            plt.close(fig_s2_rgb)

            # run PROSAIL forward runs for the different parametrizations available
            logger.info(f'Feature {feature_id}: {metadata.product_uri.iloc[0]} starting PROSAIL runs')
            for lut_params_pheno in lut_params_dir.glob('*.csv'):
                pheno_phases = lut_params_pheno.name.split('etal')[-1].split('.')[0][1::]
                if pheno_phase_selection is not None:
                    if pheno_phases not in pheno_phase_selection:
                        continue
        
                # generate lookup-table for the current angles
                fpath_lut = res_dir_scene.joinpath(f'{pheno_phases}_{trait_str}_lut.pkl')
                # if LUT exists, continue, else generate it
                if not fpath_lut.exists():
                    lut_inp = rtm_lut_config.copy()
                    lut_inp.update(angle_dict)
                    lut_inp['lut_params'] = lut_params_pheno
                    lut = generate_lut(**lut_inp)
                    # special case CCC (Canopy Chlorophyll Content) -> this is not a direct RTM output
                    if 'ccc' in traits:
                        lut['ccc'] = lut['lai'] * lut['cab']
                        # convert to g m-2 as this is the more common unit
                        # ug -> g: factor 1e-6; cm2 -> m2: factor 1e-4
                        lut['ccc'] *= 1e-2
                else:
                    continue
        
                # prepare LUT for model training
                lut = lut[band_selection + traits].copy()
                lut.dropna(inplace=True)
        
                # save LUT to file
                if not fpath_lut.exists():
                    with open(fpath_lut, 'wb+') as f:
                        pickle.dump(lut, f)

            logger.info(f'Feature {feature_id}: {metadata.product_uri.iloc[0]} finished PROSAIL runs')

if __name__ == '__main__':

    import sys
    data_dir = Path('/home/graflu/public/Evaluation/Projects/KP0031_lgraf_PhenomEn/02_Field-Campaigns')
    year = 2022 # 2022
    farms = [sys.argv[1]] #['Strickhof', 'Arenenberg', 'Witzwil']

    # get field parcel geometries organized by farm
    farm_gdf_dict = get_farms(data_dir, farms, year)

    data_dir = Path('/home/graflu/public/Evaluation/Projects/KP0031_lgraf_PhenomEn/03_WW_Traits/PhenomEn22/trait_retrieval')
    # data_dir = Path('/home/graflu/public/Evaluation/Projects/KP0031_lgraf_PhenomEn/03_WW_Traits/DigiN19')
    out_dir = data_dir.joinpath('lut_based_inversion')
    out_dir.mkdir(exist_ok=True)

    # spectral response function of Sentinel-2 for resampling PROSAIL output
    fpath_srf = Path(
        '/home/graflu/public/Evaluation/Satellite_data/Sentinel-2/Documentation/S2_Specsheets/S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.1.xlsx'
    )
    # RTM configurations for lookup-table generation
    rtm_lut_config = {
        'lut_size': 50000,
        'fpath_srf': fpath_srf,
        'remove_invalid_green_peaks': True,
        'sampling_method': 'FRS',
        'linearize_lai': False
    }
    # directory with LUT parameters for different phenological macro-stages
    lut_params_dir = Path('lut_params')

    # target trait(s)
    traits = ['lai', 'ccc']

    # loop over farms
    for farm, geom in farm_gdf_dict.items():
        logger.info(f'Working on {farm}')
        # S2 configuration (for data extraction and pre-processing)
        s2_mapper_config = {
            'feature_collection': geom,
            'cloud_cover_threshold': 50.,
            'date_start': date(2021,3,1), # date(2019,4,14),
            'date_end': date(2022,7,31), # date(2019,5,22),
            'unique_id_attribute': 'farm',
            'spatial_resolution': 10.,
            'resampling_method': cv2.INTER_NEAREST_EXACT,
            'processing_level': ProcessingLevels.L2A
        }

        get_s2_spectra(
            output_dir=out_dir,
            lut_params_dir=lut_params_dir,
            s2_mapper_config=s2_mapper_config,
            rtm_lut_config=rtm_lut_config,
            traits=traits
        )

        logger.info(f'Finished working on {farm}')
