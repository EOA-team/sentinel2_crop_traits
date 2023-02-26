'''
Implementation of the temperature-based phenology model (from field
phenotyping) to combine RTM outputs for trait retrieval.

@author Lukas Valentin Graf
'''

import geopandas as gpd
import numpy as np
import pandas as pd

from eodal.core.raster import RasterCollection
from pathlib import Path
from typing import List, Optional, Tuple

from utils import (
    calc_ww_gdd,
    from_agrometeo,
    from_meteoswiss,
    plot_trait_maps,
    read_site_characteristics
)

MERGE_TOLERANCE = 20  # Growing Degrees [deg C]

# AGDD windows for switching between phenological macro-stages
GDD_CRITICAL_SE = [740, 870]  # Growing Degrees [deg C] # updated with 2021 + 2022 data
GDD_DEFAULT_SE = 800
GDD_CRITICAL_AN = [1380, 1600]  # Growing Degrees [deg C]
GDD_DEFAULT_AN = 1490

def combine_model_results(
    inv_res: pd.DataFrame,
    traits: Optional[List[str]] = ['lai','ccc'],
    use_temperature_only: Optional[bool] = False
) -> pd.DataFrame:
    """
    Combine results of the different phenological models using
    a mixture of physiological plausibility (temperature sums) and
    inversion error (value of the cost function) to identify the switch
    between phases.

    The algorithm works as follows:

        Until the first GDD critical threshold is reached, the outputs
        of the germination-tillering model are used. Then, let the 
        cost function value decide when to switch into the next phase.

        Then, the stemelongation-endofheading model is used until we
        reach the next critical GDD threshold. Here, the procedure is the
        same as in the previous phase.

        We assign all remaining data points to the last phenological phase.

    :param inv_res:
        inversion results (i.e., trait values retrieved from imagery)
    :param traits:
        traits to retrieve. `lai` and `ccc` by default.
    :param use_temperature_onyl:
        use heat units only for determining the phenological marco-stage
        (baseline model). False by default.
    :returns:
        `DataFrame` with model results (i.e., trait values) from different
        phenological phases
    """

    inv_res_combined = []
    _inv_res = inv_res.copy()
    for _, point_df in _inv_res.groupby('point_id'):
        # sort values by GDDs
        point_df.sort_values(by='gdd_cumsum', inplace=True)
        # filter by SCL (retain classes 4 and 5, only)
        point_df = point_df[point_df.SCL.isin([4,5])].copy()
        # # drop NaNs
        # trait_cols = [x for x in point_df.columns if x.startswith(trait)]
        # point_df.dropna(inplace=True, subset=trait_cols)
        # assign new indices to ensure indices are ordered
        point_df.index = [x for x in range(point_df.shape[0])]

        # open new columns for storing the "final" trait values
        for trait_name in traits:
            point_df[f'{trait_name} (Phenology)'] = -999.
            point_df[f'{trait_name}_q05 (Phenology)'] = -999.
            point_df[f'{trait_name}_q95 (Phenology)'] = -999.
        point_df['Macro-Stage'] = ''
        # until the first GDD critical threshold is reached, the outputs
        # of the germination-tillering model are used
        # then, let the uncertainty decide when to switch into the next phase
        switch_idx = None
        # search for switch idx between lower and upper T threshold
        for idx, item in point_df[(point_df.gdd_cumsum >= GDD_CRITICAL_SE[0]) & (point_df.gdd_cumsum <= GDD_CRITICAL_SE[1])].iterrows():
            if item[f'error_germination-endoftillering'] >= item[f'error_stemelongation-endofheading']:
                switch_idx = idx
                break
        if use_temperature_only: switch_idx = None

        # if no switch was found use default GDD thresholds
        if switch_idx is None:
            point_df.loc[point_df.gdd_cumsum < GDD_DEFAULT_SE, 'Macro-Stage'] = 'germination - end of tillering'
            point_df.loc[point_df.gdd_cumsum > GDD_DEFAULT_SE, 'Macro-Stage'] = 'stem elongation - end of heading'
            point_df.loc[point_df.gdd_cumsum == GDD_DEFAULT_SE, 'Macro-Stage'] = 'stem elongation - end of heading'

            for trait_name in traits:
                point_df.loc[point_df.gdd_cumsum < GDD_DEFAULT_SE, f'{trait_name} (Phenology)'] = \
                    point_df[point_df.gdd_cumsum < GDD_DEFAULT_SE][f'{trait_name}_germination-endoftillering'].copy()
                point_df.loc[point_df.gdd_cumsum < GDD_DEFAULT_SE, f'{trait_name}_q05 (Phenology)'] = \
                    point_df[point_df.gdd_cumsum < GDD_DEFAULT_SE][f'{trait_name}_q05_germination-endoftillering'].copy()
                point_df.loc[point_df.gdd_cumsum < GDD_DEFAULT_SE, f'{trait_name}_q95 (Phenology)'] = \
                    point_df[point_df.gdd_cumsum < GDD_DEFAULT_SE][f'{trait_name}_q95_germination-endoftillering'].copy()
                
                # at the temperatue threshold we use the mean of both models
                point_df.loc[point_df.gdd_cumsum == GDD_DEFAULT_SE, f'{trait_name} (Phenology)'] = \
                    0.5 * (
                        point_df[point_df.gdd_cumsum == GDD_DEFAULT_SE][f'{trait_name}_stemelongation-endofheading'].copy() +
                        point_df[point_df.gdd_cumsum == GDD_DEFAULT_SE][f'{trait_name}_germination-endoftillering'].copy()
                    )
                point_df.loc[point_df.gdd_cumsum == GDD_DEFAULT_SE, f'{trait_name}_q05 (Phenology)'] = \
                    0.5 * (
                        point_df[point_df.gdd_cumsum == GDD_DEFAULT_SE][f'{trait_name}_q05_stemelongation-endofheading'].copy() +
                        point_df[point_df.gdd_cumsum == GDD_DEFAULT_SE][f'{trait_name}_q05_germination-endoftillering'].copy()
                    )
                point_df.loc[point_df.gdd_cumsum == GDD_DEFAULT_SE, f'{trait_name}_q95 (Phenology)'] = \
                    0.5 * (
                        point_df[point_df.gdd_cumsum == GDD_DEFAULT_SE][f'{trait_name}_q95_stemelongation-endofheading'].copy() +
                        point_df[point_df.gdd_cumsum == GDD_DEFAULT_SE][f'{trait_name}_q95_germination-endoftillering'].copy()
                    )

                point_df.loc[point_df.gdd_cumsum > GDD_DEFAULT_SE, f'{trait_name} (Phenology)'] = \
                    point_df[point_df.gdd_cumsum > GDD_DEFAULT_SE][f'{trait_name}_stemelongation-endofheading'].copy()
                point_df.loc[point_df.gdd_cumsum > GDD_DEFAULT_SE, f'{trait_name}_q05 (Phenology)'] = \
                    point_df[point_df.gdd_cumsum > GDD_DEFAULT_SE][f'{trait_name}_q05_stemelongation-endofheading'].copy()
                point_df.loc[point_df.gdd_cumsum > GDD_DEFAULT_SE, f'{trait_name}_q95 (Phenology)'] = \
                    point_df[point_df.gdd_cumsum > GDD_DEFAULT_SE][f'{trait_name}_q95_stemelongation-endofheading'].copy()
                

        else:
            point_df.loc[point_df.index < switch_idx, 'Macro-Stage'] = 'germination - end of tillering'
            point_df.loc[point_df.index == switch_idx, 'Macro-Stage'] = 'stem elongation - end of heading'
            point_df.loc[point_df.index > switch_idx, 'Macro-Stage'] = 'stem elongation - end of heading'

            for trait_name in traits:
                point_df.loc[point_df.index < switch_idx, f'{trait_name} (Phenology)'] = \
                    point_df[point_df.index < switch_idx][f'{trait_name}_germination-endoftillering'].copy()
                point_df.loc[point_df.index < switch_idx, f'{trait_name}_q05 (Phenology)'] = \
                    point_df[point_df.index < switch_idx][f'{trait_name}_q05_germination-endoftillering'].copy()
                point_df.loc[point_df.index < switch_idx, f'{trait_name}_q95 (Phenology)'] = \
                    point_df[point_df.index < switch_idx][f'{trait_name}_q95_germination-endoftillering'].copy()

                # at the switch index we use the mean of both models
                point_df.loc[point_df.index == switch_idx, f'{trait_name} (Phenology)'] = \
                    0.5 * (
                        point_df[point_df.index == switch_idx][f'{trait_name}_stemelongation-endofheading'].copy() +
                        point_df[point_df.index == switch_idx][f'{trait_name}_germination-endoftillering'].copy()
                )
                point_df.loc[point_df.index == switch_idx, f'{trait_name}_q05 (Phenology)'] = \
                    0.5 * (
                        point_df[point_df.index == switch_idx][f'{trait_name}_q05_stemelongation-endofheading'].copy() +
                        point_df[point_df.index == switch_idx][f'{trait_name}_q05_germination-endoftillering'].copy()
                    )
                point_df.loc[point_df.index == switch_idx, f'{trait_name}_q95 (Phenology)'] = \
                    0.5 * (
                        point_df[point_df.index == switch_idx][f'{trait_name}_q95_stemelongation-endofheading'].copy() +
                        point_df[point_df.index == switch_idx][f'{trait_name}_q95_germination-endoftillering'].copy()
                    )

                point_df.loc[point_df.index > switch_idx, f'{trait_name} (Phenology)'] = \
                    point_df[point_df.index > switch_idx][f'{trait_name}_stemelongation-endofheading'].copy()
                point_df.loc[point_df.index > switch_idx, f'{trait_name}_q05 (Phenology)'] = \
                    point_df[point_df.index > switch_idx][f'{trait_name}_q05_stemelongation-endofheading'].copy()
                point_df.loc[point_df.index > switch_idx, f'{trait_name}_q95 (Phenology)'] = \
                    point_df[point_df.index > switch_idx][f'{trait_name}_q95_stemelongation-endofheading'].copy()

        # switch into the next phase
        switch_idx = None
        # search for switch idx between lower and upper T threshold
        for idx, item in point_df[(point_df.gdd_cumsum >= GDD_CRITICAL_AN[0]) & (point_df.gdd_cumsum <= GDD_CRITICAL_AN[1])].iterrows():
            if item[f'error_flowering-fruitdevelopment-plantdead'] <= item[f'error_stemelongation-endofheading']:
                switch_idx = idx
                break
        if use_temperature_only: switch_idx = None

        # if no switch was found use GDD threshold
        if switch_idx is None:
            
            point_df.loc[point_df.gdd_cumsum == GDD_DEFAULT_AN, 'Macro-Stage'] = 'flowering - fruit development - plant dead'
            point_df.loc[point_df.gdd_cumsum > GDD_DEFAULT_AN, 'Macro-Stage'] = 'flowering - fruit development - plant dead'
            
            for trait_name in traits:
                # at the temperatue threshold we use the mean of both models
                point_df.loc[point_df.gdd_cumsum == GDD_DEFAULT_AN, f'{trait_name} (Phenology)'] = \
                    0.5 * (
                        point_df[point_df.gdd_cumsum == GDD_DEFAULT_AN][f'{trait_name}_flowering-fruitdevelopment-plantdead'].copy() +
                        point_df[point_df.gdd_cumsum == GDD_DEFAULT_AN][f'{trait_name}_stemelongation-endofheading'].copy()
                    )
                point_df.loc[point_df.gdd_cumsum == GDD_DEFAULT_AN, f'{trait_name}_q05 (Phenology)'] = \
                    0.5 * (
                        point_df[point_df.gdd_cumsum == GDD_DEFAULT_AN][f'{trait_name}_q05_flowering-fruitdevelopment-plantdead'].copy() +
                        point_df[point_df.gdd_cumsum == GDD_DEFAULT_AN][f'{trait_name}_q05_stemelongation-endofheading'].copy()
                    )
                point_df.loc[point_df.gdd_cumsum == GDD_DEFAULT_AN, f'{trait_name}_q95 (Phenology)'] = \
                    0.5 * (
                        point_df[point_df.gdd_cumsum == GDD_DEFAULT_AN][f'{trait_name}_q95_flowering-fruitdevelopment-plantdead'].copy() +
                        point_df[point_df.gdd_cumsum == GDD_DEFAULT_AN][f'{trait_name}_q95_stemelongation-endofheading'].copy()
                    )

                point_df.loc[point_df.gdd_cumsum > GDD_DEFAULT_AN, f'{trait_name} (Phenology)'] = \
                    point_df[point_df.gdd_cumsum > GDD_DEFAULT_AN][f'{trait_name}_flowering-fruitdevelopment-plantdead'].copy()
                point_df.loc[point_df.gdd_cumsum > GDD_DEFAULT_AN, f'{trait_name}_q05 (Phenology)'] = \
                    point_df[point_df.gdd_cumsum > GDD_DEFAULT_AN][f'{trait_name}_q05_flowering-fruitdevelopment-plantdead'].copy()
                point_df.loc[point_df.gdd_cumsum > GDD_DEFAULT_AN, f'{trait_name}_q95 (Phenology)'] = \
                    point_df[point_df.gdd_cumsum > GDD_DEFAULT_AN][f'{trait_name}_q95_flowering-fruitdevelopment-plantdead'].copy()

        else:
            point_df.loc[point_df.index == switch_idx, 'Macro-Stage'] = 'flowering - fruit development - plant dead'
            point_df.loc[point_df.index > switch_idx, 'Macro-Stage'] = 'flowering - fruit development - plant dead'
            
            for trait_name in traits:
                # at the switch index we use the mean of both models
                point_df.loc[point_df.index == switch_idx, f'{trait_name} (Phenology)'] = \
                    0.5 * (
                        point_df[point_df.index == switch_idx][f'{trait_name}_flowering-fruitdevelopment-plantdead'].copy() +
                        point_df[point_df.index == switch_idx][f'{trait_name}_stemelongation-endofheading'].copy()
                    )
                point_df.loc[point_df.index == switch_idx, f'{trait_name}_q05 (Phenology)'] = \
                    0.5 * (
                        point_df[point_df.index == switch_idx][f'{trait_name}_q05_flowering-fruitdevelopment-plantdead'].copy() +
                        point_df[point_df.index == switch_idx][f'{trait_name}_q05_stemelongation-endofheading'].copy()
                    )
                point_df.loc[point_df.index == switch_idx, f'{trait_name}_q95 (Phenology)'] = \
                    0.5 * (
                        point_df[point_df.index == switch_idx][f'{trait_name}_q95_flowering-fruitdevelopment-plantdead'].copy() +
                        point_df[point_df.index == switch_idx][f'{trait_name}_q95_stemelongation-endofheading'].copy()
                    )

                point_df.loc[point_df.index > switch_idx, f'{trait_name} (Phenology)'] = \
                    point_df[point_df.index > switch_idx][f'{trait_name}_flowering-fruitdevelopment-plantdead'].copy()
                point_df.loc[point_df.index > switch_idx, f'{trait_name}_q05 (Phenology)'] = \
                    point_df[point_df.index > switch_idx][f'{trait_name}_q05_flowering-fruitdevelopment-plantdead'].copy()
                point_df.loc[point_df.index > switch_idx, f'{trait_name}_q95 (Phenology)'] = \
                    point_df[point_df.index > switch_idx][f'{trait_name}_q95_flowering-fruitdevelopment-plantdead'].copy()
        # append results to list
        inv_res_combined.append(point_df)

    return pd.concat(inv_res_combined)

def combine_model_results_with_insitu(
    sampling_point_dir: Path,
    field_parcel_dir: Path,
    meteo_data_dir: Path,
    site_char_df: pd.DataFrame,
    inv_res_dir: Path,
    res_dir: Path,
    traits: List[str],
    trait_labels: List[str],
    trait_limits: List[Tuple[float, float]],
    plot: Optional[bool] = False,
    use_temperature_only: Optional[bool] = False
) -> None:
    """
    Function to combine predictions from PROSAIL inversion with in-situ trait
    measurements using thermal time (GDDs)

    :param sampling_point_dir:
        directory with vector files with locations of sampling points
        where trait measurements were conducted
    :param field_parcel_dir:
        directory where field parcel geometries are stored
    :param meteo_data_dir:
        directory where meteorological data (temperature) is stored
    :param site_char_df:
        field calendard DataFrame
    :param gpr_res_dir:
        directory with outputs of the GPR runs (multiple models)
    :param res_dir:
        directory where to store results (mainly plots and CSV with
        combined data)
    :param plot:
        if False (default) no maps of the obtained prediction raster are
        plotted
    :param use_temperature_onyl:
        use heat units only for determining the phenological marco-stage
        (baseline model). False by default.
    """
    # loop over sites
    site_char_df_grouped = site_char_df.groupby(by='Location')
    large_res_list = []
    for location_name, location_df in site_char_df_grouped:
        # search for the inversion results
        inv_res_dir_location = inv_res_dir.joinpath(location_name)
        if not inv_res_dir_location.exists():
            continue
        # get weather station data (single weather station per site)
        location_name_nowhitespace = location_name.replace(' ', '').lower()
        meteo_file = next(meteo_data_dir.glob(f'*_{location_name_nowhitespace}_daily_mean_temperature*'))
        # data can either come from Agrometeo or MeteoSwiss
        if meteo_file.name.endswith('meteoswiss.txt'):
            raw_meteo = from_meteoswiss(meteo_file)
            column_tmean = 'tre200d0'
        else:
            raw_meteo = from_agrometeo(meteo_file)
            columns = raw_meteo.columns
            column_tmean = [x for x in columns if x.endswith('Temperatur Durchschnitt +2 m (°C)')][0]
        # calculate GDDs
        gdd_meteo = calc_ww_gdd(
            temp_df=raw_meteo, column_tmean=column_tmean
        )
        res_dir_location = res_dir.joinpath(location_name)
        res_dir_location.mkdir(exist_ok=True)
        # shrink to time period between sowing and harvest and calculate cumulative GDDs
        # per field parcel geometry
        parcels = location_df.groupby('Parcel')
        for parcel_name, parcel_df in parcels:
            sowing_date = parcel_df['Sowing Date'].values[0]
            harvest_date = parcel_df['Harvest Date'].values[0]
            gdd_meteo_parcel = gdd_meteo[sowing_date:harvest_date].copy()
            gdd_meteo_parcel['gdd_cumsum'] = gdd_meteo_parcel.gdd.cumsum()
            gdd_meteo_parcel.reset_index(inplace=True)
            # get the field parcel geometry so that the correct in-situ observations are selected
            # I know that's ugly
            if parcel_name == 'Parzelle 35': parcel_name = 'Parzelle35'
            fpath_parcel_geom = field_parcel_dir.joinpath(f'{parcel_name.replace(" ","")}.shp')
            parcel_gdf = gpd.read_file(fpath_parcel_geom)
            # get the corresponding in-situ sampling points
            try:
                parcel_points = gpd.read_file(
                    next(
                        sampling_point_dir.joinpath(location_name).glob(f'{parcel_name.lower()}.gpkg')
                    )
                )
            except Exception as e:
                print(f'Could not find {location_name} {parcel_name.lower()}: {e}')
                continue
            res_dir_parcel = res_dir_location.joinpath(parcel_name)
            res_dir_parcel.mkdir(exist_ok=True)
            # find the inversion results available, extract data and assign GDDs
            inv_res_data_list = []
            for fpath_inv_res in inv_res_dir_location.glob('*.SAFE'):
                inv_res_date = pd.to_datetime(fpath_inv_res.name.split('_')[2][0:8])

                # check if observations is between sowing and harvest
                if inv_res_date < sowing_date or inv_res_date > harvest_date:
                        continue
                # get cumulative GDD of the date
                inv_res_gdd = gdd_meteo_parcel[
                    gdd_meteo_parcel.date == inv_res_date
                ]['gdd_cumsum'].values

                # get S2 spectra
                fpath_s2_srf = next(fpath_inv_res.glob('SRF*.tiff'))
                s2_srf_ds = RasterCollection.from_multi_band_raster(
                    fpath_s2_srf,
                    vector_features=parcel_gdf
                )

                # loop over pixels and save inversion results and spectral data
                for point_id, parcel_point in parcel_points.groupby('point_id'):
                    # save predictions and metadata
                    inv_res_data = {
                        'scene_id': fpath_inv_res.name,
                        'date': inv_res_date.date(),
                        'gdd_cumsum': inv_res_gdd[0],
                        'point_id': point_id
                    }
                    # loop over inversion results from different PROSAIL runs
                    for fpath_model in fpath_inv_res.glob('*lutinv*.tiff'):
                        pred_ds = RasterCollection.from_multi_band_raster(
                            fpath_model,
                            vector_features=parcel_gdf
                        )
                        pheno_phase_model = fpath_model.name.split('_')[0]
                        # get pixel values at sampling points
                        parcel_point_utm = parcel_point.to_crs(pred_ds[pred_ds.band_names[0]].crs)
                        parcel_point_buffered = parcel_point_utm.buffer(10)
                        pred_ds_clipped = pred_ds.clip_bands(
                            clipping_bounds=parcel_point_buffered.geometry.values[0]
                        )

                        for trait in traits:
                            inv_res_data[f'{trait}_{pheno_phase_model}'] = \
                                pred_ds_clipped[trait].reduce(['mean'])[0]['mean']
                            inv_res_data[f'{trait}_q05_{pheno_phase_model}'] = \
                                pred_ds_clipped[f'{trait}_q05'].reduce(['mean'])[0]['mean']
                            inv_res_data[f'{trait}_q95_{pheno_phase_model}'] = \
                                pred_ds_clipped[f'{trait}_q95'].reduce(['mean'])[0]['mean']
                        # get the value (error) of the cost function found
                        try:
                            inv_res_data[f'error_{pheno_phase_model}'] = \
                                pred_ds_clipped['median_error'].reduce(['mean'])[0]['mean']
                        except KeyError:
                                continue

                        s2_srf_clipped = s2_srf_ds.clip_bands(
                            clipping_bounds=parcel_point_buffered.geometry.values[0]
                        )
                        # get the most common SCL class and set the observation to that class
                        most_common_scl = np.argmax(
                            np.bincount(s2_srf_clipped['SCL'].values.data.flatten().astype(int))
                        )
                        inv_res_data['SCL'] = most_common_scl
                        sel_keys = [x for x in s2_srf_clipped.band_names if x != 'SCL']
                        for sel_key in sel_keys:
                            inv_res_data[sel_key] = s2_srf_clipped[sel_key].reduce(['mean'])[0]['mean']

                        if plot:
                            plot_trait_maps(
                                traits=traits,
                                trait_labels=trait_labels,
                                trait_limits=trait_limits,
                                inv_res_ds=pred_ds,
                                parcel_points=parcel_points,
                                parcel_gdf=parcel_gdf,
                                inv_res_date=inv_res_date,
                                pheno_phase_model=pheno_phase_model,
                                res_dir_parcel=res_dir_parcel
                            )
                    inv_res_data_list.append(inv_res_data)

            _res_inv_df = pd.DataFrame(inv_res_data_list)
            # combine phenological models
            res_inv_df = combine_model_results(
                inv_res=_res_inv_df,
                traits=traits,
                use_temperature_only=use_temperature_only
            )
            # add parcel and location name
            res_inv_df['parcel'] = parcel_name
            res_inv_df['location'] = location_name
            large_res_list.append(res_inv_df)

    large_df = pd.concat(large_res_list)
    fname = res_dir.joinpath('inv_res_gdd_insitu_points.csv')
    large_df.to_csv(fname)


if __name__ == '__main__':

    # directory where weather station and field parcel geometry data is stored
    aux_data_dir = Path('../auxiliary')
    meteo_data_dir = aux_data_dir.joinpath('Meteo')
    sampling_point_dir = aux_data_dir.joinpath('sampling_points_ww_2022')
    field_parcel_dir = aux_data_dir.joinpath('field_parcels_ww_2022')

    # field calendars
    trait_dir = Path('../in-situ_traits_2022')
    fpath_site_char = trait_dir.parent.joinpath('site_characteristics.ods')
    site_char_df = read_site_characteristics(fpath=fpath_site_char)

    # directory where inversion results are stored (stored by S2 acquisitions)
    inv_res_dir = Path('../results/lut_based_inversion')

    # traits to extract
    traits = ['lai', 'ccc']
    trait_labels = [r'$m^2$ $m^{-2}$', r'$g$ $m^{-2}$']
    trait_limits = [(0,8), (0,4)]
    use_temperature_only_opts = [True, False]

    for use_temperature_only in use_temperature_only_opts:
        # directory for storing results
        dirname = 'agdds_and_s2'
        if use_temperature_only: dirname = 'agdds_only'
        res_dir = inv_res_dir.joinpath(dirname)
        res_dir.mkdir(exist_ok=True)

        combine_model_results_with_insitu(
            sampling_point_dir=sampling_point_dir,
            field_parcel_dir=field_parcel_dir,
            meteo_data_dir=meteo_data_dir,
            site_char_df=site_char_df,
            inv_res_dir=inv_res_dir,
            res_dir=res_dir,
            traits=traits,
            trait_labels=trait_labels,
            trait_limits=trait_limits,
            plot=False,
            use_temperature_only=use_temperature_only
        )
