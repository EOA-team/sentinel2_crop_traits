'''
This script recreates the Figure showing the temporal evolution of median
GLAI per field parcel and S2 scenes in calendar dates and thermal time.

@author: Lukas Valentin Graf
'''

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from eodal.core.raster import RasterCollection
from pathlib import Path

from utils import TraitLimits

mpl.rc('font', size=16)
plt.style.use('bmh')


def get_parcel_ts(
    farm: str,
    parcel_name: str,
    parcel_gdf: gpd.GeoDataFrame,
    img_dir: Path,
) -> pd.DataFrame:
    """
    Function to retrieve the median trait time series per parcel.
    To minimize the impact of boundary effects, the parcel geometries are
    buffered 20m inwards.

    :param farm:
        name of the farm (in order to find data)
    :param parcel_name:
        name of the parcel
    :param parcel_gdf:
        geometry of the parcel to extract trait values. Will be buffered
        20m inwards to avoid contamination by mixed pixels at the boundaries.
    :param img_dir:
        directory where inversion results (i.e., traits) are stored organized
        by Sentinel-2 scene
    :returns:
        DataFrame with median trait time series per parcel.
    """
    # loop over traits
    for _, trait in enumerate(traits):

        inv_res_insitu_points = pd.read_csv(
            trait_settings[trait]['trait_data'])
        inv_res_farm = inv_res_insitu_points[
            (inv_res_insitu_points.location == farm) &
            (inv_res_insitu_points.parcel == parcel_name)
        ].copy()
        # loop over Sentinel-2 scenes and save median, q05, q95 of the data
        ts_list = []
        for s2_scene in img_dir.glob('S2*.SAFE'):
            scene_id = s2_scene.name
            inv_res_scene = inv_res_farm[
                inv_res_farm.scene_id == scene_id].copy()
            if inv_res_scene.empty:
                continue
            # select the corresponding model output based on the BBCH rating
            bbch_stage = inv_res_scene["Macro-Stage"].iloc[0].replace(' ', '')
            try:
                agdd = int(np.round(inv_res_scene.gdd_cumsum.iloc[0]))
            except IndexError:
                print('aarg')
            fpath_inv_img = s2_scene.joinpath(
                f'{bbch_stage}_lutinv_traits.tiff')
            parcel_gdf_buffered = parcel_gdf.to_crs(epsg=2056).buffer(-20)
            trait_ds = RasterCollection.from_multi_band_raster(
                fpath_raster=fpath_inv_img,
                vector_features=parcel_gdf_buffered
            )
            ts_list.append({
                'q05': np.nanquantile(trait_ds[trait].values.data, 0.05),
                'q50': np.nanquantile(trait_ds[trait].values.data, 0.5),
                'q95': np.nanquantile(trait_ds[trait].values.data, 0.95),
                'sensing_date': pd.to_datetime(
                    scene_id.split('_')[2][0:8]).date(),
                'agdd': agdd
            })

        ts_df = pd.DataFrame(ts_list)
        ts_df.sort_values(by='agdd', inplace=True)
        ts_df['farm'] = farm
        ts_df['parcel'] = parcel_name
        return ts_df


if __name__ == '__main__':

    # in-situ measurements and inversion results of traits
    data_dir = Path('../../results/lut_based_inversion')
    # data_dir = Path('/mnt/ides/Lukas/04_Work/lut_based_inversion')
    out_dir = Path('../../results/Figures')

    traits = ['lai']
    trait_settings = {
        'lai': {
            'trait_name': 'Green Leaf Area Index',
            'trait_unit': r'$m^2$ $m^{-2}$',
            'trait_limits': TraitLimits(0, 8),
            'trait_data': data_dir.joinpath(
                'agdds_only').joinpath(
                    'validation_lai').joinpath(
                        'inv_res_joined_with_insitu_lai.csv')
        }
    }

    farms = ['Arenenberg', 'SwissFutureFarm', 'Strickhof', 'Witzwil']
    farm_data_dir = Path('../../data/auxiliary/field_parcels_ww_2022')
    fpath_farms_characteristics = Path(
        '../../data/in_situ_traits_2022/site_characteristics.ods')
    farms_characterisitcs = pd.read_excel(
        fpath_farms_characteristics, sheet_name='PhenomEn_Sites_2022_short')

    data_list = []
    for farm in farms:
        farm_characterisitcs = farms_characterisitcs[
            farms_characterisitcs.Location == farm]
        img_dir = data_dir.joinpath(farm)
        # plot maps per parcel and sensing date
        for parcel_name, parcel_df in farm_characterisitcs.groupby('Parcel'):
            # get shape of the parcel
            fpath_parcel_shp = farm_data_dir.joinpath(
                farm).joinpath(f'{parcel_name}.shp')
            parcel_gdf = gpd.read_file(fpath_parcel_shp)
            parcel_ts = get_parcel_ts(farm, parcel_name, parcel_gdf, img_dir)
            # save variety information
            parcel_ts['variety'] = parcel_df['Variety'].values[0]
            data_list.append(parcel_ts)

    df = pd.concat(data_list)
    df.index = [x for x in range(df.shape[0])]
    df['sensing_date'] = pd.to_datetime(df.sensing_date, format='%Y-%m-%d')

    # plot time series of the single parcels into one figure
    f, ax = plt.subplots(figsize=(25, 10), ncols=2, sharey=True)
    sns.lineplot(
        x='sensing_date',
        y='q50',
        hue='parcel',
        data=df,
        ax=ax[0],
        legend=False,
        marker='x')
    sns.lineplot(
        x='agdd', y='q50', hue='parcel', data=df, ax=ax[1], marker='x')
    ax[0].set_xlabel('Date (YYYY-MM)')
    ax[0].set_ylabel(r'Median GLAI [$m^2$ $m^{-2}$]')
    ax[0].set_title('(a) Calendar Dates')
    plt.setp(ax[0].xaxis.get_majorticklabels(), rotation=45)
    ax[1].set_title('(b) Thermal Time')
    ax[1].set_xlabel(r'Accumulated Growing Degree Days [$deg$ $C$]')
    plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=45)
    f.savefig(out_dir.joinpath('ts_dates_agdds.png'))

    # plot time series of the varieties into one plot
    f, ax = plt.subplots(figsize=(25, 10), ncols=2, sharey=True)
    sns.lineplot(
        x='sensing_date',
        y='q50',
        hue='variety',
        data=df,
        ax=ax[0],
        legend=False,
        marker='x')
    sns.lineplot(
        x='agdd', y='q50', hue='variety', data=df, ax=ax[1], marker='x')
    ax[0].set_xlabel('')
    ax[0].set_ylabel(r'Median GLAI [$m^2$ $m^{-2}$]')
    ax[0].set_title('Calendar Dates')
    plt.setp(ax[0].xaxis.get_majorticklabels(), rotation=45)
    ax[1].set_title('AGDDs')
    ax[1].set_xlabel('')
    plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=45)
    f.savefig(out_dir.joinpath('ts_dates_agdds_variety.png'))

    # save dataframe
    df.to_csv(out_dir.joinpath('ts_dates_agdds.csv'))
