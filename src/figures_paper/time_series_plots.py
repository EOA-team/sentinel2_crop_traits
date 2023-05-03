'''
This script recreates the Figure showing the temporal evolution of median
GLAI, CCC and Cab per field parcel and S2 scenes in calendar dates and
thermal time.

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
from typing import List

from utils import TraitLimits

mpl.rc('font', size=20)
plt.style.use('bmh')


def get_parcel_ts(
    farm: str,
    parcel_name: str,
    parcel_gdf: gpd.GeoDataFrame,
    img_dir: Path,
    traits: List[str]
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

    inv_res_insitu_points = pd.read_csv(
        trait_settings['lai']['trait_data'])
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
        # loop over traits
        res_dict = {}
        for _, trait in enumerate(traits):
            res_dict.update({
                f'{trait}_q05': np.nanquantile(trait_ds[trait].values.data,
                                               0.05),
                f'{trait}_q50': np.nanquantile(trait_ds[trait].values.data,
                                               0.5),
                f'{trait}_q95': np.nanquantile(trait_ds[trait].values.data,
                                               0.95)})

        res_dict.update({
            'sensing_date': pd.to_datetime(scene_id.split('_')[2][0:8]).date(),
            'agdd': agdd})
        ts_list.append(res_dict)

    ts_df = pd.DataFrame(ts_list)
    ts_df.sort_values(by='agdd', inplace=True)
    ts_df['farm'] = farm
    ts_df['parcel'] = parcel_name
    return ts_df


if __name__ == '__main__':

    import os
    cwd = Path(__file__).parent.absolute()
    os.chdir(cwd)

    # in-situ measurements and inversion results of traits
    data_dir = Path('../../results/lut_based_inversion')
    # data_dir = Path('/mnt/ides/Lukas/04_Work/lut_based_inversion')
    out_dir = Path('../../results/Figures')

    traits = ['lai', 'ccc']
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
            parcel_ts = get_parcel_ts(
                farm, parcel_name, parcel_gdf, img_dir, traits=['lai', 'ccc'])
            # save variety information
            parcel_ts['variety'] = parcel_df['Variety'].values[0]
            data_list.append(parcel_ts)

    df = pd.concat(data_list)
    df.index = [x for x in range(df.shape[0])]
    df['sensing_date'] = pd.to_datetime(df.sensing_date, format='%Y-%m-%d')

    # calculate the leaf chlorophyll content
    df['cab_q50'] = df['ccc_q50'] / df['lai_q50'] * 100

    # save dataframe
    df.to_csv(out_dir.joinpath('ts_dates_agdds.csv'))

    # plot time series of the single parcels into one figure
    f, ax = plt.subplots(figsize=(40, 20), ncols=2, nrows=3)
    ax = ax.flatten()
    traits_to_plot = ['lai', 'ccc', 'cab']
    traits_to_plot_names = ['GLAI', 'CCC', 'Cab']
    units = [r'[$m^2$ $m^{-2}$]', r'[$g$ $m^{-2}$]', r'[$\mu g$ $cm^{-2}$]']
    jj = 0
    for idx in range(3):
        legend = False
        if idx == 2:
            legend = True
        sns.lineplot(
            x='sensing_date',
            y=f'{traits_to_plot[idx]}_q50',
            hue='parcel',
            data=df,
            ax=ax[jj],
            legend=False,
            marker='x')
        sns.lineplot(
            x='agdd',
            y=f'{traits_to_plot[idx]}_q50',
            hue='parcel',
            data=df,
            ax=ax[jj+1],
            marker='x',
            legend=legend)

        if legend:
            ax[jj+1].legend(loc='upper center',
                            bbox_to_anchor=(0., -0.3), fancybox=False,
                            shadow=False, ncol=4)

        ax[jj].set_ylabel(f'{traits_to_plot_names[idx]}' + units[idx])
        ax[jj+1].set_ylabel('')
        ax[jj].set_xlabel('')
        ax[jj+1].set_xlabel('')
        if idx == 0:
            ax[jj].set_title('(a) Calendar Dates')
            ax[jj+1].set_title('(b) Thermal Time')
        if idx == 2:
            ax[jj].set_xlabel('Date (YYYY-MM)')
            ax[jj+1].set_xlabel(r'Accumulated Growing Degree Days [$deg$ $C$]')
        jj += 2

    f.savefig(out_dir.joinpath('ts_dates_agdds.png'), bbox_inches='tight')
    plt.close(f)
