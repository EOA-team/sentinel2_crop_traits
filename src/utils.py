'''
Utility functions

@author Lukas Valentin Graf
'''

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from collections import namedtuple
from datetime import date
from eodal.core.raster import RasterCollection
from matplotlib.axes import Axes
from matplotlib_scalebar.scalebar import ScaleBar
from numbers import Number
from pathlib import Path
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error, f1_score
from typing import Dict, List, Optional, Tuple

TraitLimits = namedtuple('TraitLimits', 'lower upper')
MERGE_TOLERANCE = 20  # Growing Degrees [deg C]

tillering = [x for x in range(0,30)] # horizontal growth
stem_elong =  [x for x in range(31,60)] # vertical growth (until anthesis)
reproductive = [x for x in range(61,100)] # flowering, ripening, senescence

# base temperature of winter wheat [deg C]
TBASE = 0

def _gdd(tmean: float) -> float:
    """
    backend function for GDD calculation

    :param tmean:
        mean air temperature in deg Celsius
    :returns:
        GDD in deg Celsius
    """
    if tmean > TBASE:
        return tmean - TBASE
    else:
        return 0.

def gdd(tmean: pd.Series) -> pd.Series:
    """
    Growing Degree Day (GDD) calculation based on mean
    daily air temperature

    :param tmean:
        mean air temperature in deg Celsius
    :returns:
        GDD in deg Celsius
    """
    return tmean.apply(lambda x, _gdd=_gdd: _gdd(x))

def assign_bbch_stages(x: str) -> str:
    if x == 'germination - end of tillering':
        return 'BBCH 0-29'
    elif x == 'stem elongation - end of heading':
        return 'BBCH 31-59'
    else:
        return 'BBCH 61-99'

def assign_macro_stages(bbch_val: int) -> str:
    if bbch_val in tillering:
        return 'germination - end of tillering'
    elif bbch_val in stem_elong:
        return 'stem elongation - end of heading'
    elif bbch_val in reproductive:
        return 'flowering - fruit development - plant dead'
    else:
        return 'invalid'

def from_agrometeo(fpath: Path) -> pd.DataFrame:
    df = pd.read_csv(fpath, sep=',')
    df['date'] = pd.to_datetime(df['Datum'], format='%d.%m.%Y')
    df.index = df.date
    df.drop(columns=['Datum', 'date'], inplace=True)
    return df

def from_meteoswiss(fpath: Path) -> pd.DataFrame:
    df = pd.read_csv(fpath, sep=';')
    df['date'] = pd.to_datetime(df.time, format='%Y%m%d')
    df.index = df.date
    df.drop(columns=['time', 'stn', 'date'], inplace=True)
    return df

def calc_ww_gdd(temp_df: pd.DataFrame, column_tmean: str) -> pd.DataFrame:
    """
    Calculates Growing Degree Days from daily mean air temperature
    """
    _df = temp_df.copy()
    _df['gdd'] = gdd(temp_df[column_tmean])
    return _df

def cumulative_gdds(temp_df: pd.DataFrame, sowing_date: date, harvest_date: date) -> pd.DataFrame:
    """
    Cumulative sum of growing degree days between sowing and harvest date
    """
    _df = temp_df.copy()
    _df_crop = _df[sowing_date:harvest_date].copy()
    _df_crop['gdd_cumsum'] = _df_crop.gdd.cumsum()
    return _df_crop

def read_site_characteristics(fpath: Path, sheet_name='PhenomEn_Sites_2022_short') -> pd.DataFrame:
    """
    loads site-metadata (field calendar)
    """
    df = pd.read_excel(fpath, engine='odf', sheet_name=sheet_name)
    if 'Sowing Date' in df.columns:
        df['Sowing Date'] = pd.to_datetime(df['Sowing Date'])
    if 'Harvest Date' in df.columns:
        df['Harvest Date'] = pd.to_datetime(df['Harvest Date'])
    return df

def get_farms(data_dir: Path, farms: List[str], year: int) -> Dict[str, gpd.GeoDataFrame]:
    """
    Geometries of parcels for farms where LAI was collected
    """
    # loop over farms combine parcel geometries
    res = {}
    for farm in farms:
        parcels = []
        for fpath_farm_shp in data_dir.rglob(f'{farm}/*.shp'):
            farm_shp = gpd.read_file(fpath_farm_shp)
            parcels.append(farm_shp[['geometry']].copy())
        farm_gdf = pd.concat(parcels)
        # dissolve geometries to get a single (Multi)Polygon per farm
        farm_gdf = farm_gdf.dissolve()
        farm_gdf['farm'] = farm
        res[farm] = farm_gdf
    return res

def join_with_insitu(
    insitu_df: gpd.GeoDataFrame,
    bbch_insitu: gpd.GeoDataFrame,
    inv_res_df: pd.DataFrame,
    traits: List[str]
) -> pd.DataFrame:
    """
    Join in-situ trait measurements and S2 observations based on location
    and temperature sums (GDD). Add BBCH in-situ ratings.

    :param insitu_df:
        DataFrame with in-situ measurements of trait(s)
    :param bbch_insitu:
        in-situ BBCH ratings
    :param inv_res_df:
        inversion results from S2 obtained at the sampling locations
        where in-situ measurements and ratings were carried out
    :param traits:
        name of the traits (DataFrame columns) to join data onto
    :returns:
        joined dataframe
    """
    insitu_df.parcel = insitu_df.parcel.apply(lambda x: 'Parzelle35' if x == 'Parzelle 35' else x)
    joined_res = []
    # loop over locations, parcels and sampling points to join results on
    # thermal time scale
    for location_parcel, inv_res_parcel_point in inv_res_df.groupby(['location', 'parcel', 'point_id']):
        location_name, parcel_name, point_id = location_parcel
        # cast point id to int if possible -> otherwise the merge misses records
        try:
            point_id = int(point_id)
        except ValueError:
            pass
        # get insitu measurements and sort them by cumulative GDDs
        insitu_parcel_point = insitu_df[
            (insitu_df.location == location_name) & \
            (insitu_df.parcel == parcel_name)  & \
            (insitu_df.point_id == point_id)
        ].copy()
        insitu_parcel_point.sort_values(by='gdd_cumsum', inplace=True)
        # sort the PROSAIL inversion results
        inv_res_parcel_point.sort_values(by='gdd_cumsum', inplace=True)

        merged_tmp = pd.merge_asof(
            left=inv_res_parcel_point,
            right=insitu_parcel_point[['date', 'gdd_cumsum'] + traits],
            on='gdd_cumsum',
            direction='nearest',
            tolerance=MERGE_TOLERANCE,
            suffixes=('_model', '_insitu')
        )
        bbch_insitu_parcel_point = bbch_insitu[
            (bbch_insitu.location == location_name) & \
            (bbch_insitu.parcel == parcel_name) & \
            (bbch_insitu.point_id == point_id)
        ].copy()
        # in 2019, we do not consider the BBCH
        if bbch_insitu_parcel_point.empty:
            joined_res.append(merged_tmp)
            continue

        bbch_insitu_parcel_point.sort_values(by='gdd_cumsum', inplace=True)
        merged = pd.merge_asof(
            left=merged_tmp,
            right=bbch_insitu_parcel_point[['gdd_cumsum', 'BBCH Rating']],
            on='gdd_cumsum',
            direction='nearest',
            tolerance=MERGE_TOLERANCE,
            suffixes=('_model', '_insitu')
        )
        joined_res.append(merged)
        
    df = pd.concat(joined_res)
    # drop duplicates resulting from merge process
    df.drop_duplicates(
        subset=['parcel', 'date_insitu', 'gdd_cumsum', 'location', 'point_id'],
        inplace=True,
        keep='first'
    )
    return df

def plot_prediction(
        true: np.ndarray,
        pred: np.ndarray,
        trait_name: str,
        trait_unit: str,
        trait_lims: TraitLimits,
        ax: Optional[Axes] = None,
        pred_unc: Optional[np.ndarray | List[np.ndarray]] = None
) -> Tuple[plt.Figure, Dict[str, Number]]:
    """
    Plots predicted against true trait values.

    :param true:
        true (i.e., measured or observed) trait values
    :param pred:
        predicted trait values
    :param trait_name:
        name of the trait for labeling axis
    :param trait_unit:
        unit of the trait for labeling axis
    :param trait_lims:
        lower and upper bounds of the trait values for scaling axis
    :param pred_unc:
        uncertainties of the prediction (if available). If a single
        numpy array symmetric error bars are drawn. If a list with two
        numpy arrays draws asymmetric error bars.
    :returns:
        resulting `~plt.Figure` and dictionary with error metrics of
        the prediction against true values
    """
    # calculate error statistics
    rmse = mean_squared_error(true, pred, squared=False)
    abs_errors = abs(true - pred)
    nrmse = (rmse / true.mean()) * 100.
    median_error = (true - pred).median()
    nmad = 1.4826 * abs(median_error)
    # linear regression
    linregress_res = linregress(true, pred)
    
    modelled = linregress_res.slope * np.linspace(0,8, num=true.shape[0]) + linregress_res.intercept
    # convert error statistics to pandas DataFrame
    error_stats = {
        'N': true.shape[0],
        'RMSE': rmse,
        'NRMSE': nrmse,
        'NMAD': nmad,
        'R': linregress_res.rvalue,
        'R2': linregress_res.rvalue**2,
        'ABS_ERR_Q05': np.quantile(abs_errors.values, 0.05),
        'ABS_ERR_Q50': np.quantile(abs_errors.values, 0.5),
        'ABS_ERR_Q95': np.quantile(abs_errors.values, 0.95)
    }
    err_stats_str = f'N = {true.shape[0]}\nRMSE = {np.round(rmse,2)} ' + trait_unit + \
        f'\nnRMSE = {np.round(nrmse,2)}' +r'$\%$' + f'\nNMAD = {np.round(nmad,2)} ' + \
        trait_unit + '\n' + r'$R^2$' + f' = {np.round(linregress_res.rvalue**2,2)}'

    # scatter plot
    if ax is None:
        f, ax = plt.subplots()
    else:
        f = ax.get_figure()
    if pred_unc is None:
        sns.scatterplot(x=true, y=pred, ax=ax)
    else:
        ax.errorbar(true, pred, yerr=pred_unc, fmt='o', ecolor='grey', elinewidth=.3)
    ax.set_xlim(trait_lims.lower, trait_lims.upper)
    ax.set_xlabel(f'Reference {trait_name} [{trait_unit}]')
    ax.set_ylim(trait_lims.lower, trait_lims.upper)
    ax.set_ylabel(f'Predicted {trait_name} [{trait_unit}]')
    ax.plot(
        [x for x in range(trait_lims.lower, trait_lims.upper+1)],
        [x for x in range(trait_lims.lower, trait_lims.upper+1)],
        label='1:1 fit',
        linestyle='dashed',
        color='grey'
    )
    # plot linear regression
    ax.plot(
        np.linspace(0,8, num=true.shape[0]),
        modelled,
        linestyle='--',
        label='Linear Regression'
    )
    ax.legend(loc='upper left')
    # add error stats to plot
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    if trait_name == 'Green Leaf Area Index':
        ax.text(5, 0.2, err_stats_str, bbox=props)
    elif trait_name == 'Leaf Chlorophyll a+b Content':
        ax.text(50, 2, err_stats_str, bbox=props)
    elif trait_name == 'Canopy Chlorophyll Content':
        ax.text(2.5, 0.2, err_stats_str, bbox=props)

    return f, error_stats

def bbch_confusion_matrix(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Generate confusion matrix of in-situ rated and S2-derived
    phenological macro-stages. The confusion matrix and f1-scores
    are saved to disk as csv files.
    """
    df['BBCH Rating (Macro-Stages)'] = df['BBCH Rating'].apply(
        lambda x, assign_macro_stages=assign_macro_stages:
            assign_macro_stages(bbch_val=x)   
    )
    df = df[df['BBCH Rating (Macro-Stages)'] != 'invalid'].copy()
    # reindex dataframe to avoid errors in crosstab
    df.index = [x for x in range(df.shape[0])]
    true_stages = df['BBCH Rating (Macro-Stages)'].copy()
    true_stages.name = 'In-Situ BBCH'
    pred_stages = df['Macro-Stage'].copy()
    pred_stages.name = 'Predicted BBCH'
    # weighted average to account for label imbalance
    f1_scores = {
        'f1_scoring_weighted': f1_score(y_true=true_stages, y_pred=pred_stages, average='weighted'),
        'f1_scoring_macro': f1_score(y_true=true_stages, y_pred=pred_stages, average='macro')
    }
    # confusion matrix
    df_crosstab = pd.crosstab(true_stages, pred_stages)
    fname_conf_matrix = out_dir.joinpath('BBCH_estimation_confusion_matrix.csv')
    df_crosstab.to_csv(fname_conf_matrix)
    f1_scores_df = pd.DataFrame([f1_scores])
    fname_f1_scores = out_dir.joinpath('BBCH_estimation_f1-scores.csv')
    f1_scores_df.to_csv(fname_f1_scores)

def plot_trait_maps(
    traits: List[str],
    trait_labels: List[str],
    trait_limits: List[TraitLimits],
    inv_res_ds: RasterCollection,
    parcel_points: gpd.GeoDataFrame,
    parcel_gdf: gpd.GeoDataFrame,
    inv_res_date: date,
    pheno_phase_model: str,
    res_dir_parcel: Path
) -> None:
    """
    """
    f, ax = plt.subplots(figsize=(20,20), ncols=3, nrows=len(traits))
    idx = 0
    for trait, label, trait_limit in zip(traits, trait_labels, trait_limits):
        inv_res_ds.plot_band(
            trait,
            colormap='viridis',
            ax=ax[idx,0],
            colorbar_label=label,
            fontsize=16,
            vmin=trait_limit[0],
            vmax=trait_limit[1]
        )
        ax[idx,0].set_title(f'Inversion Result {trait.upper()}')
        inv_res_ds.plot_band(
            f'{trait}_q05',
            colormap='viridis',
            ax=ax[idx,1],
            colorbar_label=label,
            fontsize=16,
            vmin=trait_limit[0],
            vmax=trait_limit[1]
        )
        ax[idx,1].set_title(f'Q05 {trait.upper()}')
        inv_res_ds.plot_band(
            f'{trait}_q95',
            colormap='viridis',
            ax=ax[idx,2],
            colorbar_label=label,
            fontsize=16,
            vmin=trait_limit[0],
            vmax=trait_limit[1]
        )
        ax[idx,2].set_title(f'Q95 {trait.upper()}')
        # add scalebar to plot
        scalebar = ScaleBar(dx=1, units="m")
        ax[idx,0].add_artist(scalebar)
        x, y, arrow_length = 1.4, 0.3, 0.2
        ax[idx,0].annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                    arrowprops=dict(facecolor='white', width=5, headwidth=15),
                    ha='center', va='center', fontsize=20,
                    xycoords=ax[idx,0].transAxes)

        # plot sampling points and parcel boundaries
        for pdx in range(3):
            if pdx > 0:
                ax[idx,pdx].set_ylabel('')
            if idx < len(traits):
                ax[idx,pdx].set_xlabel('')
            parcel_points.to_crs(inv_res_ds[inv_res_ds.band_names[0]].crs).plot(
                markersize=40,
                ax=ax[idx,pdx],
                color='red'
            )
            parcel_gdf.to_crs(inv_res_ds[inv_res_ds.band_names[0]].crs).boundary.plot(
                ax=ax[idx,pdx],
                color='red'
            )

        idx += 1

    f.suptitle(f'{pheno_phase_model} {inv_res_date.date()}')
    # save figure
    trait_str = trait.replace(' ','-')
    fname_plot = res_dir_parcel.joinpath(
        f'{inv_res_date.date()}_{trait_str}_lutinv_{pheno_phase_model}.png'
    )
    f.savefig(fname_plot, bbox_inches='tight')
    plt.close(f)
