'''
This script recreates the Figure showing the temporal evolution of GLAI and
CCC at Parzelle35, Witzwil site at five selected sensing dates.

@author: Lukas Valentin Graf
'''

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eodal.core.raster import RasterCollection
from matplotlib_scalebar.scalebar import ScaleBar
from pathlib import Path
from scipy.stats import linregress

mpl.rc('font', size=18)

# selected dates and stages for visualization
configurations = {
    'Witzwil': {
        '20220310': {
            'AGDD': 768, 'Phase': 'GE-ET', 'fexpr': 'germination-endoftillering'},
        '20220325': {
            'AGDD': 921, 'Phase': 'SE-EH', 'fexpr': 'stemelongation-endofheading'},
        '20220414': {
            'AGDD': 1127, 'Phase': 'SE-EH', 'fexpr':
                'stemelongation-endofheading'},
        '20220519': {
            'AGDD': 1691, 'Phase': 'FL-PM', 'fexpr':
                'flowering-fruitdevelopment-plantdead'},
        '20220618': {
            'AGDD': 2317, 'Phase': 'FL-PM', 'fexpr':
                'flowering-fruitdevelopment-plantdead'}
    },
    'Arenenberg': {
        '20220310':
            {'AGDD': 641, 'Phase': 'GE-ET', 'fexpr': 'germination-endoftillering'},
        '20220325': {
            'AGDD': 766, 'Phase': 'GE-ET', 'fexpr': 'germination-endoftillering'},
        '20220414': {
            'AGDD': 948, 'Phase': 'SE-EH', 'fexpr': 'stemelongation-endofheading'},
        '20220519': {
            'AGDD': 1447, 'Phase': 'SE-EH', 'fexpr': 'stemelongation-endofheading'},
        '20220618': {
            'AGDD': 1998, 'Phase': 'FL-PM', 'fexpr': 'flowering-fruitdevelopment-plantdead'}
    },
    'Strickhof': {
        '20220310': {
            'AGDD': 388, 'Phase': 'GE-ET', 'fexpr': 'germination-endoftillering'},
        '20220325': {
            'AGDD': 516, 'Phase': 'GE-ET', 'fexpr': 'germination-endoftillering'},
        '20220414': {
            'AGDD': 684, 'Phase': 'GE-ET', 'fexpr': 'germination-endoftillering'},
        '20220516': {
            'AGDD': 1092, 'Phase': 'SE-EH', 'fexpr': 'stemelongation-endofheading'},
        '20220618': {
            'AGDD': 1687, 'Phase': 'FL-PM', 'fexpr': 'flowering-fruitdevelopment-plantdead'}
    },
    'SwissFutureFarm': {
        '20220310': {
            'AGDD': 525, 'Phase': 'GE-ET', 'fexpr': 'germination-endoftillering'},
        '20220325': {
            'AGDD': 626, 'Phase': 'GE-ET', 'fexpr': 'germination-endoftillering'},
        '20220414': {
            'AGDD': 781, 'Phase': 'GE-ET', 'fexpr': 'germination-endoftillering'},
        '20220519': {
            'AGDD': 1220, 'Phase': 'SE-EH', 'fexpr': 'stemelongation-endofheading'},
        '20220618': {
            'AGDD': 1734, 'Phase': 'FL-PM', 'fexpr': 'flowering-fruitdevelopment-plantdead'}
    }
}


def plot_trait_maps(
        data_dir: Path,
        out_dir: Path,
        shapes_path: Path
) -> None:
    """
    recreates the Figure showing the temporal evolution of GLAI and
    CCC at Parzelle35, Witzwil site at five selected sensing dates.

    :param data_dir:
        directory with inversion results (i.e., GLAI and CCC images)
    :param out_dir:
        directory where to save figure to (WTZ_Trait_Maps.png)
    :param shapes_path:
        directory with shapefiles of the parcels
    """
    for farm in configurations.keys():
        data_dir_farm = data_dir.joinpath(farm)
        shape_dir_farm = shapes_path.joinpath(farm)
        selected_dates = configurations[farm]
        # loop over dates and plot data
        if farm == 'Witzwil':
            figsize = (38, 28)
        else:
            figsize = (38, 15)
        f, ax = plt.subplots(
            figsize=figsize, ncols=5, nrows=3, sharex=True,
            sharey=True)

        # loop over the parcel shapefiles of the farm
        for parcel_shp in shape_dir_farm.glob('*.shp'):

            r2_results_glai_ccc_list = []
            parcel_name = parcel_shp.stem

            for idx, item in enumerate(selected_dates.items()):
                k, v = item
                scene = next(data_dir_farm.glob(f'S2*{k}*.SAFE'))
                fpath_raster = scene.joinpath(
                    f'{v["fexpr"]}_lutinv_traits.tiff')
                trait_ds = RasterCollection.from_multi_band_raster(
                    fpath_raster=fpath_raster,
                    vector_features=parcel_shp)

                # plot traits
                # GLAI
                if idx < 4:
                    colorbar_label = None
                else:
                    colorbar_label = r'GLAI [$m^2$ $m^{-2}$]'
                try:
                    trait_ds['lai'].plot(
                        vmin=0,
                        vmax=8,
                        colormap='viridis',
                        colorbar_label=colorbar_label,
                        ax=ax[0, idx],
                        fontsize=20
                    )
                except Exception as e:
                    print(e)
                    continue
                ax[0, idx].set_xlabel('')
                ax[0, idx].set_ylabel('')
                ax[0, idx].set_title(f'{v["Phase"]}\n{k} ({v["AGDD"]} deg C)')
                scalebar = ScaleBar(dx=1, units="m")
                ax[0, idx].add_artist(scalebar)

                # when idx is 0 set a text box with "GLAI" left to the
                # first subplot in the row rotated by 90 degrees
                if idx == 0:
                    ax[0, idx].text(
                        -0.1, 0.5, 'GLAI', fontsize=30,
                        transform=ax[0, idx].transAxes,
                        rotation=90, va='center', ha='center')

                # CCC
                if idx == 0:
                    ax[1, idx].set_ylabel('CCC [$g$ $m^{-2}$]')
                if idx < 4:
                    colorbar_label = None
                else:
                    colorbar_label = r'CCC [$g$ $m^{-2}$]'
                trait_ds['ccc'].plot(
                    vmin=0,
                    vmax=4,
                    colormap='viridis',
                    colorbar_label=colorbar_label,
                    ax=ax[1, idx],
                    fontsize=20
                )
                ax[1, idx].set_xlabel('')
                ax[1, idx].set_ylabel('')
                ax[1, idx].set_title('')
                scalebar = ScaleBar(dx=1, units="m")
                ax[1, idx].add_artist(scalebar)

                # similar to GLAI, set a text box with "CCC" left to the
                # first subplot in the row rotated by 90 degrees
                if idx == 0:
                    ax[1, idx].text(
                        -0.1, 0.5, 'CCC', fontsize=30,
                        transform=ax[1, idx].transAxes,
                        rotation=90, va='center', ha='center')

                # CAB
                if idx < 4:
                    colorbar_label = None
                else:
                    colorbar_label = r'Cab [$\mu g$ $cm^{-2}$]'
                trait_ds['cab'].plot(
                    vmin=0,
                    vmax=70,
                    colormap='viridis',
                    colorbar_label=colorbar_label,
                    ax=ax[2, idx],
                    fontsize=20
                )
                ax[2, idx].set_xlabel('')
                ax[2, idx].set_ylabel('')
                ax[2, idx].set_title('')
                scalebar = ScaleBar(dx=1, units="m")
                ax[2, idx].add_artist(scalebar)

                # similar to GLAI, set a text box with "Cab" left to the first
                # subplot in the row rotated by 90 degrees
                if idx == 0:
                    ax[2, idx].text(
                        -0.1, 0.5, 'Cab', fontsize=30,
                        transform=ax[2, idx].transAxes,
                        rotation=90, va='center', ha='center')

                # # GLAI - CCC R2
                glai_vals = trait_ds['lai'].values.flatten()
                if isinstance(glai_vals, np.ma.MaskedArray):
                    glai_vals = glai_vals.data
                glai_vals = glai_vals[~np.isnan(glai_vals)]
                ccc_vals = trait_ds['ccc'].values.flatten()
                if isinstance(ccc_vals, np.ma.MaskedArray):
                    ccc_vals = ccc_vals.data
                ccc_vals = ccc_vals[~np.isnan(ccc_vals)]
                cab_vals = trait_ds['cab'].values.flatten()
                if isinstance(cab_vals, np.ma.MaskedArray):
                    cab_vals = cab_vals.data
                cab_vals = cab_vals[~np.isnan(cab_vals)]

                # save the R2 between GLAI and CCC for each date
                # as well as the number of valid pixels
                # get the linear regression model between GLAI and CCC
                # and save the rscore
                linregress_glai_ccc = linregress(glai_vals, ccc_vals)
                linregress_glai_cab = linregress(glai_vals, cab_vals)

                r2_results_glai_ccc_list.append({
                    'date': k,
                    'r2': linregress_glai_ccc.rvalue**2,
                    'n_pixels': len(glai_vals),
                    'r2_cab': linregress_glai_cab.rvalue**2})

                for ii in range(2):
                    ax[ii, idx].set_xlabel('')
                    ax[ii, idx].set_xticklabels([])
                    ax[ii, idx].set_ylabel('')
                    ax[ii, idx].set_yticklabels([])

            fname_maps = out_dir.joinpath(
                f'{farm}_{parcel_name}_Trait_Maps.png')
            plt.tight_layout()
            f.savefig(fname_maps, bbox_inches='tight')
            plt.close(f)

            r2_results_glai_ccc = pd.DataFrame(r2_results_glai_ccc_list)
            r2_results_glai_ccc.to_csv(
                out_dir.joinpath
                (f'{farm}_{parcel_name}_Trait_Maps_GLAI-CCC_R2.csv'),
                index=False)


if __name__ == '__main__':

    import os
    cwd = Path(__file__).parent.absolute()
    os.chdir(cwd)

    # path to the shapes of the field parcels
    shapes_path = Path('../../data/auxiliary/field_parcels_ww_2022')

    data_dir = Path('../../results/lut_based_inversion')
    out_dir = Path('../../results/Figures')
    out_dir.mkdir(exist_ok=True)

    plot_trait_maps(
        data_dir=data_dir,
        out_dir=out_dir,
        shapes_path=shapes_path
    )
