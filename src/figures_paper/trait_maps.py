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
selected_dates = {
    '20220310': {'AGDD': 768, 'Phase': 'GE-ET', 'fexpr': 'germination-endoftillering'},
    '20220325': {'AGDD': 921, 'Phase': 'SE-EH', 'fexpr': 'stemelongation-endofheading'},
    '20220414': {'AGDD': 1127, 'Phase': 'SE-EH', 'fexpr': 'stemelongation-endofheading'},
    '20220519': {'AGDD': 1691, 'Phase': 'FL-PM', 'fexpr': 'flowering-fruitdevelopment-plantdead'},
    '20220618': {'AGDD': 2317, 'Phase': 'FL-PM', 'fexpr': 'flowering-fruitdevelopment-plantdead'}
}

def plot_trait_maps(data_dir: Path, out_dir: Path) -> None:
    """
    recreates the Figure showing the temporal evolution of GLAI and
    CCC at Parzelle35, Witzwil site at five selected sensing dates.

    :param data_dir:
        directory with inversion results (i.e., GLAI and CCC images)
    :param out_dir:
        directory where to save figure to (WTZ_Trait_Maps.png)
    """
    # loop over dates and plot data
    f, ax = plt.subplots(figsize=(38,18), ncols=5, nrows=2, sharex=True, sharey=True)

    r2_results_list = []

    for idx, item in enumerate(selected_dates.items()):
        k, v = item
        scene = next(data_dir.glob(f'S2*{k}*.SAFE'))
        fpath_raster = scene.joinpath(f'{v["fexpr"]}_lutinv_traits.tiff')
        trait_ds = RasterCollection.from_multi_band_raster(fpath_raster)

        if k == '20220519':
            mask = trait_ds['lai'].values == 3.9924788
            trait_ds['lai'].mask(mask=mask, inplace=True)
            trait_ds['ccc'].mask(mask=mask, inplace=True)
        
        # plot traits
        if idx < 4:
            colorbar_label = None
        else:
            colorbar_label = r'GLAI [$m^2$ $m^{-2}$]'
        trait_ds['lai'].plot(
            vmin=0,
            vmax=8,
            colormap='viridis',
            colorbar_label=colorbar_label,
            ax=ax[0,idx],
            fontsize=18
        )
        ax[0,idx].set_xlabel('')
        ax[0,idx].set_title(f'{v["Phase"]}\n{k} ({v["AGDD"]} deg C)')
        scalebar = ScaleBar(dx=1, units="m")
        ax[0,idx].add_artist(scalebar)

        if idx < 4:
            colorbar_label = None
        else:
            colorbar_label = r'CCC [$g$ $m^{-2}$]'
        trait_ds['ccc'].plot(
            vmin=0,
            vmax=4,
            colormap='viridis',
            colorbar_label=colorbar_label,
            ax=ax[1,idx],
            fontsize=18
        )
        ax[1,idx].set_title('')
        scalebar = ScaleBar(dx=1, units="m")
        ax[1,idx].add_artist(scalebar)

        # # GLAI - CCC R2
        glai_vals = trait_ds['lai'].values.flatten()
        if isinstance(glai_vals, np.ma.MaskedArray):
            glai_vals = glai_vals.data
        ccc_vals = trait_ds['ccc'].values.flatten()
        if isinstance(ccc_vals, np.ma.MaskedArray):
            ccc_vals = ccc_vals.data

        linregress_res = linregress(
            glai_vals[~np.isnan(glai_vals)],
            ccc_vals[~np.isnan(ccc_vals)]
        )
        r = linregress_res.rvalue
        r2 = r**2
        r2_results_list.append({'date': k, 'R2': r2, 'N': glai_vals.size})

        for ii in range(2):
            ax[ii,idx].set_xlabel('')
            ax[ii,idx].set_xticklabels([])
            ax[ii,idx].set_ylabel('')
            ax[ii,idx].set_yticklabels([])

    fname_maps = out_dir.joinpath('WTZ_Trait_Maps.png')
    f.savefig(fname_maps, bbox_inches='tight')
    plt.close(f)

    r2_results = pd.DataFrame(r2_results_list)
    r2_results.to_csv(out_dir.joinpath('WTZ_Trait_Maps_GLAI-CCC_R2.csv'), index=False)


if __name__ == '__main__':

    data_dir = Path('../results/lut_based_inversion/Witzwil')
    out_dir = Path('../results/Figures')
    out_dir.mkdir(exist_ok=True)

    plot_trait_maps(data_dir, out_dir)

            