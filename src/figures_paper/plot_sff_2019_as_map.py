"""
Plot the SFF-2019 data (as map) showing GLAI, CCC and Cab
and the experimental design
"""

import geopandas as gpd
import matplotlib.pyplot as plt

from eodal.core.raster import RasterCollection
from matplotlib_scalebar.scalebar import ScaleBar
from pathlib import Path
from typing import List


trait_units = {
    'lai': r'$m^2$ $m^{-2}$',
    'ccc': r'$g$ $m^{-2}$',
    'cab': r'$\mu g$ $cm^{-2}$',
}


def plot_sff_2019_as_map(
        fpath_scene: Path,
        out_dir: Path,
        fpath_design: Path,
        field_parcels: List[str]
) -> None:
    """
    Plot the 2019 data as a map

    Parameters
    ----------
    fpath_scene : Path
        Path to the scene.
    out_dir : Path
        Path to the output directory.
    fpath_design : Path
        Path to the experimental design.
    """
    # get the boundaries and design of the fields
    for field in field_parcels:
        # boundaries from shape
        fpath_boundaries = fpath_design.joinpath(f'{field}.shp')
        boundaries = gpd.read_file(fpath_boundaries).to_crs(epsg=32632)
        # design from shape
        fpath_design_shp = fpath_design.joinpath(f'19_{field}_fert_zone.shp')
        design = gpd.read_file(fpath_design_shp).to_crs(epsg=32632)
        # read the trait file
        fpath_traits = fpath_scene.joinpath(
            'germination-endoftillering_lutinv_traits.tiff')
        ds_complete = RasterCollection.from_multi_band_raster(
            fpath_raster=fpath_traits,
            vector_features=boundaries
        )
        # open a figure for plotting
        f, ax = plt.subplots(ncols=3, nrows=1, figsize=(20, 10))
        for idx, trait in enumerate(trait_units.keys()):
            ds_complete[trait].plot(
                colormap='viridis',
                colorbar_label=trait_units[trait],
                ax=ax[idx]
            )
            ax[idx].set_xlabel('')
            ax[idx].set_xticklabels([])
            ax[idx].set_ylabel('')
            ax[idx].set_yticklabels([])
            # plot the design file on top. Remove the fill and label
            # the treatment zones
            design.plot(
                ax=ax[idx],
                column='crop_type',
                edgecolor='black',
                facecolor='none',
                linewidth=8,
                alpha=0.5
            )




if __name__ == '__main__':

    import os
    cwd = Path(__file__).parent.absolute()
    os.chdir(cwd)

    fpath_scene = Path(
        '../../results/lut_based_inversion/SwissFutureFarm_2019/S2A_MSIL2A_20190420T103031_N0211_R108_T32TMT_20190420T132227.SAFE')  # noqa: E501
    out_dir = Path('../../results/figures_paper')
    parcels = ['F2', 'F3']

    # path to design from Argento et al.
    fpath_design = Path(
        '../../data/auxiliary/field_parcels_ww_2019/SwissFutureFarm')

    plot_sff_2019_as_map(
        fpath_scene=fpath_scene,
        out_dir=out_dir,
        fpath_design=fpath_design,
        field_parcels=parcels
    )
