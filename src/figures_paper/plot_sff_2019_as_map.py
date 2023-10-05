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
trait_limits = {
    'lai': (0, 2),
    'ccc': (0.5, 2),
    'cab': (40, 70),
}
trait_labels = {
    'lai': 'GLAI',
    'ccc': 'CCC',
    'cab': 'Cab',
}
# map the F names to the actual field names
field_parcel_mapping = {
    'F2': 'Grund',
    'F3': 'Schuerpunt',
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
    # open a figure for plotting
    f, ax = plt.subplots(ncols=len(field_parcels), nrows=3,
                         figsize=(10, 12))
    # get the boundaries and design of the fields
    for idx, field in enumerate(field_parcels):
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
        # calculate cab
        cab = ds_complete['ccc'] / ds_complete['lai'] * 100
        cab.rename('cab')
        ds_complete.add_band(cab)
        # plot the map
        for tdx, trait in enumerate(trait_units.keys()):
            if idx == 1:
                colorbar_label = \
                    f'{trait_labels[trait]} [{trait_units[trait]}]'
            else:
                colorbar_label = None
            ds_complete[trait].plot(
                colormap='viridis',
                colorbar_label=colorbar_label,
                ax=ax[tdx, idx],
                vmin=trait_limits[trait][0],
                vmax=trait_limits[trait][1],
                fontsize=20
            )
            ax[tdx, idx].set_xlabel('')
            ax[tdx, idx].set_xticklabels([])
            ax[tdx, idx].set_ylabel('')
            ax[tdx, idx].set_yticklabels([])
            # set the field parcel name above the first row
            if tdx == 0:
                ax[tdx, idx].set_title(
                    field_parcel_mapping[field],
                    fontsize=20
                )
            else:
                ax[tdx, idx].set_title('')
            # set the trait name on the left and rotate it by 90 degrees
            if idx == 0:
                ax[tdx, idx].set_ylabel(
                    trait_labels[trait],
                    fontsize=20,
                    rotation=90,
                    labelpad=20
                )
            # plot the design file on top. Remove the fill and label
            # the treatment zones
            design.plot(
                ax=ax[tdx, idx],
                facecolor='none',
                edgecolor='black',
                linewidth=2
            )
            # add the treatment labels
            for _, row in design.iterrows():
                ax[tdx, idx].text(
                    row.geometry.centroid.x,
                    row.geometry.centroid.y,
                    row.Treatment,
                    fontsize=14,
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='black',
                    weight='bold'
                )
            # add the scale bar
            scalebar = ScaleBar(
                1,
                location='lower right',
                box_alpha=0,
                font_properties={'size': 16}
            )
            ax[tdx, idx].add_artist(scalebar)

    # save the figure
    plt.tight_layout()
    fpath_out = out_dir.joinpath('SFF_2019_treatments_map.png')
    f.savefig(fpath_out, dpi=300, bbox_inches='tight')


if __name__ == '__main__':

    import os
    cwd = Path(__file__).parent.absolute()
    os.chdir(cwd)

    fpath_scene = Path(
        '../../results/lut_based_inversion/SwissFutureFarm_2019/S2A_MSIL2A_20190420T103031_N0211_R108_T32TMT_20190420T132227.SAFE')  # noqa: E501
    out_dir = Path('../../results/Figures')
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
