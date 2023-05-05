"""
This script plots trait maps of GLAI, CCC and Cab for all study
locations in 2022. It shows one up to two scenes per phenological
macro-stage using the least cloudy scenes.
"""

# TODO: use the temperature information and plot all available scenes

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from eodal.core.raster import RasterCollection
from matplotlib_scalebar.scalebar import ScaleBar
from pathlib import Path

phenological_macro_stages = {
    'germination-endoftillering': [0, 30],
    'stemelongation-endofheading': [31, 60],
    'flowering-fruitdevelopment-plantdead': [61, 100]
}
phenological_macro_stages_labels = {
    'germination-endoftillering': 'GE-ET',
    'stemelongation-endofheading': 'SE-EH',
    'flowering-fruitdevelopment-plantdead': 'FL-PM'
}
trait_ranges = {
    'lai': [0, 8],
    'ccc': [0, 4],
    'cab': [0, 70]
}
trait_labels = {
    'lai': r'GLAI [$m^2$ $m^{-2}$]',
    'ccc': r'CCC [$g$ $m^{-2}$]',
    'cab': r'Cab [$\mu$ $g$ $cm^{-2}$]'
}


def plot_trait_maps(
        fpath_insitu_bbch: Path,
        fpath_lut_based_inversion: Path,
        field_parcels_ww: Path,
        out_dir: Path
) -> None:
    """
    Plot trait maps of GLAI, CCC and Cab for all study locations in 2022.
    It shows one up to two scenes per phenological macro-stage using two
    randomly selected scenes.

    Parameters
    ----------
    fpath_insitu_bbch : Path
        Path to the in-situ BBCH data.
    fpath_lut_based_inversion : Path
        Path to the LUT-based inversion results.
    field_parcels_ww : Path
        Path to the field parcels.
    out_dir : Path
        Path to the output directory.
    """
    # loop over the locations found in the bbch data. Get the available
    # Sentinel-2 scenes for each location from the directory with the
    # LUT-based inversion results.
    gdf_bbch = gpd.read_file(fpath_insitu_bbch)
    for location in gdf_bbch['location'].unique():
        gdf_bbch_location = gdf_bbch.loc[
            gdf_bbch['location'] == location].copy()
        # map the BBCH Rating to the phenoligical macro-stages
        gdf_bbch_location['phenological_macro_stage'] = \
            gdf_bbch_location['BBCH Rating'].map(
                lambda x: [k for k, v in phenological_macro_stages.items()
                           if v[0] <= x <= v[1]][0])

        # loop over the *.SAFE directories available
        scenes = []
        fpath_lut_based_inversion_location = \
            fpath_lut_based_inversion.joinpath(location)
        for scene in fpath_lut_based_inversion_location.glob('*.SAFE'):
            scenes.append({
                'scene': scene.name,
                'sensing_date': scene.name.split('_')[2]
            })

        # convert scenes to a DataFrame
        df_scenes = pd.DataFrame(scenes)
        df_scenes['sensing_date'] = pd.to_datetime(
            df_scenes['sensing_date'], format='%Y%m%dT%H%M%S')
        df_scenes.sort_values(by='sensing_date', inplace=True)

        # loop over the phenological macro-stages and get the path to the
        # correct inversion results
        # read the data by parcel
        for parcel in gdf_bbch_location['parcel'].unique():
            # get the path to the parcel
            fpath_parcel = field_parcels_ww.joinpath(
                location, f'{parcel}.shp')

            fig, ax = plt.subplots(nrows=3, ncols=6, figsize=(10, 10))
            fig_counter = 0
            for phenological_macro_stage in \
                    gdf_bbch_location['phenological_macro_stage'].unique():
                # get only the rows of the current phenological macro-stage
                gdf_bbch_location_pms = gdf_bbch_location.loc[
                    gdf_bbch_location['phenological_macro_stage'] ==
                    phenological_macro_stage].copy()
                # get the minimum and maximum date of the currebnt phenological
                # macro-stage
                min_date = gdf_bbch_location_pms['date'].min()
                max_date = gdf_bbch_location_pms['date'].max()
                # get the scenes that are within the current phenological
                # macro-stage
                df_scenes_pms = df_scenes.loc[
                    (df_scenes['sensing_date'] >= min_date) &
                    (df_scenes['sensing_date'] <= max_date)].copy()
                # construct the file-paths to the corresponding inversion
                # results
                df_scenes_pms['fpath_inversion_results'] = \
                    df_scenes_pms['scene'].map(
                        lambda x: fpath_lut_based_inversion_location.joinpath(
                            x,
                            f'{phenological_macro_stage}_lutinv_traits.tiff'))
                # select two scenes
                if df_scenes_pms.empty:
                    continue
                df_scenes_pms_sel = df_scenes_pms.sample(n=2)
                df_scenes_pms_sel.sort_values(
                    by='sensing_date', inplace=True)

                # read the data for the two scenes
                for _, row in df_scenes_pms_sel.iterrows():
                    ds = RasterCollection.from_multi_band_raster(
                        fpath_raster=row.fpath_inversion_results,
                        vector_features=fpath_parcel
                    )
                    # calculate cab if not available
                    if 'cab' not in ds.band_names:
                        cab = ds['ccc'] / ds['lai'] * 100
                        cab.rename('cab')
                        ds.add_band(cab)
                    # plot GLAI, CCC and Cab
                    for idx, trait in enumerate(['lai', 'ccc', 'cab']):
                        colorbar_label = None
                        if fig_counter == 5:
                            colorbar_label = trait_labels[trait]
                        ds[trait].plot(ax=ax[idx, fig_counter],
                                       colormap='viridis',
                                       vmin=trait_ranges[trait][0],
                                       vmax=trait_ranges[trait][1],
                                       colorbar_label=colorbar_label
                                       )
                        ax[idx, fig_counter].set_xlabel('')
                        ax[idx, fig_counter].set_ylabel('')
                        ax[idx, fig_counter].set_title('')
                        ax[idx, fig_counter].set_xticks([])
                        ax[idx, fig_counter].set_yticks([])
                        scalebar = ScaleBar(dx=1, units="m")
                        ax[idx, fig_counter].add_artist(scalebar)
                        if idx == 0:
                            ax[idx, fig_counter].set_title(
                                f'{phenological_macro_stages_labels[phenological_macro_stage]}' +  # noqa: E501
                                f'\n{row.sensing_date.strftime("%Y%m%d")}')
                        # when fig_counter is 0 set the trait in text box to
                        # the left outside of the plot and rotate it by 90
                        # degrees
                        if fig_counter == 0:
                            ax[idx, fig_counter].text(
                                -0.2, 0.5, trait.upper(),
                                horizontalalignment='center',
                                verticalalignment='center',
                                transform=ax[idx, fig_counter].transAxes,
                                rotation=90,
                                fontsize=12,
                            )
                    fig_counter += 1

            # save the figure
            # fig.tight_layout()
            fig.savefig(
                out_dir.joinpath(f'{location}_{parcel}.png'),
                dpi=300, bbox_inches='tight')
            plt.close(fig)


if __name__ == '__main__':

    import os
    cwd = Path(__file__).parent.absolute()
    os.chdir(cwd)

    fpath_insitu_bbch = Path(
        '../../data/in_situ_traits_2022/in-situ_bbch.gpkg')
    fpath_lut_based_inversion = Path('../../results/lut_based_inversion')
    fpath_field_parcels_22 = Path('../../data/auxiliary/field_parcels_ww_2022')
    out_dir = Path('../../results/Figures/trait_maps_all_locations')
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_trait_maps(
        fpath_insitu_bbch,
        fpath_lut_based_inversion,
        fpath_field_parcels_22,
        out_dir
    )
