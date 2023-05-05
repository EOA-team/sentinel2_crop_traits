"""
Plot the SFF-2019 data (as map) showing GLAI, CCC and Cab
and the experimental design
"""

import matplotlib.pyplot as plt

from eodal.core.raster import RasterCollection
from matplotlib_scalebar.scalebar import ScaleBar
from pathlib import Path
from typing import List


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
    pass


if __name__ == '__main__':

    import os
    cwd = Path(__file__).parent.absolute()
    os.chdir(cwd)

    fpath_scene = Path(
        '../../results/lut_based_inversion/SwissFutureFarm_2019/S2A_MSIL2A_20190420T103031_N0211_R108_T32TMT_20190420T132227.SAFE')  # noqa: E501
    out_dir = Path('../../results/figures_paper')
    parcels = ['F2', 'F3']

    # TODO: add design to data
    fpath_design = Path('')

    plot_sff_2019_as_map(
        fpath_scene=fpath_scene,
        out_dir=out_dir,
        fpath_design=fpath_design,
        field_parcels=parcels
    )
