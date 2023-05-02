"""
Show how the redistribution of CCC and hence Cab and Car based
on GLAI is implemented in the model, i.e., the LUT generation.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

plt.style.use('bmh')
mpl.rcParams['font.size'] = 20


def show_constraints_in_lut(sample_scene: Path, out_dir: Path) -> None:
    """
    Show how the constraints are implemented in the LUT

    :param sample_scene:
        Path to a sample Sentinel-2 scene
    :param out_dir:
        Path to the output directory where to store the figure
    """
    fpath_no_constraints = \
        sample_scene.joinpath(
            'all_phases_lai-cab-ccc-car_lut_no-constraints.pkl')
    lut_no_constraints = pd.read_pickle(fpath_no_constraints)
    fpath_with_constraints = \
        sample_scene.joinpath('all_phases_lai-cab-ccc-car_lut.pkl')
    lut_with_constraints = pd.read_pickle(fpath_with_constraints)

    # Plot the LUTs
    # plot the correlation of GLAI and CCC without constraints to the left
    # and the correlation of GLAI and CCC with constraints to the right
    # color the plots by density
    fig, axes = \
        plt.subplots(ncols=2, nrows=3, figsize=(30, 20))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()
    lut_no_constraints.plot.scatter(
        x='lai', y='ccc', c='k', alpha=0.1, ax=ax1, cmap='viridis')
    lut_with_constraints.plot.scatter(
        x='lai', y='ccc', c='k', alpha=0.1, ax=ax2)
    ax1.set_title('Without physiological constraints')
    ax2.set_title('With physiological constraints')
    ax1.set_xlabel(r'Green Leaf Area Index [$m^2$ $m^{-2}$]')
    ax2.set_xlabel(r'Green Leaf Area Index [$m^2$ $m^{-2}$]')
    ax1.set_ylabel(r'Canopy Chlorophyll Content [$g$ $m^{-2}$]')
    ax2.set_ylabel(r'Canopy Chlorophyll Content [$g$ $m^{-2}$]')
    # set the limit of the LAI to 0-8
    ax1.set_xlim(0, 8)
    ax2.set_xlim(0, 8)
    # set the limit of the CCC to 0-7
    ax1.set_ylim(0, 7)
    ax2.set_ylim(0, 7)

    # the same with GLAI and Cab
    lut_no_constraints.plot.scatter(
        x='lai', y='cab', c='k', alpha=0.1, ax=ax3)
    lut_with_constraints.plot.scatter(
        x='lai', y='cab', c='k', alpha=0.1, ax=ax4)
    ax3.set_xlabel(r'Green Leaf Area Index [$m^2$ $m^{-2}$]')
    ax4.set_xlabel(r'Green Leaf Area Index [$m^2$ $m^{-2}$]')
    ax3.set_ylabel(r'Leaf Chlorophyll Content [$\mu g$ $cm^{-2}$]')
    ax4.set_ylabel(r'Leaf Chlorophyll Content [$\mu g$ $cm^{-2}$]')
    ax3.set_xlim(0, 8)
    ax3.set_ylim(0, 100)
    ax4.set_xlim(0, 8)
    ax4.set_ylim(0, 100)

    # the same Cab and Car
    lut_no_constraints.plot.scatter(
        x='cab', y='car', c='k', alpha=0.1, ax=ax5)
    lut_with_constraints.plot.scatter(
        x='cab', y='car', c='k', alpha=0.1, ax=ax6)
    ax5.set_xlabel(r'Leaf Chlorophyll Content [$\mu g$ $cm^{-2}$]')
    ax6.set_xlabel(r'Leaf Chlorophyll Content [$\mu g$ $cm^{-2}$]')
    ax5.set_ylabel(r'Leaf Carotenoid Content [$\mu g$ $cm^{-2}$]')
    ax6.set_ylabel(r'Leaf Carotenoid Content [$\mu g$ $cm^{-2}$]')
    ax5.set_xlim(0, 100)
    ax5.set_ylim(0, 30)
    ax6.set_xlim(0, 100)
    ax6.set_ylim(0, 30)

    # add a textbox to each axis numbering them a, b, c and so on
    for i, ax in enumerate(axes.flatten()):
        ax.text(
            0.05, 0.95, chr(97 + i), transform=ax.transAxes,
            fontsize=40, va='top', ha='left')

    plt.tight_layout()
    fig.savefig(out_dir / 'constraints_in_lut.png')
    plt.close(fig)


if __name__ == '__main__':

    sample_scene = Path(
        './results/lut_based_inversion/SwissFutureFarm_2019/S2A_MSIL2A_20190420T103031_N0211_R108_T32TMT_20190420T132227.SAFE')  # noqa: E501
    out_dir = Path('./results/Figures')

    show_constraints_in_lut(sample_scene, out_dir)
