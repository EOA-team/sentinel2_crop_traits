'''
This script recreates the scatterplots of the traits shown in the paper.

@author: Lukas Valentin Graf
'''

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path
from utils import plot_prediction, TraitLimits

plt.style.use('bmh')
mpl.rc('font', size=20)

trait_settings = {
    'lai': {
        'trait_name': 'Green Leaf Area Index',
        'trait_unit': r'$m^2$ $m^{-2}$',
        'trait_lims': TraitLimits(0,8),
    },
    'ccc': {
        'trait_name': 'Canopy Chlorophyll Content',
        'trait_unit': r'$g$ $m^{-2}$',
        'trait_lims': TraitLimits(0,4),
    }
}

if __name__ == '__main__':

    traits = ['lai', 'ccc']

    results_dir = Path('../../results/lut_based_inversion')
    out_dir = Path('../../results/Figures')

    for trait in traits:
        fpath_agdds = results_dir.joinpath(
            'agdds_only'
        ).joinpath(
            f'validation_{trait}'
        ).joinpath(
            f'inv_res_joined_with_insitu_{trait}.csv'
        )
        fpath_agdds_s2 = results_dir.joinpath(
            'agdds_and_s2'
        ).joinpath(
            f'validation_{trait}'
        ).joinpath(
            f'inv_res_joined_with_insitu_{trait}.csv'
        )

        df_agdds = pd.read_csv(fpath_agdds)
        df_agdds.dropna(subset=[trait], inplace=True)
        df_agdds_s2 = pd.read_csv(fpath_agdds_s2)
        df_agdds_s2.dropna(subset=[trait], inplace=True)

        f, ax = plt.subplots(ncols=3, figsize=(30,10), sharey=True)

        # NO-PHENO experiment
        _, errors_no_pheno = plot_prediction(
            true=df_agdds[trait],
            pred=df_agdds[f'{trait}_all'],
            ax=ax[0],
            **trait_settings[trait]
        )
        ax[0].set_title('(a)   NO-PHENO')
        errors_no_pheno['experiment'] = 'NO-PHENO'

        # AGDD-PHENO
        _, errors_agdd_pheno = plot_prediction(
            true=df_agdds[trait],
            pred=df_agdds[f'{trait} (Phenology)'],
            ax=ax[1],
            **trait_settings[trait]
        )
        ax[1].set_title('(b)   AGDD-PHENO')
        errors_agdd_pheno['experiment'] = 'AGDD-PHENO'

        # AGDD-S2-PHENO
        _, errors_agdd_s2_pheno = plot_prediction(
            true=df_agdds_s2[trait],
            pred=df_agdds_s2[f'{trait} (Phenology)'],
            ax=ax[2],
            **trait_settings[trait]
        )
        ax[2].set_title('(c)   AGDD-S2-PHENO')
        errors_agdd_pheno['experiment'] = 'AGDD-S2-PHENO'

        fpath_plt = out_dir.joinpath(f'{trait}_scatter_plot.png')
        f.savefig(fpath_plt, bbox_inches='tight')
        plt.close(f)
