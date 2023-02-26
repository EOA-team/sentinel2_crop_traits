'''
Develop a Green Leaf Area Index (GLAI) - Canopy Chlorophyll Content (CCC) constraint
for redistributing Chlorophyll a+b values in Lookup Tables (LUT) following a similar
approach as Wocher et al. (2020) did for establishing a relationship between leaf
chlorophyll and leaf carotinoid content based on empirical evidence.

@author: Lukas Valentin Graf
'''

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import linregress

mpl.rc('font', size=16)
plt.style.use('bmh')

def linear_regression(x, a, b):
    return a * x + b

def polynomial(x, a, b, c):
    return a * x**2 + b * x + c

def empirircal_relationship(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Establish an empirical relationship between GLAI and CCC based
    on multi-year phenotyping data.

    :param df:
        DataFrame with empirical GLAI and CCC values and phenology data
        expressed as BBCH stages
    :param out_dir:
        directory where to save results to
    """
    # fit regression model to data (drop missing values before)
    # split the fitting into vegetative and reproductive growth
    # as suggested by Gitelson et al. (2022)
    df['growth_class'] = df['phenology [BBCH]'].apply(
        lambda x: 'before flowering' if x < 60 else 'after flowering'
    )
    df.dropna(inplace=True)
    glai_linspace = np.linspace(df.greenLAI.min(), df.greenLAI.max(), 100)

    # fit a linear model to the vegetative data points
    df_veg = df[df.growth_class == 'before flowering'].copy()
    popt_veg, pcov_veg = curve_fit(
        linear_regression,
        xdata=df_veg['greenLAI'],
        ydata=df_veg['ccc'],
        bounds=[(-np.inf,0), (np.inf,np.inf)]
    )
    regr_line_veg = linear_regression(glai_linspace, *popt_veg)

    # reproductive growth stage
    df_rep = df[df.growth_class == 'after flowering'].copy()
    popt_rep, _ = curve_fit(
        polynomial,
        xdata=df_rep['greenLAI'],
        ydata=df_rep['ccc']
    )
    regr_line_rep = polynomial(glai_linspace, *popt_rep)

    # all growth stages
    popt, pcov = curve_fit(
        linear_regression,
        xdata=df['greenLAI'],
        ydata=df['ccc'],
        bounds=[(-np.inf,0), (np.inf,np.inf)]
    )
    regr_line_all = linear_regression(glai_linspace, *popt)
    a_all, b_all = popt
    regr_expr_dict = {'constraint': 'linear_regression', 'expression': f'{b_all} + {a_all} * glai'}
    r_value = linregress(df['greenLAI'], df['ccc']).rvalue
    stats_df = pd.DataFrame([{'N': df.shape[0], 'r_value': r_value, 'R2': r_value**2}])
    stats_df.to_csv(out_dir.joinpath('empirical_relationship_gcc-glai_r2.csv'), index=False)

    # upper bound determined from vegetative model using standard deviation
    # of linearly derived CCC values as suggested by Wocher et al. (2020)
    sd_ccc_veg = np.std(regr_line_veg)
    x_veg = sd_ccc_veg * glai_linspace
    a, b = popt_veg
    ub = a * x_veg + 2*b
    ub_expr = f'{a} * {sd_ccc_veg} * glai + 2*{b}'
    ub_expr_dict = {'constraint': 'upper', 'expression': ub_expr}

    # lower bound determined from reproductive model
    a, b, c = popt_rep
    sd_ccc_rep = np.std(regr_line_rep)
    x_rep = glai_linspace / sd_ccc_rep
    lb = 2*a * x_rep**2 + b * x_rep - c
    lb_expr = f'2*{a} * (glai / {sd_ccc_rep})**2 + {b} *  glai / {sd_ccc_rep} - {c}'
    lb_expr_dict = {'constraint': 'lower', 'expression': lb_expr}

    # plot data
    f, ax = plt.subplots(figsize=(10,10))
    sns.scatterplot(x='greenLAI', y='ccc', hue='growth_class', style='growth_class', data=df, ax=ax,
                    palette=['darkblue', 'orange'], hue_order=['before flowering', 'after flowering'])
    ax.set_ylabel('Empirical Canopy Chlorophyll Content [$g$ $m^{-2}$]', fontsize=16)
    ax.set_xlabel('Empirical Green Leaf Area Index [$m^2$ $m^{-2}$]', fontsize=16)
    sns.lineplot(x=glai_linspace, y=regr_line_all, ax=ax,
                 label=f'Empirical Regression',
                 color='grey', linestyle='dashed', linewidth=2
    )
    sns.lineplot(x=glai_linspace, y=ub, ax=ax, label=f'Upper Envelope',
                 color='grey', linestyle='dotted', linewidth=2)
    sns.lineplot(x=glai_linspace, y=lb, ax=ax, label=f'Lower Envelope',
                 color='grey', linestyle='dashdot', linewidth=2)
    # ax.set_title(f'Empirical; N={df.shape[0]}', size=16)
    ax.set_ylim(0,6)
    ax.set_xlim(0,df.greenLAI.max())
    ax.tick_params(axis='both', labelsize=16)

    fname = out_dir.joinpath('empirical_relationship_gcc-glai.png')
    f.savefig(fname, bbox_inches='tight')
    plt.close(f)

    # save expression of lower and upper constraints and the linear regression to csv
    reg_df = pd.DataFrame(
        data=[lb_expr_dict, ub_expr_dict, regr_expr_dict]
    )
    fname = out_dir.joinpath('empirical_relationship_gcc-glai.csv')
    reg_df.to_csv(fname, index=False)

if __name__ == '__main__':

    # TODO: update paths, add data and test code
    data_dir = Path('../../data/ccc_glai_relationship')
    out_dir = Path('../../results/Figures')
    fpath_data = data_dir.joinpath('cereals_ccc_glai.csv')
    df = pd.read_csv(fpath_data)

    empirircal_relationship(df, out_dir)
