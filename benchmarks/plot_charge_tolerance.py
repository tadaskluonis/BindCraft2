"""Figure: what a +-2 e charge-tolerance window buys over an exact charge ask.

Reads benchmarks/results/charge_tolerance_summary.csv (written by benchmarks/charge_tolerance.py)
and writes benchmarks/results/charge_tolerance.png.

Panel (a) is the absolute per-residue negative log-likelihood against the charge asked for, with
four series per backbone: each arm decoded under the controller, and each arm's own
rejection-sampling reference drawn from the same unconstrained pool. Panel (b) reduces each
backbone to the mean overhead of the controller over that reference, one marker per arm.
"""
import os
import sys

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
COLOURS = {'1ENH': '#0072B2', '1BRS': '#E69F00', '1AKE': '#009E73'}
BASE, SMALL, TINY = 7.5, 6.5, 6.0


def apply_style() -> None:
    mpl.rcParams.update({
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'font.size': BASE, 'axes.titlesize': BASE, 'axes.labelsize': BASE,
        'legend.fontsize': SMALL, 'xtick.labelsize': TINY, 'ytick.labelsize': TINY,
        'axes.titlelocation': 'left', 'axes.titlepad': 5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'lines.linewidth': 1.0, 'legend.frameon': False,
    })


def interval(spread: float, count: int) -> float:
    """Half-width of the 95% t interval on a mean. Zero when there is nothing to spread."""
    if not np.isfinite(spread) or count is None or count < 2:
        return 0.0
    return float(stats.t.ppf(0.975, count - 1) * spread / count ** 0.5)


def main() -> int:
    import matplotlib.pyplot as plt
    apply_style()
    summary = pd.read_csv(os.path.join(RESULTS, 'charge_tolerance_summary.csv'))
    tolerance = summary.loc[summary.arm == 'window', 'tolerance'].iloc[0]
    order = sorted(summary.backbone.unique(),
                   key=lambda name: summary.loc[summary.backbone == name, 'length'].iloc[0])
    offsets = sorted(summary.offset.unique())

    figure, (likelihood, overhead) = plt.subplots(
        1, 2, figsize=(7.0, 2.9), gridspec_kw={'width_ratios': [1.75, 1.0], 'wspace': 0.42})

    for backbone in order:
        rows = summary[summary.backbone == backbone].sort_values('offset')
        colour = COLOURS[backbone]
        exact = rows[rows.arm == 'exact']
        window = rows[rows.arm == 'window']

        # the two arms as decoded under the controller
        for arm, marker, dashes, size in ((exact, 'o', (), 2.8), (window, '^', (0, (1, 1.6)), 3.0)):
            likelihood.plot(arm.offset, arm.arm_nll, color=colour, lw=1.0,
                            ls='-' if not dashes else dashes, zorder=3)
            likelihood.errorbar(arm.offset, arm.arm_nll,
                                yerr=[interval(s, n) for s, n in zip(arm.arm_sd, arm.n_designs)],
                                color=colour, marker=marker, ms=size, mfc=colour, mew=0, ls='none',
                                elinewidth=0.5, capsize=1.2, zorder=4)

        # each arm's own rejection-sampling reference, plotted only where the pool supplied one
        for arm, marker, dashes, size in ((exact, 'o', (0, (4, 2)), 3.4), (window, '^', (0, (4, 1, 1, 1)), 3.6)):
            reference = arm[arm.n_rejection > 0]
            likelihood.plot(reference.offset, reference.rejection_nll, color=colour, lw=0.9,
                            ls=dashes, alpha=0.9, zorder=2)
            likelihood.errorbar(reference.offset, reference.rejection_nll,
                                yerr=[interval(s, n) for s, n in zip(reference.rejection_sd, reference.n_rejection)],
                                color=colour, marker=marker, ms=size, mfc='white', mec=colour, mew=0.7,
                                ls='none', elinewidth=0.5, capsize=1.2, zorder=4)

        last = exact.iloc[-1]
        likelihood.annotate(f'{backbone} · {int(rows.length.iloc[0])} aa',
                            xy=(last.offset, last.arm_nll), xytext=(4, 0), textcoords='offset points',
                            color=colour, va='center', fontsize=TINY)

    likelihood.set_xlabel('net charge asked for, relative to the model baseline (e)', labelpad=3)
    likelihood.set_ylabel('negative log-likelihood\n(nats per residue)', labelpad=3)
    likelihood.set_title('a  Absolute likelihood', loc='left')
    likelihood.set_xticks(offsets)
    likelihood.set_xlim(offsets[0] - 1.0, offsets[-1] + 9.5)
    likelihood.set_ylim(0.50, 1.34)

    import matplotlib.lines as mlines
    key = [
        (mlines.Line2D([], [], color='#444444', marker='o', ms=2.8, mfc='#444444', mew=0, ls='-', lw=1.0),
         'exact ask, under the controller'),
        (mlines.Line2D([], [], color='#444444', marker='o', ms=3.4, mfc='white', mec='#444444', mew=0.7,
                       ls=(0, (4, 2)), lw=0.9), 'exact ask, rejection sampling'),
        (mlines.Line2D([], [], color='#444444', marker='^', ms=3.0, mfc='#444444', mew=0, ls=(0, (1, 1.6)), lw=1.0),
         f'±{tolerance:g} e window, under the controller'),
        (mlines.Line2D([], [], color='#444444', marker='^', ms=3.6, mfc='white', mec='#444444', mew=0.7,
                       ls=(0, (4, 1, 1, 1)), lw=0.9), f'±{tolerance:g} e window, rejection sampling'),
    ]
    likelihood.legend([handle for handle, _ in key], [label for _, label in key],
                      loc='lower left', bbox_to_anchor=(0.035, 0.005), handlelength=2.2,
                      handletextpad=0.5, labelspacing=0.22, borderaxespad=0.0)

    positions = np.arange(len(order))
    shift = 0.17
    for index, backbone in enumerate(order):
        rows = summary[(summary.backbone == backbone) & (summary.n_rejection > 0)]
        colour = COLOURS[backbone]
        for arm, marker, offset_x, size, transparency in (('exact', 'o', -shift, 4.2, 1.0),
                                                          ('window', '^', shift, 4.6, 0.85)):
            excess = rows.loc[rows.arm == arm, 'excess_nll'].to_numpy(dtype=float)
            overhead.errorbar(positions[index] + offset_x, excess.mean(),
                              yerr=interval(excess.std(ddof=1), excess.size),
                              color=colour, marker=marker, ms=size, mfc=colour, mew=0, ls='none',
                              elinewidth=0.6, capsize=1.8, alpha=transparency, zorder=3)

    overhead.axhline(0, color='#999999', lw=0.5, zorder=1)
    overhead.set_xticks(positions)
    overhead.set_xticklabels([f'{backbone}\n{int(summary.loc[summary.backbone == backbone, "length"].iloc[0])} aa'
                              for backbone in order])
    overhead.set_xlim(-0.55, len(order) - 0.45)
    overhead.set_ylabel('overhead vs rejection sampling\n(nats per residue)', labelpad=3)
    overhead.set_title('b  Controller overhead', loc='left')
    overhead.text(0.98, 0.96, 'lower = better', transform=overhead.transAxes, ha='right', va='top',
                  fontsize=TINY, color='#888888', style='italic')

    destination = os.path.join(RESULTS, 'charge_tolerance.png')
    figure.savefig(destination)
    print(f'wrote {destination}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
