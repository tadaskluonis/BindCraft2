"""Figure: what the net-charge budget costs in model likelihood.

    python benchmarks/plot_charge_cost.py     # reads results/charge_cost_summary.csv

Panel (a) shows absolute per-residue NLL for designs decoded under the budget (filled markers) and
the true conditional via rejection sampling (unfilled markers), connected by backbone.
The shaded band is the unconstrained pool baseline ± 95% CI. Panel (b) is the controller's overhead,
with error bars showing both sources of noise.
"""
import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
COLOURS = {'1ENH': '#0072B2', '1BRS': '#E69F00', '1AKE': '#009E73'}
SIZES = (8, 7, 6)

def style() -> None:
    base, small, tiny = SIZES
    mpl.rcParams.update({'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
                         'font.size': base, 'axes.titlesize': base, 'axes.labelsize': base,
                         'legend.fontsize': small, 'xtick.labelsize': tiny, 'ytick.labelsize': tiny,
                         'axes.titlelocation': 'left', 'axes.titlepad': 6, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                         'ytick.major.width': 0.6, 'lines.linewidth': 1.2, 'legend.frameon': False})

def half_width(spread, count):
    return stats.t.ppf(0.975, count - 1) * spread / count ** 0.5

def overhead_interval(row):
    variance = (row.constrained_sd ** 2 / row.n_constrained) + (row.rejection_sd ** 2 / row.n_rejection)
    return 1.96 * variance ** 0.5

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', default=os.path.join(HERE, 'results', 'charge_cost_summary.csv'))
    parser.add_argument('--out', default=os.path.normpath(os.path.join(HERE, '..', 'docs', 'source', '_static', 'charge_cost.png')))
    arguments = parser.parse_args()
    style()
    summary = pd.read_csv(arguments.summary)
    order = sorted(summary.backbone.unique(), key=lambda name: summary.loc[summary.backbone == name, 'length'].iloc[0])

    figure, (nll_axes, overhead_axes) = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={'width_ratios': [1.55, 1.0]})
    offsets = sorted(summary.offset.unique())
    span = (offsets[0] - 0.6, offsets[-1] + 0.6)

    for backbone in order:
        rows = summary[summary.backbone == backbone].sort_values('offset')
        colour = COLOURS[backbone]
        pool_nll = rows.pool_nll.iloc[0]; pool_sd = rows.pool_sd.iloc[0]; pool_n = rows.n_pool.iloc[0]
        pool_half = half_width(pool_sd, pool_n)
        nll_axes.fill_between(span, [pool_nll - pool_half] * 2, [pool_nll + pool_half] * 2, 
                              color=colour, alpha=0.18, linewidth=0, zorder=1)
        
        # Constrained designs: filled markers with connecting line
        error = [half_width(sd, n) for sd, n in zip(rows.constrained_sd, rows.n_constrained)]
        nll_axes.plot(rows.offset, rows.constrained_nll, color=colour, linewidth=1.2, zorder=2)
        nll_axes.errorbar(rows.offset, rows.constrained_nll, yerr=error, color=colour, marker='o',
                          markersize=3.2, markerfacecolor=colour, markeredgewidth=0, 
                          elinewidth=0.6, capsize=1.6, linestyle='none', zorder=3)
        
        # Rejection sampling: unfilled markers with connecting line
        reference = rows.dropna(subset=['rejection_nll']).copy()
        if len(reference) > 0:
            reference_error = [half_width(sd, n) for sd, n in zip(reference.rejection_sd, reference.n_rejection)]
            nll_axes.plot(reference.offset, reference.rejection_nll, color=colour, linewidth=1.2, 
                         linestyle='--', alpha=0.6, zorder=2)
            nll_axes.errorbar(reference.offset, reference.rejection_nll, yerr=reference_error,
                              color=colour, marker='o', markersize=4.2, markerfacecolor='white', 
                              markeredgewidth=0.9, linestyle='none', elinewidth=0.6, capsize=1.6, zorder=4)
        
        last = rows.iloc[-1]
        nll_axes.annotate(f'{backbone} ({last.length} aa)', xy=(last.offset, last.constrained_nll),
                          xytext=(3, 0), textcoords='offset points', color=colour, va='center', fontsize=SIZES[1])

    nll_axes.set_xlabel('net charge asked for, relative to the model baseline (e)', labelpad=8)
    nll_axes.set_ylabel('negative log-likelihood (nats per residue)')
    nll_axes.set_title('a  Absolute likelihood: constrained designs and exact conditioning', loc='left')
    nll_axes.set_xticks(offsets)
    nll_axes.set_xlim(span[0] - 0.3, span[1] + 5.2)
    nll_axes.set_ylim(0, None)
    nll_axes.margins(y=0.02)
    nll_axes.plot([], [], color='#666666', marker='o', markersize=3.2, markerfacecolor='#666666',
                  markeredgewidth=0, linestyle='-', linewidth=1.2, label='decoded under the budget')
    nll_axes.plot([], [], color='#666666', marker='o', markersize=4.2, markerfacecolor='white',
                  markeredgewidth=0.9, linestyle='--', linewidth=1.2, label='exact conditioning (rejection sampling)')
    nll_axes.legend(loc='upper center', bbox_to_anchor=(0.40, 1.0), fontsize=SIZES[1], handlelength=1.6, borderaxespad=0.2)

    for position, backbone in enumerate(order):
        rows = summary[summary.backbone == backbone].dropna(subset=['excess_nll']).copy()
        colour = COLOURS[backbone]
        rows['half'] = rows.apply(overhead_interval, axis=1)
        rows['clear'] = rows.excess_nll - rows.half > 0
        shifts = [(index - (len(rows) - 1) / 2) * 0.055 for index in range(len(rows))]
        for shift, (_, row) in zip(shifts, rows.iterrows()):
            overhead_axes.errorbar(position + shift, row.excess_nll, yerr=row.half, color=colour,
                                   marker='o', markersize=3.2, markerfacecolor=colour if row.clear else 'white',
                                   markeredgewidth=0.8, elinewidth=0.6, capsize=1.2, linestyle='none', zorder=3)
        median = rows.excess_nll.median()
        overhead_axes.annotate(f'{median:.03f}', xy=(position + 0.31, median), fontsize=SIZES[1],
                               color=colour, va='center', ha='left')
    
    overhead_axes.axhline(0.0, color='#999999', linewidth=0.6, zorder=1)
    overhead_axes.set_xticks(range(len(order)))
    overhead_axes.set_xticklabels([f'{b}\n{summary.loc[summary.backbone == b, "length"].iloc[0]} aa' for b in order])
    overhead_axes.set_xlim(-0.6, len(order) - 0.1)
    overhead_axes.margins(y=0.12)
    overhead_axes.set_ylabel('overhead (nats per residue)')
    overhead_axes.set_title('b  The controller: excess over exact conditioning', loc='left')
    overhead_axes.annotate('lower = better', xy=(0.98, 0.97), xycoords='axes fraction', fontsize=SIZES[1],
                           color='#666666', ha='right', va='top')

    figure.tight_layout(w_pad=1.8); figure.subplots_adjust(bottom=0.15)
    os.makedirs(os.path.dirname(arguments.out) or '.', exist_ok=True)
    figure.savefig(arguments.out)
    print(f'wrote {arguments.out}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
