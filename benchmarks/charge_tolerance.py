"""What a charge TOLERANCE window buys, measured against an exact charge ask.

Same machinery as charge_cost.py, one shared unconstrained pool per backbone, and two constrained
arms per target:

* **exact**   target_charge = c, charge_tolerance = 0
* **window**  target_charge = c, charge_tolerance = t  (the design is accepted anywhere in c +- t)

Each arm gets its own rejection-sampling reference drawn from the same pool: designs at exactly c
for the exact arm, designs anywhere in c +- t for the window arm. Rejection sampling realises the
true conditional, so the gap between an arm and its own reference is that arm's controller
overhead, and the gap between the two arms is what relaxing the ask is worth.

    JAX_PLATFORMS=cpu python benchmarks/charge_tolerance.py --pool 128 --n-designs 16 --tolerance 2
    python benchmarks/charge_tolerance.py --pool 2048 --n-designs 128 --tolerance 2

Writes benchmarks/results/charge_tolerance_designs.csv and charge_tolerance_summary.csv.
"""
import argparse
import csv
import os
import statistics
import sys
import zlib

import jax
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bindcraft.model_weights import proteinmpnn_weights
from bindcraft.protein import AMINO_ACIDS
from bindcraft.proteinmpnn import ProteinMPNNSequenceModel

from charge_cost import PANEL, RESULTS, SEED, TEMPERATURE, CHUNK, designed_chain, net_charge, mean, spread


def condition_seed(backbone: str, target: int, arm: str) -> int:
    """One stream per (backbone, target, arm): shared seeds would correlate the arms' noise and
    hide the very difference being measured."""
    return zlib.crc32(f'{SEED}:{backbone}:{target}:{arm}'.encode()) % (2 ** 31)


def decode(protein, count: int, target_charge=None, tolerance: float = 0.0, seed: int = SEED) -> list[dict]:
    model = ProteinMPNNSequenceModel(data_dir=proteinmpnn_weights(), temperature=TEMPERATURE,
                                     key=jax.random.PRNGKey(seed), variant='neutral',
                                     target_charge=target_charge, charge_tolerance=tolerance)
    designs, drawn = [], 0
    while drawn < count:
        batch = min(CHUNK, count - drawn)
        for candidate in model.predict_candidates({'state': {'A': protein}}, candidate_count=batch):
            prediction = candidate['state']
            sequence = ''.join(AMINO_ACIDS[index] for index in
                               np.asarray(prediction.protein_complex['A'].sequence).argmax(-1))
            designs.append({'sequence': sequence, 'charge': net_charge(sequence),
                            'nll': float(prediction.metrics['score'])})
        drawn += batch
    return designs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool', type=int, default=512, help='unconstrained designs per backbone, for the rejection references')
    parser.add_argument('--n-designs', type=int, default=32, help='constrained designs per target and arm')
    parser.add_argument('--tolerance', type=float, default=2.0, help='half-width of the accepted charge window, in e')
    parser.add_argument('--min-bin', type=int, default=10, help='pool designs needed in a reference before it is used')
    parser.add_argument('--offsets', type=int, nargs='*', default=[-12, -8, -6, -4, -2, 0, 2, 4, 6, 8, 12],
                        help="targets, as offsets from the backbone's own unconstrained mean charge")
    arguments = parser.parse_args()
    tolerance = abs(arguments.tolerance)
    os.makedirs(RESULTS, exist_ok=True)
    design_rows, summary_rows = [], []

    for entry, chain, description in PANEL:
        protein = designed_chain(entry, chain)
        length = len(protein)
        pool = decode(protein, arguments.pool)
        pool_charges = [design['charge'] for design in pool]
        pool_nll = mean(d['nll'] for d in pool)
        pool_sd = spread(d['nll'] for d in pool)
        preferred = int(round(mean(pool_charges)))
        print(f'\n=== {entry} {description}  L={length}  pool {len(pool)}  '
              f'charge {min(pool_charges):+d} to {max(pool_charges):+d} (mean {mean(pool_charges):+.2f})  '
              f'NLL {pool_nll:.4f}   tolerance +-{tolerance:g}', flush=True)
        for design in pool:
            design_rows.append({'backbone': entry, 'length': length, 'arm': 'unconstrained',
                                'target': '', 'tolerance': '', 'charge': design['charge'], 'nll': design['nll']})

        for offset in arguments.offsets:
            target = preferred + offset
            for arm, width in (('exact', 0.0), ('window', tolerance)):
                designs = decode(protein, arguments.n_designs, target_charge=target, tolerance=width,
                                 seed=condition_seed(entry, target, arm))
                charges = [design['charge'] for design in designs]
                arm_nll = mean(d['nll'] for d in designs)
                for design in designs:
                    design_rows.append({'backbone': entry, 'length': length, 'arm': arm, 'target': target,
                                        'tolerance': width, 'charge': design['charge'], 'nll': design['nll']})
                reference = [d['nll'] for d in pool if abs(d['charge'] - target) <= width]
                enough = len(reference) >= arguments.min_bin
                row = {'backbone': entry, 'description': description, 'length': length,
                       'preferred_charge': preferred, 'offset': offset, 'target': target, 'arm': arm,
                       'tolerance': width, 'demand_per_residue': abs(offset) / length,
                       'n_designs': len(designs),
                       'in_window': sum(abs(charge - target) <= width for charge in charges),
                       'on_exact_target': sum(charge == target for charge in charges),
                       'charge_sd': spread(charges),
                       'arm_nll': arm_nll, 'arm_sd': spread(d['nll'] for d in designs),
                       'n_pool': len(pool), 'pool_nll': pool_nll, 'pool_sd': pool_sd,
                       'n_rejection': len(reference) if enough else 0,
                       'rejection_nll': mean(reference) if enough else '',
                       'rejection_sd': spread(reference) if enough else '',
                       'excess_nll': (arm_nll - mean(reference)) if enough else '',
                       'cost_vs_pool': arm_nll - pool_nll}
                summary_rows.append(row)
                note = (f'   rejection {row["rejection_nll"]:.4f} (n={row["n_rejection"]})'
                        f'   excess {row["excess_nll"]:+.4f}' if enough else '   rejection n/a')
                print(f'  target {target:+4d} (offset {offset:+3d})  {arm:<6s} +-{width:g}'
                      f'   in window {row["in_window"]:3d}/{len(designs)}   NLL {arm_nll:.4f}'
                      f'   cost {row["cost_vs_pool"]:+.4f}{note}', flush=True)

    for name, rows in (('charge_tolerance_designs.csv', design_rows), ('charge_tolerance_summary.csv', summary_rows)):
        with open(os.path.join(RESULTS, name), 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'wrote {os.path.join(RESULTS, name)} ({len(rows)} rows)', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
