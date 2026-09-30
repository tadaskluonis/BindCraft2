"""Where does the controller's overhead come from: the soft steering or the hard mask?

The budget is enforced by two stages. Turning the steering down to nothing (charge_lambda_max = 0)
leaves the hard reachability mask alone, which still guarantees the target. Sweeping the cap
therefore splits the overhead between them. A tolerance sweep does the complementary thing: it
delays the point at which the mask can bind at all.

Reference for every row is rejection sampling over the SAME acceptance window, taken from the
512-design unconstrained pool already in benchmarks/results/charge_cost_designs.csv.
"""
import csv
import os
import statistics
import sys

import jax
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from bindcraft.model_weights import proteinmpnn_weights
from bindcraft.protein import AMINO_ACIDS, Protein, ResidueFlags
from bindcraft.proteinmpnn import ProteinMPNNSequenceModel

POOL_CSV = os.path.join(HERE, 'results', 'charge_cost_designs.csv')
OUT = os.path.join(HERE, 'results', 'charge_controller_decomposition.csv')
STRUCTURES = os.path.join(HERE, 'structures')
CASES = (('1ENH', 'A', 54, -5, 64), ('1AKE', 'A', 214, -11, 32))
CAPS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
TOLERANCES = (0, 1, 2, 3)
TEMPERATURE = 0.1
SEED = 90210

def net_charge(sequence):
    return sum(map(sequence.count, 'KR')) - sum(map(sequence.count, 'DE'))

def mean(values):
    values = list(values)
    return float(statistics.fmean(values)) if values else float('nan')

def load_pool():
    pool = {}
    with open(POOL_CSV) as handle:
        for row in csv.DictReader(handle):
            if row['condition'] == 'unconstrained':
                pool.setdefault(row['backbone'], []).append((int(row['charge']), float(row['nll'])))
    return pool

def chain(entry, name):
    protein = Protein.from_structure(os.path.join(STRUCTURES, f'{entry}.pdb'), chains=name)[name]
    return protein.replace(flags=protein.flags | int(ResidueFlags.DESIGN))

def decode(protein, count, target, tolerance, cap):
    model = ProteinMPNNSequenceModel(data_dir=proteinmpnn_weights(), temperature=TEMPERATURE,
                                     key=jax.random.PRNGKey(SEED), variant='neutral', target_charge=target,
                                     charge_tolerance=tolerance, charge_lambda_max=cap)
    out = []
    while len(out) < count:
        batch = min(32, count - len(out))
        for candidate in model.predict_candidates({'state': {'A': protein}}, candidate_count=batch):
            prediction = candidate['state']
            sequence = ''.join(AMINO_ACIDS[i] for i in np.asarray(prediction.protein_complex['A'].sequence).argmax(-1))
            out.append((net_charge(sequence), float(prediction.metrics['score']), sequence))
    return out

def main():
    pool = load_pool()
    rows = []
    for entry, name, length, target, count in CASES:
        protein = chain(entry, name)
        pool_nll = mean(nll for _, nll in pool[entry])
        print(f'\n=== {entry} L={length} target {target:+d}  pool NLL {pool_nll:.4f}', flush=True)
        for label, caps, tolerances in (('lambda_max', CAPS, (0,)), ('tolerance', (8.0,), TOLERANCES)):
            for cap in caps:
                for tolerance in tolerances:
                    designs = decode(protein, count, target, tolerance, cap)
                    inside = [d for d in designs if abs(d[0] - target) <= tolerance]
                    reference = [nll for charge, nll in pool[entry] if abs(charge - target) <= tolerance]
                    charged = mean(sum(map(d[2].count, 'DEKR')) for d in designs)
                    row = {'backbone': entry, 'length': length, 'target': target, 'sweep': label,
                           'lambda_max': cap, 'tolerance': tolerance, 'n': len(designs),
                           'in_window': len(inside), 'mean_charge': mean(d[0] for d in designs),
                           'nll': mean(d[1] for d in designs), 'charged_residues': charged,
                           'n_reference': len(reference), 'reference_nll': mean(reference) if reference else '',
                           'excess': (mean(d[1] for d in designs) - mean(reference)) if reference else '',
                           'pool_nll': pool_nll}
                    rows.append(row)
                    print(f'  {label:<10} cap {cap:>4.1f}  tol {tolerance}  in window {len(inside):3d}/{len(designs)}'
                          f'   NLL {row["nll"]:.4f}   ref {row["reference_nll"] if reference else "n/a":.4}'
                          f'   excess {row["excess"]:+.4f}   charged residues {charged:.1f}', flush=True)
    with open(OUT, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f'\nwrote {OUT} ({len(rows)} rows)')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
