"""Render the measured packed-versus-solo RL throughput benchmark."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    rows = []
    for key, label in [('baoding', 'Baoding-Allegro'), ('pen', 'Pen-Spinning-Allegro')]:
        packed, solo = data['packed_before'][key], data[f'{key}_solo']
        p, s = packed['wall_s_per_iteration_mean'], solo['wall_s_per_iteration_mean']
        rows.append(dict(task=label, packed_s=p, solo_s=s, speedup=p/s,
                         packed_intervals=packed['n_intervals'], solo_intervals=solo['n_intervals']))
    fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
    for i, row in enumerate(rows):
        ax.bar(i-.18, row['packed_s'], width=.34, color='#cc7a39', label='Two policies/GPU' if i == 0 else None)
        ax.bar(i+.18, row['solo_s'], width=.34, color='#277da8', label='One policy/GPU' if i == 0 else None)
        ax.text(i, row['packed_s']+.07, f"{row['speedup']:.2f}× faster alone", ha='center')
    ax.set(xticks=[0, 1], xticklabels=[r['task'] for r in rows], ylabel='Measured seconds / PPO iteration',
           title='GH200: dedicated GPU speeds up each RL expert')
    ax.set_ylim(0, max(r['packed_s'] for r in rows)*1.25)
    ax.legend()
    ax.grid(axis='y', alpha=.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    with args.output.with_suffix('.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
