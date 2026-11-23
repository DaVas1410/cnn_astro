#!/usr/bin/env python3
"""
Performance Analysis for SLURM Job Array Dataset Generation

Reads the per-task JSON files from data/raw/perf/ and produces
efficiency plots comparing generation speed across dimensions, k-values,
and parallelization.

Usage:
    python src/plot_performance.py                          # default path
    python src/plot_performance.py /path/to/perf_dir        # custom path
    python src/plot_performance.py --output report.png      # save to file
"""

import sys
import os
import json
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_perf_data(perf_dir):
    """Load all task_*_perf.json files from the perf directory."""
    pattern = os.path.join(perf_dir, 'task*_perf.json')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f'ERROR: No perf JSON files found in {perf_dir}')
        sys.exit(1)

    data = []
    for fp in files:
        with open(fp) as f:
            data.append(json.load(f))
    print(f'Loaded {len(data)} task performance files from {perf_dir}')
    return data


def plot_performance(data, output_file=None):
    """Generate performance analysis plots."""
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('SLURM Job Array — Generation Performance', fontsize=16, fontweight='bold')
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # ─── 1. Total time per task (bar chart) ────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    task_ids = [d['task_id'] for d in data]
    total_times = [d['total_time_minutes'] for d in data]
    labels = [f"k{d['k_min']}/{d['k_max']}" for d in data]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(data)))

    ax1.barh(range(len(data)), total_times, color=colors)
    ax1.set_yticks(range(len(data)))
    ax1.set_yticklabels(labels, fontsize=7)
    ax1.set_xlabel('Time (minutes)')
    ax1.set_title('Total Time per Task')
    ax1.invert_yaxis()

    # ─── 2. Images/second per dimension (grouped bar) ──────────────
    ax2 = fig.add_subplot(gs[0, 1])
    all_dims = sorted(set(
        p['dimension'] for d in data for p in d['per_dimension']
    ))
    dim_speeds = {dim: [] for dim in all_dims}
    dim_labels = {dim: [] for dim in all_dims}

    for d in data:
        for p in d['per_dimension']:
            dim_speeds[p['dimension']].append(p['images_per_second'])
            dim_labels[p['dimension']].append(f"k{d['k_min']}/{d['k_max']}")

    x = np.arange(len(all_dims))
    means = [np.mean(dim_speeds[dim]) for dim in all_dims]
    stds = [np.std(dim_speeds[dim]) for dim in all_dims]
    bars = ax2.bar(x, means, yerr=stds, capsize=4,
                   color=['#2196F3', '#4CAF50', '#FF9800'][:len(all_dims)])
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_dims)
    ax2.set_ylabel('Images / second')
    ax2.set_title('Avg Generation Speed by Dimension')
    for bar, m in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{m:.1f}', ha='center', va='bottom', fontsize=9)

    # ─── 3. Time per dimension (stacked bar) ──────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    dim_colors = {'128x128': '#2196F3', '256x256': '#4CAF50', '512x512': '#FF9800'}
    bottom = np.zeros(len(data))

    for dim in all_dims:
        times = []
        for d in data:
            t = 0
            for p in d['per_dimension']:
                if p['dimension'] == dim:
                    t = p['time_seconds'] / 60
            times.append(t)
        ax3.bar(range(len(data)), times, bottom=bottom,
                label=dim, color=dim_colors.get(dim, '#999'))
        bottom += np.array(times)

    merge_times = [d['merge_time_seconds'] / 60 for d in data]
    ax3.bar(range(len(data)), merge_times, bottom=bottom,
            label='Merge', color='#E91E63', alpha=0.7)

    ax3.set_xticks(range(len(data)))
    ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax3.set_ylabel('Time (minutes)')
    ax3.set_title('Time Breakdown per Task')
    ax3.legend(fontsize=8, loc='upper left')

    # ─── 4. Speed vs dimension size (scatter) ─────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    for dim in all_dims:
        h = int(dim.split('x')[0])
        pixels = h * h
        speeds = dim_speeds[dim]
        ax4.scatter([pixels] * len(speeds), speeds, label=dim, s=30, alpha=0.6)

    ax4.set_xlabel('Pixels per image')
    ax4.set_ylabel('Images / second')
    ax4.set_title('Generation Speed vs Image Size')
    ax4.set_xscale('log')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ─── 5. Speed vs k_min (line plot per dimension) ──────────────
    ax5 = fig.add_subplot(gs[2, 0])
    for dim in all_dims:
        k_mins_dim = []
        speeds_dim = []
        for d in data:
            for p in d['per_dimension']:
                if p['dimension'] == dim:
                    k_mins_dim.append(d['k_min'])
                    speeds_dim.append(p['images_per_second'])
        if k_mins_dim:
            # Average speeds for same k_min
            unique_k = sorted(set(k_mins_dim))
            avg_speeds = [np.mean([s for k, s in zip(k_mins_dim, speeds_dim) if k == uk])
                          for uk in unique_k]
            ax5.plot(unique_k, avg_speeds, 'o-', label=dim, markersize=5)

    ax5.set_xlabel('k_min')
    ax5.set_ylabel('Avg images / second')
    ax5.set_title('Speed vs k_min by Dimension')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_xscale('log', base=2)

    # ─── 6. File sizes and total dataset summary ──────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    file_sizes = [d['merged_file_size_mb'] for d in data]
    ax6.bar(range(len(data)), file_sizes, color=colors)
    ax6.set_xticks(range(len(data)))
    ax6.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax6.set_ylabel('File size (MB)')
    ax6.set_title('Merged HDF5 File Sizes')

    total_size_gb = sum(file_sizes) / 1024
    total_images = sum(d['total_images'] for d in data)
    total_time = sum(d['total_time_minutes'] for d in data)
    ax6.text(0.98, 0.95,
             f'Total: {total_images:,} images\n'
             f'{total_size_gb:.2f} GB\n'
             f'{total_time:.1f} CPU-min',
             transform=ax6.transAxes, ha='right', va='top',
             fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # ─── Save or show ─────────────────────────────────────────────
    if output_file:
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f'Saved: {output_file}')
    else:
        plt.show()

    plt.close(fig)

    # ─── Print summary table ──────────────────────────────────────
    print('\n' + '=' * 90)
    print(f'{"Task":>5} {"k_min":>6} {"k_max":>6} {"Dims":>6} {"Images":>8} '
          f'{"Time(min)":>10} {"img/s":>8} {"Size(MB)":>10}')
    print('-' * 90)
    for d in data:
        print(f'{d["task_id"]:>5} {d["k_min"]:>6} {d["k_max"]:>6} '
              f'{len(d["dimensions_generated"]):>6} {d["total_images"]:>8} '
              f'{d["total_time_minutes"]:>10.1f} {d["overall_images_per_second"]:>8.1f} '
              f'{d["merged_file_size_mb"]:>10.1f}')
    print('-' * 90)
    print(f'{"TOTAL":>5} {"":>6} {"":>6} {"":>6} {total_images:>8} '
          f'{total_time:>10.1f} {"":>8} {sum(file_sizes):>10.1f}')
    print('=' * 90)


def main():
    # Default to data/raw/perf relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_perf_dir = os.path.join(project_root, 'data', 'raw', 'perf')
    
    parser = argparse.ArgumentParser(description='Plot SLURM generation performance')
    parser.add_argument('perf_dir', nargs='?',
                        default=default_perf_dir,
                        help='Directory containing task*_perf.json files')
    parser.add_argument('--output', '-o', default=None,
                        help='Save plot to file (e.g., performance.png)')
    args = parser.parse_args()

    data = load_perf_data(args.perf_dir)
    plot_performance(data, args.output)


if __name__ == '__main__':
    main()
