#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_odom_phases.py
===================
Standalone analysis plotter — NO ROS required.

Loads the 3 CSV files produced by odom_phase_recorder.py and renders
a premium 3-panel comparison figure:

  Panel 1  —  X-Y trajectory (top-down path, all 3 phases overlaid)
  Panel 2  —  Cumulative path length vs elapsed time (speed proxy)
  Panel 3  —  Lateral deviation from teach path (error signal)

Usage
-----
  python plot_odom_phases.py --dir ~/jetracer/analysis
  python plot_odom_phases.py --teach teach.csv --blind blind.csv --visual visual.csv
  python plot_odom_phases.py --dir ~/jetracer/analysis --out comparison.png --no-show

Python 2/3 compatible.
"""

from __future__ import division, print_function

import argparse
import csv
import math
import os
import sys


# ── Plotting — deferred import so headless import is possible ────────────────
import matplotlib
matplotlib.use('Agg')   # must be set BEFORE importing pyplot; Agg works headless

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np


# ── Colour palette ───────────────────────────────────────────────────────────
COLOUR_TEACH  = 'blue'      # the reference path
COLOUR_VISUAL = 'green'     # visual correction active (should track)

ALPHA_PATH    = 0.90
ALPHA_FILL    = 0.15

PHASE_LABELS = {
    'teach':  'Teach',
    'visual': 'Repeat - camera active',
}

PHASE_COLOURS = {
    'teach':  COLOUR_TEACH,
    'visual': COLOUR_VISUAL,
}


# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_csv(path):
    """
    Load a CSV written by odom_phase_recorder and return a dict of
    equal-length numpy float64 arrays:
      {'elapsed_s': ..., 'x': ..., 'y': ..., 'yaw': ..., 'speed': ...}

    'speed' is computed from finite differences of (x,y,elapsed_s) so
    it works whether or not a 'vx' column is present in the CSV
    (PoseWithCovarianceStamped has no twist field).
    """
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        return None

    data = {'elapsed_s': [], 'x': [], 'y': [], 'yaw': [], 'ncc_score': []}
    try:
        with open(path, 'r') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    data['elapsed_s'].append(float(row.get('elapsed_s', 0)))
                    data['x'].append(float(row['x']))
                    data['y'].append(float(row['y']))
                    data['yaw'].append(float(row.get('yaw', 0)))
                    data['ncc_score'].append(float(row.get('ncc_score', float('nan'))))
                except (KeyError, ValueError):
                    pass
    except Exception as e:
        print('WARNING: Could not load %s - %s' % (path, e))
        return None

    if not data['x']:
        return None

    # Convert to numpy for vectorised ops
    out = {k: np.asarray(v, dtype=np.float64) for k, v in data.items()}

    # Compute instantaneous speed from position finite differences [m/s]
    dx  = np.diff(out['x'])
    dy  = np.diff(out['y'])
    dt  = np.diff(out['elapsed_s'])
    dt[dt < 1e-6] = 1e-6          # guard against zero dt
    spd = np.sqrt(dx**2 + dy**2) / dt
    # Prepend first sample so length matches
    out['speed'] = np.concatenate([[spd[0] if len(spd) else 0.0], spd])

    # ── Normalise to origin ───────────────────────────────────────────────────
    # Translate and rotate so the FIRST sample is at (x=0, y=0, yaw=0).
    # This makes all three phases directly comparable on the same plot
    # regardless of the physical start position/heading of each run.
    x0   = float(out['x'][0])
    y0   = float(out['y'][0])
    yaw0 = float(out['yaw'][0])
    cos_y = math.cos(-yaw0)
    sin_y = math.sin(-yaw0)
    dx_c  = out['x'] - x0
    dy_c  = out['y'] - y0
    out['x']   =  dx_c * cos_y - dy_c * sin_y
    out['y']   =  dx_c * sin_y + dy_c * cos_y
    out['yaw'] = (out['yaw'] - yaw0 + math.pi) % (2.0 * math.pi) - math.pi

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cumulative_distance(x_arr, y_arr):
    """Cumulative arc-length along a path (numpy arrays)."""
    dx = np.diff(x_arr)
    dy = np.diff(y_arr)
    seg = np.sqrt(dx ** 2 + dy ** 2)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _point_to_segment_dist(px, py, ax, ay, bx, by):
    """
    Perpendicular distance from point P to line segment AB.
    Returns the closest distance (not signed).
    """
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    ab2 = abx * abx + aby * aby
    if ab2 < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = (apx * abx + apy * aby) / ab2
    t = max(0.0, min(1.0, t))
    projx = ax + t * abx
    projy = ay + t * aby
    return math.hypot(px - projx, py - projy)


def _lateral_deviations(ref_x, ref_y, qx, qy):
    """
    For every point in query path (qx, qy), find the minimum perpendicular
    distance to the reference path (ref_x, ref_y).

    Returns np.ndarray of distances (metres).
    """
    deviations = np.empty(len(qx), dtype=np.float64)
    ref_x = np.asarray(ref_x)
    ref_y = np.asarray(ref_y)

    for i in range(len(qx)):
        px, py = float(qx[i]), float(qy[i])
        best = 1e18
        for j in range(len(ref_x) - 1):
            d = _point_to_segment_dist(
                px, py,
                float(ref_x[j]),   float(ref_y[j]),
                float(ref_x[j+1]), float(ref_y[j+1]))
            if d < best:
                best = d
        deviations[i] = best

    return deviations


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

_MARKER_START = dict(marker='o', markersize=10, markeredgewidth=2,
                     markeredgecolor='black', zorder=6)
_MARKER_END   = dict(marker='s', markersize=10, markeredgewidth=2,
                     markeredgecolor='black', zorder=6)


def _draw_path(ax, data, colour, label, lw=2.2):
    """Draw X-Y path with start (circle) and end (square) markers."""
    x, y = data['x'], data['y']
    ax.plot(x, y, color=colour, linewidth=lw, alpha=ALPHA_PATH,
            label=label, zorder=3)
    # Start marker
    ax.plot(x[0], y[0], color=colour, **_MARKER_START)
    # End marker
    ax.plot(x[-1], y[-1], color=colour, **_MARKER_END)


def _style_ax(ax, xlabel='', ylabel='', title=''):
    """Apply standard styling to an axes."""
    ax.set_facecolor('white')
    ax.tick_params(labelsize=9)
    ax.grid(True, linewidth=0.6, linestyle='--', alpha=0.7)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    if title:
        ax.set_title(title, fontsize=11, fontweight='bold')


# ─────────────────────────────────────────────────────────────────────────────
# Main plot function
# ─────────────────────────────────────────────────────────────────────────────

def make_plot(phases, out_png=None, show=True):
    """
    Parameters
    ----------
    phases : dict  {name: data_dict}  name ∈ {teach, blind, visual}
    out_png : str or None  — save path; None = don't save
    show    : bool         — display interactive window
    """
    has_teach  = 'teach'  in phases and phases['teach']  is not None
    has_visual = 'visual' in phases and phases['visual'] is not None

    available = [k for k in ('teach', 'visual') if phases.get(k) is not None]
    if not available:
        sys.exit('ERROR: No valid CSV data to plot.')

    # ── Figure layout ─────────────────────────────────────────────────────────
    # Use matplotlib.gridspec.GridSpec — available since matplotlib 1.x.
    # (fig.add_gridspec was only added in 3.1 and is absent on Jetson Nano.)
    fig = plt.figure(figsize=(14, 16), facecolor='white')
    gs  = gridspec.GridSpec(
        4, 2,
        height_ratios=[3, 1.4, 1.4, 1.4],
        hspace=0.40,
        wspace=0.28,
        left=0.08, right=0.97, top=0.93, bottom=0.06,
    )

    ax_xy   = fig.add_subplot(gs[0, :])   # top-full: X-Y trajectory
    ax_dist = fig.add_subplot(gs[1, 0])   # bottom-left: cumulative distance
    ax_spd  = fig.add_subplot(gs[1, 1])   # bottom-right: speed profile
    ax_dev  = fig.add_subplot(gs[2, :])   # bottom-full: lateral deviation
    ax_ncc  = fig.add_subplot(gs[3, :])   # very bottom-full: NCC score

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.suptitle(
        'JetRacer Odometry - 2-Phase Comparison',
        color='black', fontsize=16, fontweight='bold', y=0.97)

    # ═════════════════════════════════════════════════════════════════════════
    # Panel 1: X-Y Trajectory
    # ═════════════════════════════════════════════════════════════════════════
    _style_ax(ax_xy, xlabel='X  (m)', ylabel='Y  (m)',
              title='Top-Down Path  (circle=start, square=end)')

    if has_teach:
        d = phases['teach']
        _draw_path(ax_xy, d, COLOUR_TEACH, PHASE_LABELS['teach'], lw=2.0)

    if has_visual:
        d = phases['visual']
        _draw_path(ax_xy, d, COLOUR_VISUAL, PHASE_LABELS['visual'], lw=2.0)

    ax_xy.set_aspect('equal', adjustable='datalim')

    # Build combined legend with path lines + start/end markers.
    # Avoid labelcolor= — added in matplotlib 3.2, absent on Jetson.
    legend_markers = [
        Line2D([0], [0], marker='o', color='black', label='Start',
               markersize=8, linestyle='None',
               markeredgewidth=1.5, markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='black', label='End',
               markersize=8, linestyle='None',
               markeredgewidth=1.5, markeredgecolor='black'),
    ]
    handles, labels = ax_xy.get_legend_handles_labels()
    ax_xy.legend(
        handles=handles + legend_markers,
        labels=labels + ['Start', 'End'],
        loc='upper left', framealpha=0.8, fontsize=9,
    )

    # ═════════════════════════════════════════════════════════════════════════
    # Panel 2: Cumulative distance vs elapsed time
    # ═════════════════════════════════════════════════════════════════════════
    _style_ax(ax_dist, xlabel='Elapsed time  (s)',
              ylabel='Distance travelled  (m)',
              title='Cumulative Distance vs Time')

    for name in available:
        d = phases[name]
        cum_d = _cumulative_distance(d['x'], d['y'])
        ax_dist.plot(d['elapsed_s'], cum_d,
                     color=PHASE_COLOURS[name],
                     linewidth=1.8,
                     label=PHASE_LABELS[name])

    ax_dist.legend(loc='upper left', framealpha=0.8, fontsize=8)

    # ═════════════════════════════════════════════════════════════════════════
    # Panel 3: Forward speed profile
    # ═════════════════════════════════════════════════════════════════════════
    _style_ax(ax_spd, xlabel='Elapsed time  (s)',
              ylabel='Forward speed  (m/s)',
              title='Speed Profile')

    for name in available:
        d = phases[name]
        # speed computed from |Δpos|/Δt in _load_csv (no twist field available)
        spd = d['speed']
        ax_spd.plot(d['elapsed_s'], spd,
                    color=PHASE_COLOURS[name],
                    linewidth=1.4,
                    alpha=0.85,
                    label=PHASE_LABELS[name])

    ax_spd.legend(loc='upper left', framealpha=0.8, fontsize=8)
    ax_spd.set_ylim(bottom=0.0)

    # ═════════════════════════════════════════════════════════════════════════
    # Panel 4: Lateral deviation from teach path
    # ═════════════════════════════════════════════════════════════════════════
    _style_ax(ax_dev, xlabel='Cumulative distance along own path  (m)',
              ylabel='Deviation from teach path  (m)',
              title='Lateral Deviation from Teach Path')

    ax_dev.axhline(0, color=COLOUR_TEACH, linewidth=0.8,
                   linestyle='--', alpha=0.6)

    if has_teach:
        teach = phases['teach']
        ref_x, ref_y = teach['x'], teach['y']

        for name in ('visual',):
            if not phases.get(name):
                continue
            d = phases[name]
            devs  = _lateral_deviations(ref_x, ref_y, d['x'], d['y'])
            cum_d = _cumulative_distance(d['x'], d['y'])
            c     = PHASE_COLOURS[name]
            ax_dev.plot(cum_d, devs,
                        color=c,
                        linewidth=1.6,
                        alpha=0.85,
                        label=PHASE_LABELS[name])
            ax_dev.fill_between(cum_d, devs, 0,
                                color=c, alpha=ALPHA_FILL)

            # Annotate RMS error
            rms = float(np.sqrt(np.mean(devs ** 2)))
            max_d = float(np.max(devs))
            ax_dev.annotate(
                '%s  RMS=%.3f m  max=%.3f m' % (PHASE_LABELS[name], rms, max_d),
                xy=(cum_d[-1] * 0.5, rms),
                color=c, fontsize=8, style='italic',
            )

        ax_dev.legend(loc='upper left', framealpha=0.8, fontsize=8)
        ax_dev.set_ylim(bottom=0.0)
    else:
        ax_dev.text(0.5, 0.5,
                    'Teach CSV required to compute deviation',
                    ha='center', va='center', transform=ax_dev.transAxes,
                    color='black', fontsize=11)

    # ═════════════════════════════════════════════════════════════════════════
    # Panel 5: NCC Score
    # ═════════════════════════════════════════════════════════════════════════
    _style_ax(ax_ncc, xlabel='Cumulative distance along own path  (m)',
              ylabel='NCC Score',
              title='Normalized Cross-Correlation Score')

    ax_ncc.axhline(0.02, color='red', linewidth=0.8, linestyle='--', alpha=0.6, label='Threshold (0.02)')

    for name in ('teach', 'visual'):
        if not phases.get(name):
            continue
        d = phases[name]
        if 'ncc_score' in d and not np.all(np.isnan(d['ncc_score'])):
            cum_d = _cumulative_distance(d['x'], d['y'])
            c     = PHASE_COLOURS[name]
            ax_ncc.plot(cum_d, d['ncc_score'],
                        color=c,
                        linewidth=1.6,
                        alpha=0.85,
                        label=PHASE_LABELS[name])
            
    ax_ncc.legend(loc='upper right', framealpha=0.8, fontsize=8)
    ax_ncc.set_ylim(-0.1, 1.05)

    # ── Watermark summarising files ───────────────────────────────────────────
    label_parts = []
    for name in ('teach', 'visual'):
        if phases.get(name) is not None:
            n = len(phases[name]['x'])
            label_parts.append('%s (%d pts)' % (PHASE_LABELS[name], n))
    fig.text(0.98, 0.005,
             '  |  '.join(label_parts),
             ha='right', va='bottom', color='black', fontsize=7)

    # ── Save / show ───────────────────────────────────────────────────────────
    if out_png:
        out_png = os.path.expanduser(out_png)
        fig.savefig(out_png, dpi=180, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print('Saved: %s' % out_png)

    if show:
        # Backend cannot be switched after pyplot is imported — Agg is used.
        # On Jetson (headless SSH), always pass --no-show and view the PNG instead.
        try:
            plt.show()
        except Exception:
            print('NOTE: Cannot display window (headless?). Use --no-show and open the PNG.')
    plt.close(fig)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser():
    p = argparse.ArgumentParser(
        description='Plot JetRacer odometry across 3 phases.'
    )
    p.add_argument(
        '--dir', default=None,
        help='Directory containing teach.csv, visual.csv. '
             'Takes precedence over individual --teach/--visual flags.',
    )
    p.add_argument('--teach',  default=None, help='Path to teach CSV.')
    p.add_argument('--visual', default=None, help='Path to visual-repeat CSV.')
    p.add_argument(
        '--out', default=None,
        help='Output PNG path (default: <dir>/odom_comparison.png or ./odom_comparison.png).',
    )
    p.add_argument(
        '--no-show', action='store_true',
        help='Do not display an interactive window (save only).',
    )
    return p


if __name__ == '__main__':
    args = _build_parser().parse_args()

    # Resolve CSV paths
    teach_csv  = args.teach
    visual_csv = args.visual
    out_png    = args.out

    if args.dir:
        d = os.path.expanduser(args.dir)
        if not teach_csv:
            teach_csv  = os.path.join(d, 'teach.csv')
        if not visual_csv:
            visual_csv = os.path.join(d, 'visual.csv')
        if not out_png:
            out_png = os.path.join(d, 'odom_comparison.png')

    if not out_png:
        out_png = os.path.join(os.getcwd(), 'odom_comparison.png')

    # Load
    phases = {
        'teach':  _load_csv(teach_csv)  if teach_csv  else None,
        'visual': _load_csv(visual_csv) if visual_csv else None,
    }

    found = [k for k, v in phases.items() if v is not None]
    if not found:
        sys.exit('ERROR: No CSV files found. Check paths:\n'
                 '  teach  = %s\n  visual = %s'
                 % (teach_csv, visual_csv))

    missing = [k for k, v in phases.items() if v is None]
    if missing:
        print('NOTE: Missing phases (will be skipped): %s' % ', '.join(missing))

    print('Loaded phases: %s' % ', '.join(found))

    # Compute and print summary stats
    for name in found:
        d = phases[name]
        cum = _cumulative_distance(d['x'], d['y'])
        print('  %-8s  samples=%d  total_dist=%.2f m  duration=%.1f s' % (
            name, len(d['x']), cum[-1], d['elapsed_s'][-1]))

    # Plot
    make_plot(phases, out_png=out_png, show=not args.no_show)
