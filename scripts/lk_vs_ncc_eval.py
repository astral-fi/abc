#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lk_vs_ncc_eval.py
=================
Standalone evaluation script — NO ROS master required.

Loads a recorded teach run (saved images + poses in the standard
<run_dir>/full/ + <run_dir>/frame_XXXXXX_pose.txt layout),
computes BOTH the LK and phase-correlate (NCC/FFT) horizontal offset
for every consecutive keyframe pair, and outputs a CSV + optional plot.

Usage
-----
  python lk_vs_ncc_eval.py --run_dir ~/jetracer/teach_runs/2026-04-02_10-30-00
                           [--gt_csv  path/to/ground_truth.csv]
                           [--out_csv results.csv]
                           [--plot]

CSV format
----------
  frame_idx, timestamp, lk_offset_px, ncc_offset_px, lk_confidence,
  lateral_error_m

Ground-truth CSV format (optional)
-----------------------------------
  timestamp, lateral_error_m
  (floating-point seconds; joined to frame timestamps by nearest-neighbour)

Python 2/3 compatible.
"""

from __future__ import division, print_function

import argparse
import csv
import glob
import json
import math
import os
import re
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Import the local LK module (must be in the same directory or on PYTHONPATH)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from lk_tracker import LKTracker
except ImportError as _e:
    sys.exit("ERROR: Cannot import lk_tracker.py — %s\n"
             "Make sure lk_vs_ncc_eval.py and lk_tracker.py are in the "
             "same directory." % _e)


# ---------------------------------------------------------------------------
# Constants — must match jetracer_teach_repeat_core.launch defaults
# ---------------------------------------------------------------------------
DEFAULT_RESIZE_W   = 320
DEFAULT_RESIZE_H   = 120
DEFAULT_FOV_DEG    = 160.0
DEFAULT_HEADING_GAIN = 1.0   # applied to both for comparison; set to 1 to
                              # get raw pixel offset, not yaw angle


# ---------------------------------------------------------------------------
# Helpers mirroring visual_pose_localiser._preprocess()
# ---------------------------------------------------------------------------

def _preprocess(bgr, resize_w, resize_h):
    """
    Reproduce the NCC preprocessing pipeline from visual_pose_localiser.py.
    Returns a float32 HxW array (zero-mean, unit-std, local contrast).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if resize_w > 0 and resize_h > 0:
        gray = cv2.resize(gray, (resize_w, resize_h),
                          interpolation=cv2.INTER_AREA)
    img   = gray.astype(np.float32) / 255.0
    blur  = cv2.GaussianBlur(img, (0, 0), 1.2)
    norm  = img - blur
    std   = float(np.std(norm))
    if std < 1e-6:
        std = 1.0
    return norm / std


def _ncc_offset(teach_desc, query_desc):
    """
    Phase-correlate horizontal offset (reproduces _estimate_yaw_error).
    Returns dx_px (float).
    """
    shift, _ = cv2.phaseCorrelate(
        teach_desc.astype(np.float32),
        query_desc.astype(np.float32),
    )
    return float(shift[0])


# ---------------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------------

def _extract_idx(path):
    m = re.search(r'(\d+)', os.path.basename(path))
    return int(m.group(1)) if m else None


def _read_pose_stamp(path):
    """Read pose file; return timestamp from file mtime as fallback."""
    try:
        with open(path, 'r') as fh:
            data = json.loads(fh.read())
        # The original format does not store a timestamp directly in the JSON;
        # use file modification time as a reasonable proxy.
        stamp = data.get('stamp', os.path.getmtime(path))
        return float(stamp)
    except Exception:
        return float(os.path.getmtime(path))


def _load_run(run_dir, resize_w, resize_h):
    """
    Load all keyframes from a teach run directory.

    Returns a sorted list of dicts:
      {
        'idx':       int,
        'timestamp': float (seconds, best-effort),
        'img_path':  str,
        'bgr':       ndarray (H, W, 3) uint8,
        'gray_u8':   ndarray (H, W) uint8  (resized, pre-CLAHE),
        'desc':      ndarray (resize_h, resize_w) float32,
      }
    """
    run_dir = os.path.expanduser(run_dir)
    full_dir = os.path.join(run_dir, 'full')

    if not os.path.isdir(full_dir):
        sys.exit("ERROR: 'full' subdirectory not found in %s" % run_dir)

    img_files  = sorted(glob.glob(os.path.join(full_dir, '*.png')))
    pose_files = sorted(glob.glob(os.path.join(run_dir, '*_pose.txt')))

    img_map  = {}
    for p in img_files:
        i = _extract_idx(p)
        if i is not None:
            img_map[i] = p

    pose_map = {}
    for p in pose_files:
        i = _extract_idx(p)
        if i is not None:
            pose_map[i] = p

    common_idx = sorted(set(img_map) & set(pose_map))
    if not common_idx:
        sys.exit("ERROR: No matched image+pose pairs found in %s" % run_dir)

    frames = []
    for i in common_idx:
        bgr = cv2.imread(img_map[i], cv2.IMREAD_COLOR)
        if bgr is None:
            print("WARNING: Cannot read %s — skipping." % img_map[i])
            continue

        # Resized gray (uint8) for LK
        gray_small = cv2.cvtColor(
            cv2.resize(bgr, (resize_w, resize_h), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
        # Float32 descriptor for NCC
        desc = _preprocess(bgr, resize_w, resize_h)

        stamp = _read_pose_stamp(pose_map[i])

        frames.append({
            'idx':       i,
            'timestamp': stamp,
            'img_path':  img_map[i],
            'bgr':       bgr,
            'gray_u8':   gray_small,
            'desc':      desc,
        })

    print("Loaded %d keyframes from %s" % (len(frames), run_dir))
    return frames


def _load_ground_truth(gt_csv_path):
    """
    Load ground-truth CSV with columns: timestamp, lateral_error_m.
    Returns list of (timestamp_float, lateral_error_float).
    """
    rows = []
    try:
        with open(gt_csv_path, 'r') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    rows.append((
                        float(row['timestamp']),
                        float(row['lateral_error_m']),
                    ))
                except (KeyError, ValueError):
                    pass
    except Exception as e:
        print("WARNING: Could not load ground-truth CSV (%s): %s" % (
            gt_csv_path, e))
    return rows


def _nearest_gt(gt_rows, timestamp):
    """
    Find the ground-truth lateral error nearest to `timestamp`.
    Returns float or None.
    """
    if not gt_rows:
        return None
    best = min(gt_rows, key=lambda r: abs(r[0] - timestamp))
    return best[1]


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(run_dir, gt_csv=None, out_csv='lk_vs_ncc_results.csv',
                   plot=False, resize_w=DEFAULT_RESIZE_W,
                   resize_h=DEFAULT_RESIZE_H):
    """
    Core evaluation function.

    Parameters
    ----------
    run_dir   : str   Path to the teach run directory.
    gt_csv    : str or None  Path to ground-truth lateral-error CSV.
    out_csv   : str   Output CSV filename.
    plot      : bool  Show matplotlib plot when done.
    resize_w  : int   Target image width (must match teach parameters).
    resize_h  : int   Target image height (must match teach parameters).
    """
    frames  = _load_run(run_dir, resize_w, resize_h)
    gt_rows = _load_ground_truth(gt_csv) if gt_csv else []

    tracker = LKTracker(img_w=resize_w, img_h=resize_h)

    results = []   # list of dicts for CSV

    prev_frame = None

    for idx, frame in enumerate(frames):
        row = {
            'frame_idx':      frame['idx'],
            'timestamp':      frame['timestamp'],
            'lk_offset_px':   None,
            'ncc_offset_px':  None,
            'lk_confidence':  None,
            'lateral_error_m': _nearest_gt(gt_rows, frame['timestamp']),
        }

        if prev_frame is not None:
            # ---- NCC (phase correlate) offset ----
            ncc_dx = _ncc_offset(prev_frame['desc'], frame['desc'])
            row['ncc_offset_px'] = ncc_dx

            # ---- LK (ORB match) offset + rotation ----
            # match_to_keyframe now returns (offset_px, rotation_rad, confidence)
            lk_dx, lk_rot_rad, lk_conf = tracker.match_to_keyframe(
                frame['gray_u8'], prev_frame['gray_u8']
            )
            row['lk_offset_px']  = lk_dx      # may be None
            row['lk_confidence'] = lk_conf

        # Also push frame through LK tracking (updates IIR flow speed)
        tracker.track(frame['gray_u8'])

        results.append(row)
        prev_frame = frame

        if (idx + 1) % 10 == 0 or (idx + 1) == len(frames):
            sys.stdout.write("\rProcessed %d / %d frames..." % (
                idx + 1, len(frames)))
            sys.stdout.flush()

    print("")  # newline after progress

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    fieldnames = [
        'frame_idx', 'timestamp',
        'lk_offset_px', 'ncc_offset_px',
        'lk_confidence', 'lateral_error_m',
    ]

    out_path = out_csv
    if not os.path.isabs(out_path):
        out_path = os.path.join(run_dir, out_path)

    with open(out_path, 'w') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames,
                                lineterminator='\n')
        writer.writeheader()
        for row in results:
            # Replace None with empty string for cleaner CSV
            clean_row = {
                k: ('' if v is None else v)
                for k, v in row.items()
            }
            writer.writerow(clean_row)

    print("CSV written to: %s" % out_path)

    # ------------------------------------------------------------------
    # Optional plot
    # ------------------------------------------------------------------
    if plot:
        _plot_results(results, gt_rows)

    return results


def _plot_results(results, gt_rows):
    """Matplotlib side-by-side comparison of LK vs NCC offset signals."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available — skipping plot.")
        return

    frame_ids = [r['frame_idx'] for r in results]

    lk_vals  = [r['lk_offset_px']  if r['lk_offset_px']  is not None else float('nan')
                for r in results]
    ncc_vals = [r['ncc_offset_px'] if r['ncc_offset_px'] is not None else float('nan')
                for r in results]
    conf_vals = [r['lk_confidence'] if r['lk_confidence'] is not None else float('nan')
                 for r in results]
    gt_vals = [r['lateral_error_m'] if r['lateral_error_m'] is not None else float('nan')
               for r in results]

    has_gt = any(not math.isnan(v) for v in gt_vals)
    n_rows = 3 if has_gt else 2

    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows), sharex=True)

    # --- Panel 1: LK vs NCC offsets ---
    axes[0].plot(frame_ids, ncc_vals, label='NCC (phase-correlate)', color='steelblue',
                 linewidth=1.2)
    axes[0].plot(frame_ids, lk_vals,  label='LK (ORB match)',         color='darkorange',
                 linewidth=1.2, linestyle='--')
    axes[0].set_ylabel('Horizontal offset (px)')
    axes[0].set_title('LK vs NCC Horizontal Offset Signals')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='black', linewidth=0.5)

    # --- Panel 2: LK confidence ---
    axes[1].fill_between(frame_ids, conf_vals, alpha=0.4, color='darkorange')
    axes[1].plot(frame_ids, conf_vals, color='darkorange', linewidth=1.0)
    axes[1].axhline(0.4, color='red', linewidth=0.8, linestyle='--',
                    label='fallback threshold (0.4)')
    axes[1].set_ylabel('LK confidence')
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    # --- Panel 3: lateral error (optional) ---
    if has_gt:
        axes[2].plot(frame_ids, gt_vals, color='forestgreen', linewidth=1.2)
        axes[2].set_ylabel('Lateral error (m)')
        axes[2].set_xlabel('Frame index')
        axes[2].grid(True, alpha=0.3)
    else:
        axes[-1].set_xlabel('Frame index')

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser(
        description='Evaluate LK vs NCC offset signals on a recorded teach run.'
    )
    p.add_argument(
        '--run_dir', required=True,
        help='Path to the teach run directory (must contain full/ and *_pose.txt).',
    )
    p.add_argument(
        '--gt_csv', default=None,
        help='Optional ground-truth CSV (columns: timestamp, lateral_error_m).',
    )
    p.add_argument(
        '--out_csv', default='lk_vs_ncc_results.csv',
        help='Output CSV filename (default: lk_vs_ncc_results.csv in run_dir).',
    )
    p.add_argument(
        '--plot', action='store_true',
        help='Show matplotlib comparison plot after writing CSV.',
    )
    p.add_argument(
        '--resize_w', type=int, default=DEFAULT_RESIZE_W,
        help='Image resize width — must match teach parameters (default: 115).',
    )
    p.add_argument(
        '--resize_h', type=int, default=DEFAULT_RESIZE_H,
        help='Image resize height — must match teach parameters (default: 44).',
    )
    return p


if __name__ == '__main__':
    args = _build_parser().parse_args()
    run_evaluation(
        run_dir=args.run_dir,
        gt_csv=args.gt_csv,
        out_csv=args.out_csv,
        plot=args.plot,
        resize_w=args.resize_w,
        resize_h=args.resize_h,
    )
