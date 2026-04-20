#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lk_tracker.py
=============
Standalone Lucas-Kanade optical-flow + ORB keyframe-matching module for
the JetRacer teach-repeat stack.

No ROS imports — importable and testable without a ROS environment.
Python 2/3 dual-compatible (uses __future__ imports throughout).

Designed for the paper's native 115×44 image resolution on a Jetson Nano:
  - All heavy lifting delegated to OpenCV C++ (no per-pixel Python loops).
  - Vectorised numpy for all arithmetic.
  - CLAHE applied once per frame before any feature operation.

Public API
----------
  tracker = LKTracker(img_w=115, img_h=44)

  # Feed the current live gray frame (uint8):
  flow_vecs, valid_mask = tracker.track(gray_uint8)

  # Match current frame against a stored keyframe gray:
  offset_px, confidence = tracker.match_to_keyframe(curr_gray, kf_gray)

  # Smoothed optical-flow magnitude (along-path speed proxy):
  speed = tracker.estimate_flow_speed()
"""

from __future__ import division, print_function

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Lucas-Kanade parameters
# ---------------------------------------------------------------------------
_LK_WIN_SIZE  = (15, 15)
_LK_MAX_LEVEL = 2
_LK_CRITERIA  = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

# Shi-Tomasi feature detection
_ST_MAX_FEATURES   = 50
_ST_QUALITY        = 0.01
_ST_MIN_DIST       = 5

# ORB parameters
_ORB_N_FEATURES    = 50

# Forward-backward error threshold for track validation (pixels)
_FB_ERROR_THRESH   = 2.0

# Minimum valid tracked points before soft-reinit
_MIN_TRACKED_PTS   = 15

# CLAHE
_CLAHE_CLIP_LIMIT  = 2.0
_CLAHE_TILE_GRID   = (4, 4)

# RANSAC homography inlier distance (pixels)
_RANSAC_THRESH     = 5.0

# IIR low-pass alpha for flow speed estimate
_IIR_ALPHA         = 0.15


class LKTracker(object):
    """
    Lucas-Kanade optical flow tracker with ORB keyframe matching.

    Parameters
    ----------
    img_w : int
        Expected image width in pixels (default 115).
    img_h : int
        Expected image height in pixels (default 44).
    sky_fraction : float
        Top fraction of the image to mask out during feature detection
        (sky / ceiling rejection).  Default 0.20 (top 20 %).
    lk_flow_alpha : float
        IIR smoothing coefficient for ``estimate_flow_speed()``.
        Range (0, 1] — smaller = heavier smoothing.  Default 0.15.
    """

    def __init__(self, img_w=115, img_h=44, sky_fraction=0.20,
                 lk_flow_alpha=_IIR_ALPHA):
        self.img_w         = int(img_w)
        self.img_h         = int(img_h)
        self.sky_fraction  = float(sky_fraction)
        self._alpha        = float(lk_flow_alpha)

        # CLAHE instance (shared across calls for efficiency)
        self._clahe = cv2.createCLAHE(
            clipLimit=_CLAHE_CLIP_LIMIT,
            tileGridSize=_CLAHE_TILE_GRID,
        )

        # ORB detector (shared instance). Tuned for extremely small 115x44 images.
        # Defaults (edgeThreshold=31) mathematically kill 100% of features on an image 44px tall.
        self._orb = cv2.ORB_create(
            nfeatures=_ORB_N_FEATURES,
            edgeThreshold=5,
            patchSize=15,
            fastThreshold=10
        )

        # BFMatcher with cross-check (no ratio-test needed)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Tracking state
        self._prev_gray  = None   # uint8 CLAHE gray of previous frame
        self._prev_pts   = None   # (N, 1, 2) float32 tracked points
        self._flow_vecs  = None   # (N, 2) float32 displacement vectors
        self._valid_mask = None   # (N,) bool

        # IIR-filtered flow speed
        self._flow_speed_iir = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_clahe(self, gray_uint8):
        """Apply CLAHE to a uint8 grayscale image. Returns uint8."""
        if gray_uint8.dtype != np.uint8:
            gray_uint8 = gray_uint8.astype(np.uint8)
        return self._clahe.apply(gray_uint8)

    def _sky_mask(self):
        """
        Build a uint8 mask that zeros out the top `sky_fraction` of the image.
        Shi-Tomasi will only detect points where mask != 0.
        Returns an array of shape (img_h, img_w) dtype uint8.
        """
        mask = np.ones((self.img_h, self.img_w), dtype=np.uint8) * 255
        sky_rows = int(round(self.img_h * self.sky_fraction))
        if sky_rows > 0:
            mask[:sky_rows, :] = 0
        return mask

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_features(self, gray_uint8):
        """
        Detect Shi-Tomasi corner features, ignoring the top sky_fraction.

        Parameters
        ----------
        gray_uint8 : np.ndarray  shape (H, W)  dtype uint8
            Already CLAHE-enhanced grayscale frame.

        Returns
        -------
        pts : np.ndarray  shape (N, 1, 2)  dtype float32
            Detected feature points, ready for ``cv2.calcOpticalFlowPyrLK``.
            Returns None if no features found.
        """
        enhanced = self._apply_clahe(gray_uint8)
        mask     = self._sky_mask()

        corners = cv2.goodFeaturesToTrack(
            enhanced,
            maxCorners=_ST_MAX_FEATURES,
            qualityLevel=_ST_QUALITY,
            minDistance=_ST_MIN_DIST,
            mask=mask,
        )
        return corners  # (N,1,2) float32 or None

    def track(self, curr_frame):
        """
        Frame-to-frame Lucas-Kanade optical flow with forward-backward
        error check.  Triggers a soft re-initialisation if fewer than
        ``_MIN_TRACKED_PTS`` valid points remain or if the median FB
        error exceeds ``_FB_ERROR_THRESH`` pixels.

        Parameters
        ----------
        curr_frame : np.ndarray  shape (H, W) or (H, W, C)  uint8
            Current camera frame.  Converted to gray internally.

        Returns
        -------
        flow_vecs : np.ndarray  shape (N, 2)  float32  or  None
            Displacement vectors (dx, dy) for each valid tracked point.
        valid_mask : np.ndarray  shape (N,)  bool  or  None
            True for points that passed the FB check.

        Side effects
        ------------
        Updates ``self._flow_vecs``, ``self._valid_mask``,
        ``self._prev_gray``, ``self._prev_pts``, and
        ``self._flow_speed_iir``.
        """
        # Convert to uint8 gray
        if curr_frame.ndim == 3:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = curr_frame.copy()
        if curr_gray.dtype != np.uint8:
            curr_gray = np.clip(curr_gray * 255.0, 0, 255).astype(np.uint8)

        curr_clahe = self._apply_clahe(curr_gray)

        # --- First frame or hard reinit ---
        if self._prev_gray is None or self._prev_pts is None:
            self._prev_gray = curr_clahe
            self._prev_pts  = self.detect_features(curr_clahe)
            self._flow_vecs  = None
            self._valid_mask = None
            return None, None

        # --- Forward LK pass ---
        next_pts, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, curr_clahe, self._prev_pts, None,
            winSize=_LK_WIN_SIZE,
            maxLevel=_LK_MAX_LEVEL,
            criteria=_LK_CRITERIA,
        )

        # Guard: calcOpticalFlowPyrLK can return None
        if next_pts is None or status_fwd is None:
            self._prev_gray = curr_clahe
            self._prev_pts  = self.detect_features(curr_clahe)
            self._flow_vecs  = None
            self._valid_mask = None
            return None, None

        # --- Backward FB check ---
        prev_pts_back, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
            curr_clahe, self._prev_gray, next_pts, None,
            winSize=_LK_WIN_SIZE,
            maxLevel=_LK_MAX_LEVEL,
            criteria=_LK_CRITERIA,
        )

        if prev_pts_back is None or status_bwd is None:
            # Degenerate — accept forward result only
            fb_error = np.zeros(len(self._prev_pts), dtype=np.float32)
        else:
            # FB error: L2 distance between original and round-tripped points
            diff     = self._prev_pts - prev_pts_back          # (N,1,2)
            fb_error = np.sqrt(
                np.sum(diff ** 2, axis=2)
            ).ravel()                                           # (N,)

        fwd_ok    = status_fwd.ravel().astype(bool)
        bwd_ok    = (status_bwd.ravel().astype(bool)
                     if prev_pts_back is not None else fwd_ok)
        fb_ok     = fwd_ok & bwd_ok & (fb_error < _FB_ERROR_THRESH)

        # --- Flow vectors for valid points ---
        n_valid   = int(np.sum(fb_ok))
        if n_valid > 0:
            p0 = self._prev_pts[fb_ok].reshape(-1, 2)   # (M, 2)
            p1 = next_pts[fb_ok].reshape(-1, 2)          # (M, 2)
            flow_vecs = p1 - p0                           # (M, 2)
        else:
            flow_vecs = np.zeros((0, 2), dtype=np.float32)

        # --- IIR flow speed update ---
        if n_valid > 0:
            magnitudes = np.sqrt(
                flow_vecs[:, 0] ** 2 + flow_vecs[:, 1] ** 2
            )
            instant_speed = float(np.median(magnitudes))
        else:
            instant_speed = 0.0
        self._flow_speed_iir = (
            self._alpha * instant_speed
            + (1.0 - self._alpha) * self._flow_speed_iir
        )

        # --- Soft reinit decision ---
        median_fb = float(np.median(fb_error[fwd_ok])) if np.any(fwd_ok) else 0.0
        needs_reinit = (
            n_valid < _MIN_TRACKED_PTS
            or median_fb > _FB_ERROR_THRESH
        )

        # Advance state
        self._flow_vecs  = flow_vecs
        self._valid_mask = fb_ok
        self._prev_gray  = curr_clahe

        if needs_reinit:
            # Soft reinit: detect fresh features for the NEXT frame
            self._prev_pts = self.detect_features(curr_clahe)
        else:
            # Keep only validated surviving points
            self._prev_pts = next_pts[fb_ok].reshape(-1, 1, 2)

        return flow_vecs, fb_ok

    def match_to_keyframe(self, curr_frame, keyframe_gray):
        """
        ORB-based keyframe matching with RANSAC homography outlier rejection.

        Computes a horizontal pixel offset between the current frame and a
        stored keyframe — suitable for yaw / heading correction.

        Parameters
        ----------
        curr_frame : np.ndarray  shape (H, W) or (H, W, C)  uint8
        keyframe_gray : np.ndarray  shape (H, W)  uint8 or float32
            The stored teach keyframe.  Accepts both the raw uint8 gray
            from disk and the float32 normalised descriptor used by NCC.

        Returns
        -------
        horizontal_offset_px : float or None
            Positive = current frame shifted right relative to keyframe.
            None if matching failed (too few matches, RANSAC failed, etc.).
        confidence : float
            Scalar in [0, 1].  Ratio of RANSAC inliers to total matches.
            0.0 on failure.
        """
        # --- Convert inputs to uint8 gray ---
        if curr_frame.ndim == 3:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = curr_frame.copy()
        if curr_gray.dtype != np.uint8:
            curr_gray = np.clip(curr_gray * 255.0, 0, 255).astype(np.uint8)

        if keyframe_gray.dtype != np.uint8:
            kf_gray = np.clip(keyframe_gray * 255.0, 0, 255).astype(np.uint8)
        else:
            kf_gray = keyframe_gray

        # --- CLAHE both images ---
        curr_clahe = self._apply_clahe(curr_gray)
        kf_clahe   = self._apply_clahe(kf_gray)

        # --- ORB detect + describe ---
        kp_curr, des_curr = self._orb.detectAndCompute(curr_clahe, None)
        kp_kf,   des_kf   = self._orb.detectAndCompute(kf_clahe,   None)

        if des_curr is None or des_kf is None:
            return None, 0.0
        if len(kp_curr) < 4 or len(kp_kf) < 4:
            return None, 0.0

        # --- BFMatcher (cross-check = True, so no ratio test needed) ---
        matches = self._bf.match(des_curr, des_kf)
        if len(matches) < 4:
            return None, 0.0

        # Sort by distance for reproducibility (not strictly required)
        matches = sorted(matches, key=lambda m: m.distance)

        src_pts = np.array(
            [kp_curr[m.queryIdx].pt for m in matches],
            dtype=np.float32,
        ).reshape(-1, 1, 2)
        dst_pts = np.array(
            [kp_kf[m.trainIdx].pt for m in matches],
            dtype=np.float32,
        ).reshape(-1, 1, 2)

        # --- RANSAC homography to remove geometric outliers ---
        try:
            H, inlier_mask = cv2.findHomography(
                src_pts, dst_pts,
                cv2.RANSAC,
                _RANSAC_THRESH,
            )
        except cv2.error:
            return None, 0.0

        if H is None or inlier_mask is None:
            return None, 0.0

        n_inliers = int(np.sum(inlier_mask))
        n_matches  = len(matches)

        if n_inliers < 4:
            return None, 0.0

        confidence = float(n_inliers) / float(n_matches)

        # --- Horizontal offset = mean x-displacement of inlier pair ---
        inlier_idx = inlier_mask.ravel().astype(bool)
        src_inlier = src_pts[inlier_idx].reshape(-1, 2)
        dst_inlier = dst_pts[inlier_idx].reshape(-1, 2)
        # offset > 0  →  current scene appears shifted RIGHT vs keyframe
        offset_px  = float(np.median(src_inlier[:, 0] - dst_inlier[:, 0]))

        return offset_px, confidence

    def estimate_flow_speed(self):
        """
        Return the IIR-smoothed median optical-flow magnitude (px/frame).

        This is updated by every call to ``track()``.  Useful as a
        proxy for along-path travel speed between keyframe correction
        events.

        Returns
        -------
        float
            Smoothed flow speed in pixels per frame.  0.0 before the
            first successful tracking result.
        """
        return self._flow_speed_iir

    def reset(self):
        """Hard reset — clears all tracking state."""
        self._prev_gray       = None
        self._prev_pts        = None
        self._flow_vecs       = None
        self._valid_mask      = None
        self._flow_speed_iir  = 0.0


# ---------------------------------------------------------------------------
# Quick self-test (run as: python lk_tracker.py)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys

    print("LKTracker self-test")
    print("  OpenCV version:", cv2.__version__)

    W, H = 115, 44
    tracker = LKTracker(img_w=W, img_h=H)

    # Synthetic frame: random noise so Shi-Tomasi finds corners
    rng   = np.random.RandomState(42)
    frame0 = rng.randint(0, 256, (H, W), dtype=np.uint8)

    # Shift frame0 by 5 px horizontally to create frame1
    shift_px = 5
    frame1   = np.zeros_like(frame0)
    frame1[:, shift_px:] = frame0[:, :W - shift_px]

    # Track
    tracker.track(frame0)
    flow_vecs, valid = tracker.track(frame1)
    if flow_vecs is not None and len(flow_vecs) > 0:
        mean_dx = float(np.mean(flow_vecs[:, 0]))
        print("  track(): mean dx = %.2f px  (expected ~%.1f)" % (mean_dx, float(shift_px)))
    else:
        print("  track(): no valid flow  (OK for purely random noise)")

    # match_to_keyframe
    offset, conf = tracker.match_to_keyframe(frame1, frame0)
    if offset is not None:
        print("  match_to_keyframe(): offset=%.2f px  confidence=%.3f  "
              "(expected ~%.1f)" % (offset, conf, float(shift_px)))
    else:
        print("  match_to_keyframe(): insufficient features on synthetic image "
              "(expected — pure noise has poor ORB corners)")

    # estimate_flow_speed
    speed = tracker.estimate_flow_speed()
    print("  estimate_flow_speed(): %.4f px/frame" % speed)

    print("Self-test complete.")
    sys.exit(0)
