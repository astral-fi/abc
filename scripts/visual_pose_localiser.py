#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import base64
import glob
import json
import math
import os
import re
import sys

import cv2
import numpy as np
import rospy
import tf.transformations as tft
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose2D, PoseWithCovarianceStamped
from sensor_msgs.msg import Image
from std_msgs.msg import String

# ---------------------------------------------------------------------------
# LK tracker — optional; falls back gracefully if import fails
# (e.g. first run before lk_tracker.py is deployed alongside this file)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from lk_tracker import LKTracker
    _LK_AVAILABLE = True
except ImportError as _e:
    rospy.logwarn('[visual_pose_localiser] lk_tracker not available (%s). '
                  'LK hybrid mode will be disabled.', str(_e))
    _LK_AVAILABLE = False
    LKTracker = None


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


class VisualPoseLocaliser(object):
    """
    Visual + pose localiser for teach-repeat.

    - Uses stored pose files as nominal route
    - Uses image matching in a local search window to correct route index
    - Applies a bounded heading correction from horizontal image shift
    """

    def __init__(self):
        rospy.init_node('visual_pose_localiser')

        self.bridge = CvBridge()

        # Inputs / outputs
        self.run_dir_param = rospy.get_param('~run_dir', os.path.expanduser('~/jetracer/teach_runs'))
        self.image_topic = rospy.get_param('~image_topic', '/csi_cam_0/image_raw')
        self.odom_topic = rospy.get_param('~odom_topic', '/odom_combined')
        self.goal_topic = rospy.get_param('~goal_topic', 'goal')

        # Route / progress
        self.goal_radius = float(rospy.get_param('~goal_radius', 0.25))
        self.publish_hz = float(rospy.get_param('~publish_hz', 15.0))
        self.loop_route = bool(rospy.get_param('~loop_route', False))
        self.lookahead_steps = int(rospy.get_param('~lookahead_steps', 2))
        self.behind_skip_threshold_deg = float(rospy.get_param('~behind_skip_threshold_deg', 85.0))
        self.max_skip_ahead = int(rospy.get_param('~max_skip_ahead', 200))
        # ~start_index: force a fixed starting waypoint index.
        # Set to 0 to always begin from the first waypoint (useful when odom
        # is not reset between teach and repeat).  -1 means "nearest" (default).
        self.start_index = int(rospy.get_param('~start_index', -1))

        # Visual matching
        self.search_range = int(rospy.get_param('~search_range', 8))
        self.corr_threshold = float(rospy.get_param('~corr_threshold', 0.02))
        self.max_index_jump = int(rospy.get_param('~max_index_jump', 5))
        self.heading_gain = float(rospy.get_param('~heading_gain', 0.8))
        self.heading_sign = float(rospy.get_param('~heading_sign', -1.0))
        self.max_heading_correction_deg = float(rospy.get_param('~max_heading_correction_deg', 12.0))

        # Image preprocessing params (must match teach/repeat settings)
        self.resize_w = int(rospy.get_param('/image_resize_width', rospy.get_param('~image_resize_width', 115)))
        self.resize_h = int(rospy.get_param('/image_resize_height', rospy.get_param('~image_resize_height', 44)))
        self.image_fov_deg = float(rospy.get_param('/image_field_of_view_width_deg', 160.0))
        self.image_fov_rad = math.radians(self.image_fov_deg)
        
        # Mask out the ceiling to prevent LK ORB tracker from tracking zero-parallax distant features
        self.sky_mask_ratio = float(rospy.get_param('~sky_mask_ratio', 0.20))

        # ── LK hybrid correction params (all prefixed lk_) ──────────────────
        # ~use_lk_hybrid: enable LK-first / NCC-fallback mode.
        self.use_lk_hybrid = bool(rospy.get_param('~use_lk_hybrid', True))
        # ~lk_confidence_threshold: fall back to NCC when LK confidence < this.
        self.lk_confidence_threshold = float(
            rospy.get_param('~lk_confidence_threshold', 0.4))
        # ~lk_flow_alpha: IIR smoothing for LK flow speed estimate.
        self.lk_flow_alpha = float(rospy.get_param('~lk_flow_alpha', 0.15))

        # Disable if lk_tracker.py could not be imported
        if self.use_lk_hybrid and not _LK_AVAILABLE:
            rospy.logwarn('[visual_pose_localiser] ~use_lk_hybrid=true but '
                          'lk_tracker.py is not importable. '
                          'Falling back to NCC-only mode.')
            self.use_lk_hybrid = False

        # Instantiate tracker (only when hybrid mode is active)
        self._lk_tracker = (
            LKTracker(
                img_w=self.resize_w,
                img_h=self.resize_h,
                lk_flow_alpha=self.lk_flow_alpha,
            )
            if self.use_lk_hybrid else None
        )

        self.max_heading_correction_rad = math.radians(self.max_heading_correction_deg)

        # State
        self.have_odom = False
        self.have_image = False
        self.rx = 0.0
        self.ry = 0.0
        self.ryaw = 0.0
        self.current_desc = None
        self.current_gray_u8 = None   # uint8 resized (115x44) for LK speed tracking
        self.current_gray_full = None # uint8 full res for exact ORB keyframe matching
        self.current_img_stamp = None
        self.idx = 0
        self._last_correction_source = 'none'  # 'lk' or 'ncc'
        self._last_lk_confidence = 0.0
        
        # Dead-reckoning state for blindly advancing index during visual dropouts
        self.last_rx = None
        self.last_ry = None
        self.accum_dist = 0.0

        self.run_dir, self.samples = self._load_route(self.run_dir_param)
        if not self.samples:
            rospy.logfatal('[visual_pose_localiser] No matched pose/image samples found in %s', self.run_dir)
            raise RuntimeError('No samples')

        self.goal_pub = rospy.Publisher(self.goal_topic, Pose2D, queue_size=1)
        # Debug topic: JSON string with correction source + LK confidence each cycle
        self.debug_pub = rospy.Publisher(
            '/teach_repeat/correction_debug', String, queue_size=5)
        rospy.Subscriber(self.odom_topic, PoseWithCovarianceStamped, self._odom_cb, queue_size=1)
        rospy.Subscriber(self.image_topic, Image, self._image_cb, queue_size=1)

        rospy.loginfo('[visual_pose_localiser] Loaded %d samples from %s', len(self.samples), self.run_dir)
        rospy.loginfo('[visual_pose_localiser] image_topic=%s odom_topic=%s search_range=%d corr_threshold=%.2f',
                      self.image_topic, self.odom_topic, self.search_range, self.corr_threshold)
        rospy.loginfo('[visual_pose_localiser] use_lk_hybrid=%s lk_confidence_threshold=%.2f lk_flow_alpha=%.3f',
                      str(self.use_lk_hybrid), self.lk_confidence_threshold, self.lk_flow_alpha)

    @staticmethod
    def _extract_idx(path):
        m = re.search(r'(\d+)', os.path.basename(path))
        return int(m.group(1)) if m else None

    @staticmethod
    def _read_pose(path):
        with open(path, 'r') as fh:
            data = json.loads(fh.read())
        x = float(data['position']['x'])
        y = float(data['position']['y'])
        qx = float(data['orientation']['x'])
        qy = float(data['orientation']['y'])
        qz = float(data['orientation']['z'])
        qw = float(data['orientation']['w'])
        _, _, yaw = tft.euler_from_quaternion([qx, qy, qz, qw])
        return x, y, yaw

    def _preprocess(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self.resize_w > 0 and self.resize_h > 0:
            gray = cv2.resize(gray, (self.resize_w, self.resize_h), interpolation=cv2.INTER_AREA)

        img = gray.astype(np.float32) / 255.0

        # Local contrast normalization (robust to lighting change)
        blur = cv2.GaussianBlur(img, (0, 0), 1.2)
        norm = img - blur
        std = float(np.std(norm))
        if std < 1e-6:
            std = 1.0
        norm = norm / std
        return norm

    def _resolve_run_dir(self, path):
        path = os.path.expanduser(path)
        if not os.path.isdir(path):
            return path

        pose_files = glob.glob(os.path.join(path, '*_pose.txt'))
        if pose_files:
            return path

        subs = [os.path.join(path, d) for d in sorted(os.listdir(path)) if os.path.isdir(os.path.join(path, d))]
        candidates = []
        for d in subs:
            if glob.glob(os.path.join(d, '*_pose.txt')):
                candidates.append(d)
        if candidates:
            return candidates[-1]
        return path

    @staticmethod
    def _load_orb_from_sidecar(feat_path):
        """
        Load ORB keypoints and descriptors from a *_features.json sidecar
        written by data_save.py.  Returns (kp_list, des_ndarray) or (None, None)
        if the file is absent / malformed / empty (backward compatible).
        """
        if not os.path.isfile(feat_path):
            return None, None
        try:
            with open(feat_path, 'r') as fh:
                data = json.loads(fh.read())
            if not data:
                return None, None

            kp_dicts = data.get('orb_keypoints', [])
            kp_list  = [
                cv2.KeyPoint(
                    x=float(k['x']), y=float(k['y']),
                    _size=float(k.get('size', 1)),
                    _angle=float(k.get('angle', -1)),
                    _response=float(k.get('response', 0)),
                    _octave=int(k.get('octave', 0)),
                )
                for k in kp_dicts
            ]

            des_b64   = data.get('orb_descriptors_b64', '')
            des_shape = data.get('orb_descriptors_shape', [0, 32])
            if des_b64 and des_shape[0] > 0:
                raw = base64.b64decode(des_b64.encode('ascii'))
                des = np.frombuffer(raw, dtype=np.uint8).reshape(
                    des_shape[0], des_shape[1])
            else:
                des = None

            return kp_list, des
        except Exception:
            return None, None

    def _load_route(self, run_dir_param):
        run_dir = self._resolve_run_dir(run_dir_param)
        pose_files = sorted(glob.glob(os.path.join(run_dir, '*_pose.txt')))

        # Build image map by index from full/frame_XXXXXX.png
        img_files = sorted(glob.glob(os.path.join(run_dir, 'full', '*.png')))
        img_map = {}
        for p in img_files:
            i = self._extract_idx(p)
            if i is not None:
                img_map[i] = p

        n_orb_loaded = 0
        samples = []
        for pose_path in pose_files:
            i = self._extract_idx(pose_path)
            if i is None or i not in img_map:
                continue

            try:
                x, y, yaw = self._read_pose(pose_path)
                bgr = cv2.imread(img_map[i], cv2.IMREAD_COLOR)
                if bgr is None:
                    continue
                desc = self._preprocess(bgr)

                # ── Try to load ORB features from sidecar (new optional field) ──
                # Falls back gracefully if sidecar absent (old teach runs).
                feat_path = os.path.join(
                    run_dir, 'frame_%06d_features.json' % i)
                orb_kp, orb_des = self._load_orb_from_sidecar(feat_path)

                if orb_kp is None and self.use_lk_hybrid:
                    # Sidecar absent: extract ORB on-the-fly from stored image
                    try:
                        gray_small = cv2.cvtColor(
                            cv2.resize(bgr, (self.resize_w, self.resize_h),
                                       interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY,
                        )
                        orb_tmp = cv2.ORB_create(nfeatures=50)
                        orb_kp, orb_des = orb_tmp.detectAndCompute(
                            gray_small, None)
                    except Exception:
                        orb_kp, orb_des = None, None

                if orb_kp is not None:
                    n_orb_loaded += 1

                # Store the pristine full-resolution gray image to guarantee high-quality ORB features
                # CRITICAL: Crop the ceiling out so ORB only tracks floor features (avoids zero-parallax failure)
                gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                top_mask = int(gray_full.shape[0] * self.sky_mask_ratio)
                cropped_gray_full = gray_full[top_mask:, :]
                samples.append((x, y, yaw, desc, orb_kp, orb_des, cropped_gray_full))
            except Exception as e:
                rospy.logwarn('[visual_pose_localiser] Skipping sample %s (%s)', pose_path, str(e))

        if self.use_lk_hybrid:
            rospy.loginfo('[visual_pose_localiser] ORB features loaded/extracted '
                          'for %d / %d keyframes.', n_orb_loaded, len(samples))
        return run_dir, samples

    def _odom_cb(self, msg):
        q = msg.pose.pose.orientation
        _, _, yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.rx = msg.pose.pose.position.x
        self.ry = msg.pose.pose.position.y
        self.ryaw = yaw
        self.have_odom = True

    def _image_cb(self, msg):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.current_desc = self._preprocess(bgr)
            self.current_img_stamp = msg.header.stamp

            # Retain the full-resolution uncompressed image exclusively for LK ORB feature detection
            # CRITICAL: Crop the ceiling out identically to keyframe handling
            raw_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            top_mask = int(raw_gray.shape[0] * self.sky_mask_ratio)
            self.current_gray_full = raw_gray[top_mask:, :]
            
            # Build uint8 gray thumbnail for LK speed tracking
            gray_small = cv2.resize(raw_gray, (self.resize_w, self.resize_h), interpolation=cv2.INTER_AREA)
            self.current_gray_u8 = gray_small

            # Feed LK tracker every incoming frame (maintains optical flow state)
            if self._lk_tracker is not None:
                self._lk_tracker.track(gray_small)

            self.have_image = True
        except CvBridgeError as e:
            rospy.logwarn_throttle(2.0, '[visual_pose_localiser] Image conversion failed: %s', str(e))

    def _nearest_index(self):
        best_i = 0
        best_d2 = 1e18
        for i, (x, y, _, _, _, _) in enumerate(self.samples):
            dx = x - self.rx
            dy = y - self.ry
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        return best_i

    def _indices_in_window(self, center, half):
        n = len(self.samples)
        if n == 0:
            return []
        out = []
        for k in range(-half, half + 1):
            j = center + k
            if self.loop_route:
                j %= n
                out.append(j)
            else:
                if 0 <= j < n:
                    out.append(j)
        # de-dup while preserving order
        seen = set()
        uniq = []
        for j in out:
            if j not in seen:
                seen.add(j)
                uniq.append(j)
        return uniq

    @staticmethod
    def _corr(a, b):
        av = a.ravel()
        bv = b.ravel()
        am = av - np.mean(av)
        bm = bv - np.mean(bv)
        denom = (np.linalg.norm(am) * np.linalg.norm(bm))
        if denom < 1e-12:
            return -1.0
        return float(np.dot(am, bm) / denom)

    def _estimate_yaw_error(self, teach_desc, query_desc):
        # Returns horizontal shift in pixels; convert to radians by FOV.
        shift, _ = cv2.phaseCorrelate(teach_desc.astype(np.float32), query_desc.astype(np.float32))
        dx = float(shift[0])
        width_px = float(query_desc.shape[1])
        yaw = self.heading_sign * self.heading_gain * (dx / width_px) * self.image_fov_rad
        yaw = _clamp(yaw, -self.max_heading_correction_rad, self.max_heading_correction_rad)
        return yaw, dx

    def _bearing_to_point(self, gx, gy):
        return _wrap(math.atan2(gy - self.ry, gx - self.rx) - self.ryaw)

    def _skip_ahead_if_goal_behind(self):
        threshold = math.radians(self.behind_skip_threshold_deg)
        start_idx = self.idx
        best_idx = self.idx
        best_abs = float('inf')

        steps_limit = min(self.max_skip_ahead, len(self.samples))
        for step in range(steps_limit):
            if self.loop_route:
                cand = (start_idx + step) % len(self.samples)
            else:
                cand = start_idx + step
                if cand >= len(self.samples):
                    break

            gx, gy = self.samples[cand][0], self.samples[cand][1]
            b = abs(self._bearing_to_point(gx, gy))
            if b < best_abs:
                best_abs = b
                best_idx = cand
            if b <= threshold:
                self.idx = cand
                break
        else:
            self.idx = best_idx

    def _advance_open_loop(self):
        """
        Calculates physical delta-distance traveled frame-over-frame 
        to blindly push the waypoint index forward along the path.
        This guarantees the robot never gets permanently "stuck" doing donuts
        if the camera momentarily loses the path during violent circular turns.
        """
        if self.last_rx is None:
            self.last_rx = self.rx
            self.last_ry = self.ry
            return
            
        dist_moved = math.hypot(self.rx - self.last_rx, self.ry - self.last_ry)
        self.last_rx = self.rx
        self.last_ry = self.ry
        self.accum_dist += dist_moved
        
        while True:
            if self.loop_route:
                next_i = (self.idx + 1) % len(self.samples)
            else:
                next_i = min(self.idx + 1, len(self.samples) - 1)
                
            if next_i == self.idx:
                break
                
            step_d = math.hypot(self.samples[next_i][0] - self.samples[self.idx][0],
                                self.samples[next_i][1] - self.samples[self.idx][1])
                                
            # If the robot has physically driven past the next waypoint distance, snap to it!
            if self.accum_dist >= step_d and step_d > 0.001:
                self.accum_dist -= step_d
                self.idx = next_i
            else:
                break

    def _visual_update(self):
        """
        Returns (yaw_corr_rad, dx_px, best_corr).

        When ~use_lk_hybrid is True:
          1. Attempts LK ORB match_to_keyframe() for orientation correction.
          2. Falls back to NCC (phase correlate) if LK confidence < threshold
             or LK returns None.
        When ~use_lk_hybrid is False:
          Pure NCC path (identical to original behaviour).
        """
        if self.current_desc is None:
            return 0.0, 0.0, -1.0

        candidates = self._indices_in_window(self.idx, self.search_range)
        if not candidates:
            return 0.0, 0.0, -1.0

        # ── NCC index search (always runs — needed to advance self.idx) ────────
        best_i    = self.idx
        best_corr = -2.0
        for i in candidates:
            c = self._corr(self.samples[i][3], self.current_desc)
            if c > best_corr:
                best_corr = c
                best_i = i

        if best_corr < self.corr_threshold:
            self._last_correction_source = 'none'
            self._last_lk_confidence     = 0.0
            return 0.0, 0.0, best_corr

        # Bound sudden index jumps for stability
        delta = best_i - self.idx
        if self.loop_route:
            n = len(self.samples)
            if delta > n // 2:
                delta -= n
            elif delta < -n // 2:
                delta += n

        delta = int(_clamp(delta, -self.max_index_jump, self.max_index_jump))
        if self.loop_route:
            self.idx = (self.idx + delta) % len(self.samples)
        else:
            self.idx = int(_clamp(self.idx + delta, 0, len(self.samples) - 1))
            
        # If visual tracker snapped the index, reset the blind dead-reckoning residual
        if delta != 0:
            self.accum_dist = 0.0

        # ── Yaw correction: LK-first, NCC-fallback ──────────────────────────

        teach_sample = self.samples[self.idx]
        teach_desc   = teach_sample[3]

        used_lk      = False
        lk_confidence = 0.0

        if self.use_lk_hybrid and self._lk_tracker is not None \
                and self.current_gray_full is not None:
                
            # Use raw uncorrupted FULL RESOLUTION uint8 image stored at index 6
            teach_gray_full = teach_sample[6]

            lk_dx, lk_confidence = self._lk_tracker.match_to_keyframe(
                self.current_gray_full, teach_gray_full
            )

            if lk_dx is not None and lk_confidence >= self.lk_confidence_threshold:
                # LK succeeded: convert pixel offset to yaw correction
                full_width_px = float(self.current_gray_full.shape[1])
                yaw_corr = self.heading_sign * self.heading_gain * \
                    (lk_dx / full_width_px) * self.image_fov_rad
                yaw_corr = _clamp(yaw_corr,
                                  -self.max_heading_correction_rad,
                                  self.max_heading_correction_rad)
                used_lk  = True
                dx       = lk_dx

        if not used_lk:
            # NCC fallback (original logic; always available)
            yaw_corr, dx = self._estimate_yaw_error(teach_desc, self.current_desc)
            lk_confidence = 0.0

        self._last_correction_source = 'lk' if used_lk else 'ncc'
        self._last_lk_confidence     = lk_confidence
        return yaw_corr, dx, best_corr

    def run(self):
        rate = rospy.Rate(self.publish_hz)

        while not rospy.is_shutdown() and not self.have_odom:
            rospy.loginfo_throttle(2.0, '[visual_pose_localiser] Waiting for odom...')
            rate.sleep()

        while not rospy.is_shutdown() and not self.have_image:
            rospy.loginfo_throttle(2.0, '[visual_pose_localiser] Waiting for camera image...')
            rate.sleep()

        if self.start_index >= 0:
            self.idx = int(_clamp(self.start_index, 0, len(self.samples) - 1))
            rospy.loginfo('[visual_pose_localiser] Forced start index %d '
                         '(~start_index param)', self.idx)
        else:
            self.idx = self._nearest_index()
            rospy.loginfo('[visual_pose_localiser] Starting from nearest index %d', self.idx)

        self._finished = False

        while not rospy.is_shutdown():
            if self._finished:
                # Mission permanently complete. Halt and hold position without re-evaluating vision.
                msg = Pose2D()
                msg.x = self.rx
                msg.y = self.ry
                msg.theta = self.ryaw  # Match current yaw exactly so there is 0 heading error
                self.goal_pub.publish(msg)
                
                rospy.loginfo_throttle(2.0, '[visual_pose_localiser] Reached end of route! Holding position.')
                rate.sleep()
                continue

            self._advance_open_loop()

            yaw_corr, dx, corr = self._visual_update()

            # GEOMETRIC GOAL SEARCH: Lookahead steps is often too short (e.g. 0.05m).
            # This causes massive numeric explosion in the Pure Pursuit controller's 
            # curvature calculation (L^2 denominator). We physically search forward.
            search_idx = self.idx
            dist_accum = 0.0
            
            while dist_accum < 0.35:  # ensure lookahead vector is geometrically stable
                next_idx = (search_idx + 1) % len(self.samples) if self.loop_route else min(search_idx + 1, len(self.samples) - 1)
                if next_idx == search_idx:
                    break
                    
                pt_dx = self.samples[next_idx][0] - self.samples[search_idx][0]
                pt_dy = self.samples[next_idx][1] - self.samples[search_idx][1]
                dist_accum += math.hypot(pt_dx, pt_dy)
                search_idx = next_idx
                
            goal_idx = search_idx

            # Extract the raw geometrical X, Y coordinates from the mapped waypoints
            tx, ty = self.samples[self.idx][0], self.samples[self.idx][1]
            gx, gy = self.samples[goal_idx][0], self.samples[goal_idx][1]
                          
            # CRITICAL FIX: The recorded map orientations (`tth`, `gth`) are completely corrupted because 
            # the JetRacer skid-steered violently during the teaching phase! 
            # We strictly bypass them and calculate the 'True Track Heading' from the literal (x,y) curve geometry!
            
            # 1. Compute True Taught Path Heading (true_tth) from adjacent waypoint deltas
            next_idx = (self.idx + 1) % len(self.samples) if self.loop_route else min(self.idx + 1, len(self.samples) - 1)
            t_dx = self.samples[next_idx][0] - tx
            t_dy = self.samples[next_idx][1] - ty
            true_tth = math.atan2(t_dy, t_dx) if (abs(t_dx) > 0.001 or abs(t_dy) > 0.001) else self.samples[self.idx][2]
            
            # 2. Extract the relative forward and lateral shifts to the future goal point
            vx = gx - tx
            vy = gy - ty
            local_x = vx * math.cos(true_tth) + vy * math.sin(true_tth)
            local_y = -vx * math.sin(true_tth) + vy * math.cos(true_tth)
            
            # 3. Project this geometrically flawless arc out from the robot's visually-corrected nose
            # We explicitly enforce a strong heading gain of 1.0 here for pure coordinate rotation!
            true_visual_yaw = _wrap(self.ryaw + (dx / float(self.resize_w)) * self.image_fov_rad * self.heading_sign)
            
            reactive_gx = self.rx + local_x * math.cos(true_visual_yaw) - local_y * math.sin(true_visual_yaw)
            reactive_gy = self.ry + local_x * math.sin(true_visual_yaw) + local_y * math.cos(true_visual_yaw)
            
            # Numerically stabilize terminal parking string natively to prevent endless twitching
            dist_to_goal = math.hypot(reactive_gx - self.rx, reactive_gy - self.ry)
            if dist_to_goal < 0.20 and not self._finished:
                scale = 0.20 / max(dist_to_goal, 0.01)
                reactive_gx = self.rx + (reactive_gx - self.rx) * scale
                reactive_gy = self.ry + (reactive_gy - self.ry) * scale
            
            # 4. Map the final coordinates directly
            msg = Pose2D()
            msg.x = reactive_gx
            msg.y = reactive_gy
            # For controller blend stabilization at the end
            msg.theta = true_visual_yaw
            
            # Numerically stabilize terminal parking: prevent coasting past finish
            if not self.loop_route and self.idx >= len(self.samples) - 2:
                # We have reached the absolute final visual waypoint
                self._finished = True
                rospy.loginfo('[visual_pose_localiser] Mission Complete! Latching parking state.')
            self.goal_pub.publish(msg)

            # ── Debug topic: JSON-formatted correction diagnostics ──────────
            lk_speed = (
                self._lk_tracker.estimate_flow_speed()
                if self._lk_tracker is not None else 0.0
            )
            debug_payload = json.dumps({
                'stamp':          self.current_img_stamp.to_sec()
                                  if self.current_img_stamp else 0.0,
                'keyframe_idx':   self.idx,
                'goal_idx':       goal_idx,
                'source':         self._last_correction_source,
                'lk_confidence':  round(self._last_lk_confidence, 4),
                'lk_flow_speed':  round(lk_speed, 4),
                'ncc_score':      round(corr, 4),
                'dx_px':          round(dx, 3),
                'yaw_corr_deg':   round(math.degrees(yaw_corr), 3),
            }, sort_keys=True)
            self.debug_pub.publish(String(data=debug_payload))

            rospy.loginfo_throttle(
                1.0,
                '[visual_pose_localiser] idx=%d goal=%d corr=%.3f '
                'dx=%.2fpx yaw_corr=%.2fdeg src=%s lk_conf=%.2f lk_spd=%.2f',
                self.idx, goal_idx, corr, dx,
                math.degrees(yaw_corr),
                self._last_correction_source,
                self._last_lk_confidence,
                lk_speed,
            )

            rate.sleep()


if __name__ == '__main__':
    try:
        VisualPoseLocaliser().run()
    except rospy.ROSInterruptException:
        pass
