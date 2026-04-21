#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
odom_phase_recorder.py
======================
Lightweight ROS node that records /odom_combined
(geometry_msgs/PoseWithCovarianceStamped) to a CSV file
for a single named phase.

Run once per phase, Ctrl-C to stop and save.

Usage
-----
  # Phase 1 — teach while driving the route manually
  rosrun abc odom_phase_recorder.py _phase:=teach _out_dir:=~/jetracer/analysis

  # Phase 2 — repeat run with camera COVERED (dead-reckoning only)
  rosrun abc odom_phase_recorder.py _phase:=blind _out_dir:=~/jetracer/analysis

  # Phase 3 — repeat run with camera UNCOVERED (full visual correction)
  rosrun abc odom_phase_recorder.py _phase:=visual _out_dir:=~/jetracer/analysis

CSV columns
-----------
  elapsed_s  — seconds since recording started (float)
  x          — odom x position (metres)
  y          — odom y position (metres)
  yaw        — heading in radians [-π, π]
  vx         — forward velocity (m/s) from twist field
  omega      — angular velocity (rad/s) from twist field

Python 2/3 dual-compatible.
"""

from __future__ import division, print_function

import csv
import os
import sys

import rospy
import tf.transformations as tft
from geometry_msgs.msg import PoseWithCovarianceStamped


VALID_PHASES = ('teach', 'blind', 'visual')


class OdomPhaseRecorder(object):
    """
    Subscribes to the odom topic and flushes rows to a CSV file.
    Automatically saves on ROS shutdown (Ctrl-C).
    """

    def __init__(self):
        rospy.init_node('odom_phase_recorder', anonymous=True)

        # ── Parameters ────────────────────────────────────────────────────────
        self.phase   = rospy.get_param('~phase',       'teach')
        out_dir      = os.path.expanduser(
            rospy.get_param('~out_dir',  '~/jetracer/analysis'))
        odom_topic   = rospy.get_param('~odom_topic',  '/odom_combined')
        # Session tag added to CSV name so multiple runs don't overwrite each other
        session      = rospy.get_param('~session',     '')

        # ── Validation ────────────────────────────────────────────────────────
        if self.phase not in VALID_PHASES:
            rospy.logfatal(
                '[odom_phase_recorder] Unknown phase "%s". Choose from: %s',
                self.phase, str(VALID_PHASES))
            raise ValueError('Unknown phase: %s' % self.phase)

        # ── Output file ───────────────────────────────────────────────────────
        try:
            os.makedirs(out_dir)
        except OSError:
            pass   # already exists — Python 2/3 compat alternative to exist_ok

        fname = ('%s_%s.csv' % (self.phase, session)) if session \
                else ('%s.csv' % self.phase)
        out_path = os.path.join(out_dir, fname)

        self._fh     = open(out_path, 'w')
        self._writer = csv.writer(self._fh, lineterminator='\n')
        self._writer.writerow(['elapsed_s', 'x', 'y', 'yaw', 'vx', 'omega'])

        self._count = 0
        self._t0    = None
        self._out_path = out_path

        # ── ROS plumbing ──────────────────────────────────────────────────────
        rospy.Subscriber(odom_topic, PoseWithCovarianceStamped,
                         self._odom_cb, queue_size=20)

        rospy.on_shutdown(self._on_shutdown)

        rospy.loginfo(
            '[odom_phase_recorder] ─────────────────────────────────────')
        rospy.loginfo(
            '[odom_phase_recorder] Phase    : %s', self.phase.upper())
        rospy.loginfo(
            '[odom_phase_recorder] Saving to: %s', out_path)
        rospy.loginfo(
            '[odom_phase_recorder] Listening: %s', odom_topic)
        rospy.loginfo(
            '[odom_phase_recorder] Press Ctrl-C to stop and save.')

    # ── Callback ──────────────────────────────────────────────────────────────

    def _odom_cb(self, msg):
        stamp = msg.header.stamp.to_sec()
        if self._t0 is None:
            self._t0 = stamp

        q   = msg.pose.pose.orientation
        _, _, yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])

        x  = float(msg.pose.pose.position.x)
        y  = float(msg.pose.pose.position.y)
        vx = float(msg.twist.twist.linear.x)
        om = float(msg.twist.twist.angular.z)

        self._writer.writerow([
            round(stamp - self._t0, 5),
            round(x,   5),
            round(y,   5),
            round(yaw, 6),
            round(vx,  5),
            round(om,  6),
        ])
        self._count += 1

        if self._count % 50 == 0:
            rospy.loginfo_throttle(
                5.0, '[odom_phase_recorder] %s — %d samples recorded.',
                self.phase, self._count)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def _on_shutdown(self):
        self._fh.flush()
        self._fh.close()
        rospy.loginfo(
            '[odom_phase_recorder] SAVED %d samples → %s',
            self._count, self._out_path)

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        OdomPhaseRecorder().run()
    except rospy.ROSInterruptException:
        pass
