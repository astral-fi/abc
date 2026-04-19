#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
# odom_integrator.py
#
# Converts /odom_raw (nav_msgs/Odometry) → /odom (nav_msgs/Odometry)
# with a continuously integrated 2-D pose estimate.
#
# WHY THIS NODE EXISTS
# ─────────────────────────────────────────────────────────────────────────────
# The JetRacer base launch (`roslaunch jetracer jetracer.launch`) publishes
# wheel/encoder data as /odom_raw.  That message carries valid velocity
# information in its twist field (linear.x and angular.z) but the pose field
# is typically NOT integrated — it either stays at the origin or contains
# only a raw encoder tick count scaled to metres.
#
# QVPR/teach-repeat's data_collect.py and localiser.py subscribe to /odom and
# REQUIRE an integrated pose (x, y, yaw) in that message.  This node provides
# it by dead-reckoning from the velocity measurements using a standard
# Euler-integration unicycle model.
#
# INTEGRATION MODEL (unicycle / differential-drive approximation)
# ─────────────────────────────────────────────────────────────────────────────
# Given twist (v, ω) measured over interval dt:
#
#   x   += v * cos(yaw) * dt
#   y   += v * sin(yaw) * dt
#   yaw += ω * dt
#
# This is the forward-Euler unicycle model.  It accumulates drift on long
# runs (every odometry system does without external corrections), but it is
# perfectly adequate for the ~10–50 m indoor routes teach-repeat targets.
#
# INPUT ASSUMPTIONS
# ─────────────────────────────────────────────────────────────────────────────
# /odom_raw must have:
#   twist.twist.linear.x   — forward velocity in m/s (positive = forward)
#   twist.twist.angular.z  — yaw rate in rad/s (positive = CCW / left turn)
#
# If your /odom_raw actually carries a good integrated pose in its pose field,
# set the ROS parameter ~use_raw_pose to true and this node will pass it
# through unchanged (with the frame renaming only).
#
# SUBSCRIBES:  /odom_raw  (nav_msgs/Odometry)
# PUBLISHES:   /odom      (nav_msgs/Odometry)
# TF:          broadcasts odom → base_link (optional, enabled by default)
# =============================================================================

from __future__ import division, print_function

import math
import rospy
import tf

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, TransformStamped


class OdomIntegrator(object):
    """
    Integrates /odom_raw twist into a 2-D pose and republishes as /odom.
    """

    def __init__(self):
        rospy.init_node('odom_integrator')

        # ── Parameters ────────────────────────────────────────────────────────
        # If True: trust the pose field of /odom_raw directly (pass-through).
        # Set True only if your jetracer driver actually integrates its own pose.
        self.use_raw_pose = rospy.get_param('~use_raw_pose', False)

        # Frame IDs for the published /odom message and TF broadcast.
        self.odom_frame  = rospy.get_param('~odom_frame',   'odom')
        self.base_frame  = rospy.get_param('~base_frame',   'base_link')

        # Whether to broadcast the TF odom → base_link transform.
        # Set False if another node (e.g. robot_pose_ekf) already does this.
        self.publish_tf  = rospy.get_param('~publish_tf',   True)

        # ── State ─────────────────────────────────────────────────────────────
        self.x    = 0.0
        self.y    = 0.0
        self.yaw  = 0.0
        self.last_time = None   # rospy.Time of the previous message

        # ── ROS interfaces ────────────────────────────────────────────────────
        self.pub = rospy.Publisher('/odom', Odometry, queue_size=10)

        if self.publish_tf:
            self.tf_br = tf.TransformBroadcaster()

        rospy.Subscriber('/odom_raw', Odometry, self._odom_raw_cb, queue_size=10)

        rospy.loginfo(
            "[odom_integrator] Started.\n"
            "  /odom_raw  →  /odom\n"
            "  use_raw_pose : %s\n"
            "  publish_tf   : %s  (%s → %s)",
            self.use_raw_pose, self.publish_tf,
            self.odom_frame, self.base_frame
        )

    # ── Helper: build quaternion from yaw ─────────────────────────────────────

    @staticmethod
    def _yaw_to_quat(yaw):
        """Return a geometry_msgs/Quaternion for a pure yaw rotation."""
        q = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw)
        return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

    # ── Main callback ─────────────────────────────────────────────────────────

    def _odom_raw_cb(self, raw_msg):
        """
        Receive /odom_raw, integrate the twist if needed, publish /odom.
        """
        now = raw_msg.header.stamp

        # ── Pose source: integrate or pass through ────────────────────────────
        if self.use_raw_pose:
            # Trust the pose in /odom_raw directly.
            q = raw_msg.pose.pose.orientation
            _, _, self.yaw = tf.transformations.euler_from_quaternion(
                [q.x, q.y, q.z, q.w])
            self.x = raw_msg.pose.pose.position.x
            self.y = raw_msg.pose.pose.position.y

        else:
            # Dead-reckoning integration from twist.
            if self.last_time is None:
                # First message — just initialise the clock, publish nothing yet.
                self.last_time = now
                return

            dt = (now - self.last_time).to_sec()
            self.last_time = now

            # Guard against negative or unreasonably large dt
            # (e.g. a clock jump on startup or after pause).
            if dt <= 0.0 or dt > 0.5:
                rospy.logwarn_throttle(
                    5.0,
                    "[odom_integrator] Skipping dt=%.4f s (out of range 0–0.5 s).",
                    dt)
                return

            v   = raw_msg.twist.twist.linear.x
            omega = raw_msg.twist.twist.angular.z

            # Euler integration (unicycle model)
            self.x   += v * math.cos(self.yaw) * dt
            self.y   += v * math.sin(self.yaw) * dt
            self.yaw += omega * dt

            # Keep yaw in [-pi, pi] to avoid floating-point drift
            self.yaw = (self.yaw + math.pi) % (2.0 * math.pi) - math.pi

        # ── Build output Odometry message ─────────────────────────────────────
        odom_msg = Odometry()
        odom_msg.header.stamp    = now
        odom_msg.header.frame_id = self.odom_frame
        odom_msg.child_frame_id  = self.base_frame

        # Integrated pose
        odom_msg.pose.pose.position.x  = self.x
        odom_msg.pose.pose.position.y  = self.y
        odom_msg.pose.pose.position.z  = 0.0
        odom_msg.pose.pose.orientation = self._yaw_to_quat(self.yaw)

        # Pass twist through unchanged — teach-repeat may use velocity too
        odom_msg.twist = raw_msg.twist

        # Simple diagonal covariance.
        # Position covariance grows with travel (wheel odometry drifts);
        # here we use a conservative fixed value appropriate for short runs.
        # Row-major 6×6 matrix: [x, y, z, roll, pitch, yaw]
        odom_msg.pose.covariance = [
            0.01,  0.0,   0.0,  0.0,  0.0,  0.0,
            0.0,   0.01,  0.0,  0.0,  0.0,  0.0,
            0.0,   0.0,   1e9,  0.0,  0.0,  0.0,   # z — planar robot
            0.0,   0.0,   0.0,  1e9,  0.0,  0.0,   # roll
            0.0,   0.0,   0.0,  0.0,  1e9,  0.0,   # pitch
            0.0,   0.0,   0.0,  0.0,  0.0,  0.05,  # yaw
        ]
        odom_msg.twist.covariance = [
            0.005, 0.0,  0.0,  0.0,  0.0,  0.0,
            0.0,   1e9,  0.0,  0.0,  0.0,  0.0,
            0.0,   0.0,  1e9,  0.0,  0.0,  0.0,
            0.0,   0.0,  0.0,  1e9,  0.0,  0.0,
            0.0,   0.0,  0.0,  0.0,  1e9,  0.0,
            0.0,   0.0,  0.0,  0.0,  0.0,  0.02,
        ]

        self.pub.publish(odom_msg)

        # ── TF broadcast ──────────────────────────────────────────────────────
        if self.publish_tf:
            q = tf.transformations.quaternion_from_euler(0.0, 0.0, self.yaw)
            self.tf_br.sendTransform(
                (self.x, self.y, 0.0),
                q,
                now,
                self.base_frame,
                self.odom_frame,
            )

        rospy.logdebug(
            "[odom_integrator] x=%.3f  y=%.3f  yaw=%.2f°",
            self.x, self.y, math.degrees(self.yaw))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        OdomIntegrator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
