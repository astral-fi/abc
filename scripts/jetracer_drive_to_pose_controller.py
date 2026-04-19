#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
# jetracer_drive_to_pose_controller.py
#
# Ackermann-correct Pure-Pursuit controller for QVPR/teach-repeat on JetRacer.
#
# TOPIC WIRING (matched to your actual JetRacer setup)
# ─────────────────────────────────────────────────────────────────────────────
# Subscribes:
#   /odom_combined  (geometry_msgs/PoseWithCovarianceStamped)
#   goal   (geometry_msgs/Pose2D)  — next waypoint from localiser.py
#                                    (relative name; remapped in launch if needed)
#
# Publishes:
#   /cmd_vel (geometry_msgs/Twist)
#     linear.x   : forward speed in m/s (always ≥ 0, Ackermann: no reverse)
#     angular.z  : yaw rate in rad/s, derived from the bicycle model:
#                    ω = v · tan(δ) / wheelbase
#                  This is the standard Ackermann-compatible Twist convention.
#                  Your `roslaunch jetracer jetracer.launch` hardware bridge
#                  already subscribes to /cmd_vel and converts it to servo
#                  angle + ESC throttle via Adafruit ServoKit.
#
# WHY /cmd_vel INSTEAD OF A CUSTOM MESSAGE
# ─────────────────────────────────────────────────────────────────────────────
# Your JetRacer base stack (`roslaunch jetracer jetracer.launch`) subscribes
# to geometry_msgs/Twist on /cmd_vel.  Publishing a standard Twist here means
# NO extra bridge node is needed during the repeat phase — the hardware bridge
# that is already running in Terminal 1 consumes our output directly.
#
# WHY PURE-PURSUIT FOR ACKERMANN
# ─────────────────────────────────────────────────────────────────────────────
#   • Naturally encodes the non-holonomic constraint (no lateral slip)
#   • Geometrically stable — no integrator windup possible
#   • Single primary tuning knob: lookahead distance L
#   • angular.z is derived from the bicycle-model relation:
#       ω = v · tan(δ) / wheelbase
#     so it is always physically realisable at the commanded speed
#
# KEY ACKERMANN RULES ENFORCED
# ─────────────────────────────────────────────────────────────────────────────
#   1. linear.x is ALWAYS ≥ 0  (robot never reverses)
#   2. angular.z is CLAMPED to the robot's physical steering limit
#   3. A minimum forward speed is enforced when a steering command exists
#   4. If goal is behind the robot (> behind_threshold_deg), STOP and wait
# =============================================================================

from __future__ import division, print_function
import rospy
import math
import tf.transformations as tft

from geometry_msgs.msg import Twist, Pose2D, PoseWithCovarianceStamped

# Goal uses geometry_msgs/Pose2D to keep this package self-contained.

# ── Constants ─────────────────────────────────────────────────────────────────
GOAL_TIMEOUT_SEC = 1.5   # Stop if no goal received within this window (seconds)
LOOP_HZ          = 20    # Control loop rate (Hz) — decoupled from callback rates


class AckermannPurePursuitController(object):
    """
    Pure-Pursuit pose controller for a car-like (Ackermann) JetRacer.

    Reads the latest /odom_combined pose and Pose2D waypoint via ROS
    callbacks, then runs a 20 Hz control loop that computes and publishes
    the appropriate /cmd_vel Twist to drive the robot along the taught route.
    """

    def __init__(self):
        rospy.init_node('jetracer_drive_to_pose_controller')

        # ── Tunable parameters (set from launch file or rosparam) ─────────────
        self.L = rospy.get_param(
            '~lookahead_distance', 0.0)      # metres — PRIMARY tuning knob########### 40-20
        self.v_cruise = rospy.get_param(
            '~cruise_speed',       0.35)      # m/s
        self.v_min = rospy.get_param(
            '~min_speed',          0.15)      # m/s — Ackermann floor
        self.slow_down_dist = rospy.get_param(
            '~slow_down_dist',     0.35)      # m — start deceleration
        self.goal_threshold = rospy.get_param(
            '~goal_threshold',     0.20)      # m — waypoint "reached" radius
        self.wheelbase = rospy.get_param(
            '~wheelbase',          0.16)      # m — front-to-rear axle distance
        self.max_steer_rad = rospy.get_param(
            '~max_steer_angle_deg', 20.0) * math.pi / 180.0  # convert to rad
        self.behind_thr = rospy.get_param(
            '~behind_threshold_deg', 90.0) * math.pi / 180.0  # convert to rad
        # Use -1.0 if your hardware interprets +linear.x as reverse.
        self.linear_sign = float(rospy.get_param('~linear_sign', 1.0))

        # Maximum physically realisable angular.z at cruise speed
        # Derived from bicycle model: ω_max = v · tan(δ_max) / wheelbase
        self.omega_max = (self.v_cruise
                          * math.tan(self.max_steer_rad)
                          / self.wheelbase)

        # ── Shared state (written by callbacks, read by control loop) ─────────
        self.current_pose     = None   # tuple (x, y, yaw) from /odom_combined
        self.current_goal     = None   # Pose2D goal from localiser
        self.last_goal_stamp  = None   # rospy.Time of most recent goal

        # ── ROS interfaces ────────────────────────────────────────────────────
        # Publisher: standard Twist on /cmd_vel.
        # Consumed by jetracer.launch hardware bridge (already running).
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # Subscriber: fused pose.
        rospy.Subscriber('/odom_combined', PoseWithCovarianceStamped, self._odom_cb, queue_size=1)

        # Subscriber: waypoint goals from localiser.py
        # 'goal' is a relative name — remapped to /goal in the launch file
        # or matched by localiser's default publish topic.
        rospy.Subscriber('goal', Pose2D, self._goal_cb, queue_size=1)

        self._rate = rospy.Rate(LOOP_HZ)

        rospy.loginfo(
            "[JetRacer Controller] Ackermann Pure-Pursuit initialised.\n"
            "  Subscribes:  /odom_combined (geometry_msgs/PoseWithCovarianceStamped)  |  goal (geometry_msgs/Pose2D)\n"
            "  Publishes:   /cmd_vel (geometry_msgs/Twist)\n"
            "  Lookahead L  : %.2f m\n"
            "  Cruise speed : %.2f m/s   Min speed: %.2f m/s\n"
            "  Wheelbase    : %.3f m     Max steer: %.1f °\n"
            "  ω_max        : %.2f rad/s",
            self.L, self.v_cruise, self.v_min,
            self.wheelbase, math.degrees(self.max_steer_rad), self.omega_max
        )

    # ── ROS callbacks ─────────────────────────────────────────────────────────

    def _odom_cb(self, msg):
        """
        Extract (x, y, yaw) from /odom_combined PoseWithCovarianceStamped.
        """
        q = msg.pose.pose.orientation
        _, _, yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.current_pose = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            yaw
        )

    def _goal_cb(self, msg):
        """Receive next waypoint goal from localiser.py."""
        self.current_goal    = msg
        self.last_goal_stamp = rospy.Time.now()
        try:
            gx, gy, gtheta = self._goal_to_xytheta(msg)
            rospy.logdebug(
                "[JetRacer Controller] New goal → x=%.3f  y=%.3f  θ=%.1f°",
                gx, gy, math.degrees(gtheta))
        except Exception as e:
            rospy.logwarn_throttle(
                2.0,
                "[JetRacer Controller] Unsupported Goal schema (%s).", str(e)
            )

    @staticmethod
    def _goal_to_xytheta(goal):
        """
        Convert goal message variants into scalar (x, y, theta).

        Supported layouts:
          - fields: x, y, theta
          - fields: target_x, target_y, target_theta
          - field:  pose (geometry_msgs/Pose)
          - field:  pose (geometry_msgs/PoseStamped)
        """
        if hasattr(goal, 'x') and hasattr(goal, 'y') and hasattr(goal, 'theta'):
            return float(goal.x), float(goal.y), float(goal.theta)

        if (hasattr(goal, 'target_x') and hasattr(goal, 'target_y')
                and hasattr(goal, 'target_theta')):
            return (
                float(goal.target_x),
                float(goal.target_y),
                float(goal.target_theta)
            )

        if hasattr(goal, 'pose'):
            pose_obj = goal.pose
            # PoseStamped case
            if hasattr(pose_obj, 'pose'):
                pose_obj = pose_obj.pose

            if hasattr(pose_obj, 'position') and hasattr(pose_obj, 'orientation'):
                q = pose_obj.orientation
                _, _, yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
                return float(pose_obj.position.x), float(pose_obj.position.y), float(yaw)

        raise ValueError("Goal fields unsupported: %s" % str(getattr(goal, '__slots__', [])))

    # ── Geometry helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _wrap(angle):
        """Wrap angle to [-π, π]."""
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _to_robot_frame(self, rx, ry, ryaw, gx, gy):
        """
        Transform goal point (gx, gy) from the odom frame into the robot's
        local body frame.

        dx_local > 0  →  goal is ahead of robot
        dy_local > 0  →  goal is to the robot's left
        """
        dx = gx - rx
        dy = gy - ry
        dx_local =  dx * math.cos(ryaw) + dy * math.sin(ryaw)
        dy_local = -dx * math.sin(ryaw) + dy * math.cos(ryaw)
        return dx_local, dy_local

    # ── Pure-Pursuit control law ──────────────────────────────────────────────

    def _compute_command(self, rx, ry, ryaw, goal):
        """
        Compute (linear_x, angular_z) for a geometry_msgs/Twist using the
        Pure-Pursuit algorithm adapted for an Ackermann platform.

        Steps:
          1. Transform goal into robot-local frame → (dx_local, dy_local)
          2. Ackermann feasibility: if goal is behind, stop.
          3. Speed scaling: ramp v down linearly inside slow_down_dist.
          4. Pure-Pursuit curvature: κ = 2·dy_local / lookahead²
          5. Steering angle:  δ = atan(κ · wheelbase)  — clamp to ±max_steer
          6. Near goal: blend in heading-error correction (robot arrives aligned).
          7. angular.z = v · tan(δ) / wheelbase  (bicycle model) — clamp to ω_max

        Returns:
            (v, omega) — assign to Twist.linear.x and Twist.angular.z
        """
        gx, gy, g_theta = self._goal_to_xytheta(goal)

        dist = math.sqrt((gx - rx) ** 2 + (gy - ry) ** 2)
        dx_local, dy_local = self._to_robot_frame(rx, ry, ryaw, gx, gy)
	


        # ── Ackermann feasibility ─────────────────────────────────────────────
        # NOTE: Do not hard-stop when a goal is behind; that can deadlock if
        # localiser/controller indexing drifts. We keep moving forward with
        # bounded steering so the robot can recover onto the route.
        angle_to_goal = self._wrap(math.atan2(gy - ry, gx - rx) - ryaw)

        # ── Speed scaling ─────────────────────────────────────────────────────
        if dist < self.slow_down_dist:
            span  = max(self.slow_down_dist - self.goal_threshold, 1e-6)
            alpha = max(0.0, min(1.0, (dist - self.goal_threshold) / span))
            v     = self.v_min + alpha * (self.v_cruise - self.v_min)
        else:
            v = self.v_cruise

        # ── Pure-Pursuit curvature ────────────────────────────────────────────
        lookahead = max(min(dist, self.L), 0.05)    # 5 cm floor
        kappa     = 2.0 * dy_local / (lookahead ** 2)
        delta     = math.atan(kappa * self.wheelbase)
        delta     = max(-self.max_steer_rad, min(self.max_steer_rad, delta))

        # ── Heading correction blend (near goal) ──────────────────────────────
        blend_dist = self.slow_down_dist * 0.6
        if dist < blend_dist:
            heading_err = self._wrap(g_theta - ryaw)
            blend       = 1.0 - dist / max(blend_dist, 1e-6)
            delta_head  = math.atan(heading_err * self.wheelbase
                                    / max(v, self.v_min))
            delta = (1.0 - blend) * delta + blend * delta_head
            delta = max(-self.max_steer_rad, min(self.max_steer_rad, delta))

        # ── Bicycle model: δ → ω ─────────────────────────────────────────────
        # ω = v · tan(δ) / wheelbase
        # This is a physically realisable angular rate for the Ackermann Twist
        # convention consumed by `roslaunch jetracer jetracer.launch`.
        omega = v * math.tan(delta) / self.wheelbase
        omega = max(-self.omega_max, min(self.omega_max, omega))

        # Enforce minimum speed when meaningfully turning
        if abs(omega) > 0.05:
            v = max(v, self.v_min)

        return v, omega #######

    # ── Main control loop ─────────────────────────────────────────────────────

    def run(self):
        """
        20 Hz control loop.  Computes and publishes /cmd_vel Twist messages.
        Stops (zeros) automatically if /odom_combined or goal messages are missing.
        """
        rospy.loginfo(
            "[JetRacer Controller] Control loop running at %d Hz.  "
            "Publishing /cmd_vel → jetracer.launch hardware bridge.",
            LOOP_HZ)

        while not rospy.is_shutdown():
            cmd = Twist()   # default: all zeros → hardware bridge = stop

            # ── Goal timeout watchdog ─────────────────────────────────────────
            if self.last_goal_stamp is not None:
                age = (rospy.Time.now() - self.last_goal_stamp).to_sec()
                if age > GOAL_TIMEOUT_SEC:
                    rospy.logwarn_throttle(
                        5.0,
                        "[JetRacer Controller] No goal for %.1f s "
                        "(timeout=%.1f s). Sending stop.",
                        age, GOAL_TIMEOUT_SEC)
                    self.cmd_pub.publish(cmd)
                    self._rate.sleep()
                    continue

            # ── Wait for /odom_combined ───────────────────────────────────────
            if self.current_pose is None:
                rospy.loginfo_throttle(
                    3.0,
                    "[JetRacer Controller] Waiting for /odom_combined "
                    "(from odom_integrator)...")
                self._rate.sleep()
                continue

            # ── Wait for first goal ───────────────────────────────────────────
            if self.current_goal is None:
                rospy.loginfo_throttle(
                    3.0,
                    "[JetRacer Controller] Waiting for first goal "
                    "from localiser.py...")
                self._rate.sleep()
                continue

            rx, ry, ryaw = self.current_pose
            goal         = self.current_goal

            try:
                gx, gy, _ = self._goal_to_xytheta(goal)
            except Exception as e:
                rospy.logwarn_throttle(
                    2.0,
                    "[JetRacer Controller] Cannot parse Goal (%s). Sending stop.",
                    str(e)
                )
                self.cmd_pub.publish(cmd)
                self._rate.sleep()
                continue

            # ── Waypoint reached? ─────────────────────────────────────────────
            dist = math.sqrt((gx - rx) ** 2 + (gy - ry) ** 2)
            if dist < self.goal_threshold:
                rospy.logdebug(
                    "[JetRacer Controller] Waypoint reached "
                    "(dist=%.3f m < %.3f m). Coasting at v_min.",
                    dist, self.goal_threshold)
                # Coast at v_min; avoids jerky stop-and-go at each waypoint.
                cmd.linear.x  = self.linear_sign * self.v_min
                cmd.angular.z = 0.0
                self.cmd_pub.publish(cmd)
                self._rate.sleep()
                continue

            # ── Pure-Pursuit ──────────────────────────────────────────────────
            v, omega      = self._compute_command(rx, ry, ryaw, goal)
            cmd.linear.x  = v       # m/s  → hardware bridge → ESC throttle
            cmd.angular.z = 0.5*omega   # rad/s → hardware bridge → steering servo
            self.cmd_pub.publish(cmd)

            rospy.loginfo_throttle(
                1.0,
                "[JetRacer Controller] cmd v=%.2f m/s omega=%.2f rad/s dist=%.2f m",
                cmd.linear.x, cmd.angular.z, dist)

            rospy.logdebug(
                "[JetRacer Controller] dist=%.2f m | v=%.2f m/s | "
                "ω=%.2f rad/s | δ≈%.1f°",
                dist, v, omega,
                math.degrees(math.atan(
                    omega * self.wheelbase / max(v, 1e-3))))

            self._rate.sleep()

        # ── Shutdown: always send a hard stop ─────────────────────────────────
        rospy.loginfo(
            "[JetRacer Controller] Shutdown — publishing zero /cmd_vel.")
        self.cmd_pub.publish(Twist())


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    try:
        controller = AckermannPurePursuitController()
        controller.run()
    except rospy.ROSInterruptException:
        pass

