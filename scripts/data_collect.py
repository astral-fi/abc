#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import json
import math

import rospy
import tf.transformations as tft

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Image
from std_msgs.msg import String


class DataCollect(object):
    """Collect waypoint trigger events from odometry and publish them for saver."""

    def __init__(self):
        rospy.init_node('data_collect')

        self.distance_threshold = float(rospy.get_param('~distance_threshold', 0.3))
        self.angle_threshold_deg = float(rospy.get_param('~angle_threshold_deg', 12.0))
        self.use_angle_trigger = bool(rospy.get_param('~use_angle_trigger', False))
        self.waypoint_topic = rospy.get_param('~waypoint_topic', '/teach_repeat/waypoint')

        self._angle_threshold = math.radians(self.angle_threshold_deg)
        self._last_pose = None
        self._latest_image_stamp = None
        self._seq = 0

        self.pub = rospy.Publisher(self.waypoint_topic, String, queue_size=20)

        rospy.Subscriber('odom', PoseWithCovarianceStamped, self._odom_cb, queue_size=10)
        rospy.Subscriber('image', Image, self._image_cb, queue_size=10)

        rospy.loginfo('[data_collect] Started. topic=%s dist=%.3f m angle_trigger=%s angle=%.1f deg',
                  self.waypoint_topic, self.distance_threshold,
                  str(self.use_angle_trigger), self.angle_threshold_deg)

    @staticmethod
    def _yaw_from_quat(q):
        _, _, yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
        return yaw

    @staticmethod
    def _wrap(a):
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    def _image_cb(self, msg):
        self._latest_image_stamp = msg.header.stamp

    def _emit_waypoint(self, msg):
        pose = msg.pose.pose
        payload = {
            'seq': int(self._seq),
            'stamp': msg.header.stamp.to_sec(),
            'position': {
                'x': float(pose.position.x),
                'y': float(pose.position.y),
                'z': float(pose.position.z),
            },
            'orientation': {
                'x': float(pose.orientation.x),
                'y': float(pose.orientation.y),
                'z': float(pose.orientation.z),
                'w': float(pose.orientation.w),
            },
            'has_image': bool(self._latest_image_stamp is not None),
        }

        self.pub.publish(String(data=json.dumps(payload, sort_keys=True)))
        self._seq += 1

    def _odom_cb(self, msg):
        pose = msg.pose
        x = float(pose.pose.position.x)
        y = float(pose.pose.position.y)
        yaw = self._yaw_from_quat(pose.pose.orientation)

        if self._last_pose is None:
            self._last_pose = (x, y, yaw)
            self._emit_waypoint(msg)
            return

        lx, ly, lyaw = self._last_pose
        dist = math.sqrt((x - lx) ** 2 + (y - ly) ** 2)
        dyaw = abs(self._wrap(yaw - lyaw))

        if dist >= self.distance_threshold or (
            self.use_angle_trigger and dyaw >= self._angle_threshold):
            self._last_pose = (x, y, yaw)
            self._emit_waypoint(msg)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        DataCollect().run()
    except rospy.ROSInterruptException:
        pass
