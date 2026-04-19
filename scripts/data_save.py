#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import datetime
import json
import math
import os

import cv2
import rospy
import tf.transformations as tft
from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Image


class DataSave(object):
    """Save poses and images into the original teach-run folder structure."""

    def __init__(self):
        rospy.init_node('data_save')

        self.save_dir = os.path.expanduser(rospy.get_param('~save_dir', '~/jetracer/teach_runs'))
        self.timestamp_folder = bool(rospy.get_param('~timestamp_folder', True))
        self.image_topic = rospy.get_param('~image_topic', '/csi_cam_0/image_raw')
        self.odom_topic = rospy.get_param('~odom_topic', '/odom_combined')
        self.distance_threshold = float(rospy.get_param('~distance_threshold', 0.30))
        self.min_linear_speed = float(rospy.get_param('~min_linear_speed', 0.03))
        self.angle_threshold_deg = float(rospy.get_param('~angle_threshold_deg', 12.0))
        self.capture_on_angle = bool(rospy.get_param('~capture_on_angle', False))

        self._angle_threshold = math.radians(self.angle_threshold_deg)
        self.bridge = CvBridge()
        self._latest_img = None
        self._last_pose = None
        self._last_saved_pose = None
        self._idx = 0

        self.run_dir = self._prepare_run_dir()
        self.full_dir = os.path.join(self.run_dir, 'full')
        os.makedirs(self.full_dir)

        rospy.Subscriber(self.image_topic, Image, self._image_cb, queue_size=10)
        rospy.Subscriber(self.odom_topic, PoseWithCovarianceStamped, self._odom_cb, queue_size=10)

        rospy.loginfo('[data_save] Saving teach run to: %s (distance_threshold=%.1f m)',
                      self.run_dir, self.distance_threshold)

    def _prepare_run_dir(self):
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        if self.timestamp_folder:
            name = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        else:
            name = 'latest'

        run_dir = os.path.join(self.save_dir, name)
        suffix = 1
        while os.path.exists(run_dir):
            run_dir = os.path.join(self.save_dir, '%s_%02d' % (name, suffix))
            suffix += 1

        os.makedirs(run_dir)
        return run_dir

    @staticmethod
    def _yaw_from_quat(q):
        _, _, yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
        return yaw

    @staticmethod
    def _wrap(a):
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    def _image_cb(self, msg):
        try:
            self._latest_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logwarn_throttle(2.0, '[data_save] image conversion failed: %s', str(e))

    def _odom_cb(self, msg):
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        theta = self._yaw_from_quat(msg.pose.pose.orientation)
        stamp = msg.header.stamp.to_sec()
        self._last_pose = (x, y, theta, stamp)

        print(self._last_pose)
        

        if self._latest_img is None:
            return

        if self._last_saved_pose is None:
            self._save_current()
            return

        lx, ly, lth, _ = self._last_saved_pose
        dist = math.hypot(x - lx, y - ly)
        dth_deg = abs(math.degrees(self._wrap(theta - lth)))

        if dist >= self.distance_threshold or (
            self.capture_on_angle and dth_deg >= self.angle_threshold_deg):
            self._save_current()

    def _save_current(self):
        x, y, theta, stamp = self._last_pose

        try:
            img = self._latest_img
            image_name = 'frame_%06d.png' % self._idx
            image_path = os.path.join(self.full_dir, image_name)
            cv2.imwrite(image_path, img)

            pose_obj = {
                'position': {'x': float(x), 'y': float(y), 'z': 0.0},
                'orientation': dict(zip(
                    ['x', 'y', 'z', 'w'],
                    map(float, tft.quaternion_from_euler(0.0, 0.0, theta))
                )),
            }
            pose_path = os.path.join(self.run_dir, 'frame_%06d_pose.txt' % self._idx)
            with open(pose_path, 'w') as fh:
                fh.write(json.dumps(pose_obj, sort_keys=True))

            self._last_saved_pose = (x, y, theta, stamp)
            self._idx += 1
            rospy.loginfo_throttle(1.0, '[data_save] saved frames: %d', self._idx)
        except Exception as e:
            rospy.logwarn('[data_save] failed saving current frame: %s', str(e))

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        DataSave().run()
    except rospy.ROSInterruptException:
        pass
