#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import json
import math
import os

import rospy
import tf.transformations as tft

from geometry_msgs.msg import Pose2D, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry


def _wrap(angle):
	return (angle + math.pi) % (2.0 * math.pi) - math.pi


class PoseFileLocaliser(object):
	"""
	Goal publisher that follows stored *_pose.txt waypoints from a teach run.

	This is a robust fallback when the image-based localiser pipeline is
	unavailable or incompatible with a dataset/cache layout.
	"""

	def __init__(self):
		rospy.init_node('pose_file_localiser')

		self.run_dir_param = rospy.get_param(
			'~run_dir', os.path.expanduser('~/jetracer/teach_runs'))
		self.goal_radius = float(rospy.get_param('~goal_radius', 0.25))
		self.publish_hz = float(rospy.get_param('~publish_hz', 15.0))
		# IMPORTANT: keep this <= controller's behind_threshold_deg to avoid deadlock.
		# If controller stops at ~90°, localiser must skip at or before that.
		self.behind_skip_threshold_deg = float(
			rospy.get_param('~behind_skip_threshold_deg', 85.0))
		self.max_skip_ahead = int(rospy.get_param('~max_skip_ahead', 200))
		self.loop_route = bool(rospy.get_param('~loop_route', True))

		self.run_dir, self.pose_dir, self.pose_files, self.checked_dirs = self._resolve_pose_files(
			self.run_dir_param)
		self.poses = self._load_pose_files(self.pose_files)

		if not self.poses:
			rospy.logfatal(
				'[pose_file_localiser] No poses loaded. run_dir_param=%s resolved_run_dir=%s',
				self.run_dir_param,
				self.run_dir)
			rospy.logfatal(
				'[pose_file_localiser] Checked these directories for pose files: %s',
				', '.join(self.checked_dirs) if self.checked_dirs else '(none)')
			raise RuntimeError('No poses')

		self.goal_pub = rospy.Publisher('goal', Pose2D, queue_size=1)
		rospy.Subscriber('/odom_combined', PoseWithCovarianceStamped, self._odom_cb, queue_size=1)

		self.have_odom = False
		self.rx = 0.0
		self.ry = 0.0
		self.ryaw = 0.0
		self.idx = 0

		rospy.loginfo(
			'[pose_file_localiser] Loaded %d poses (files=%d) from %s (pose_dir=%s)',
			len(self.poses), len(self.pose_files), self.run_dir, self.pose_dir)
		rospy.loginfo(
			'[pose_file_localiser] behind_skip_threshold_deg=%.1f max_skip_ahead=%d goal_radius=%.2f loop_route=%s',
			self.behind_skip_threshold_deg, self.max_skip_ahead, self.goal_radius, str(self.loop_route))

	@staticmethod
	def _pose_globs(dirpath):
		patterns = [
			'*_pose.txt',
			'pose_*.txt',
			'*_pose.json',
			'pose_*.json',
		]
		files = []
		for pat in patterns:
			files.extend(glob.glob(os.path.join(dirpath, pat)))
		# Deterministic ordering and de-duplication.
		return sorted(set(files))

	def _pose_files_for_path(self, path):
		"""Return (pose_dir, pose_files, checked_dirs) for a concrete directory."""
		checked = []
		if not os.path.isdir(path):
			return None, [], checked

		candidates = [
			path,
			os.path.join(path, 'pose'),
			os.path.join(path, 'poses'),
		]

		for d in candidates:
			checked.append(d)
			files = self._pose_globs(d)
			if files:
				return d, files, checked

		# Heuristic: some datasets keep pose files in a sibling "pose" directory.
		parent = os.path.dirname(path.rstrip(os.sep))
		base = os.path.basename(path.rstrip(os.sep))
		parent_pose = os.path.join(parent, 'pose')
		parent_pose_run = os.path.join(parent_pose, base)

		for d in [parent_pose_run, parent_pose]:
			if not d:
				continue
			checked.append(d)
			if not os.path.isdir(d):
				continue
			files = self._pose_globs(d)
			# If we're in the parent pose dir, prefer files that start with the run folder name.
			if d == parent_pose and files:
				pref = [f for f in files if os.path.basename(f).startswith(base)]
				if pref:
					return d, sorted(pref), checked
			if files:
				return d, files, checked

		return None, [], checked

	def _resolve_pose_files(self, path):
		"""
		Resolve pose files from either:
		  - an explicit run dir, or
		  - a parent directory containing timestamped run subfolders.

		Returns: (resolved_run_dir, pose_dir, pose_files, checked_dirs)
		"""
		path = os.path.expanduser(path)
		checked_dirs = []

		if not os.path.isdir(path):
			return path, path, [], checked_dirs

		pose_dir, pose_files, checked = self._pose_files_for_path(path)
		checked_dirs.extend(checked)
		if pose_files:
			return path, pose_dir, pose_files, checked_dirs

		# Treat it as a parent directory: choose the latest subfolder that has pose files.
		candidates = []
		try:
			names = sorted(os.listdir(path))
		except Exception:
			names = []

		for name in names:
			sub = os.path.join(path, name)
			if not os.path.isdir(sub):
				continue
			pdir, pfiles, pchecked = self._pose_files_for_path(sub)
			checked_dirs.extend(pchecked)
			if pfiles:
				candidates.append((sub, pdir, pfiles))

		if not candidates:
			return path, path, [], checked_dirs

		candidates.sort(key=lambda t: t[0])
		chosen_sub, chosen_pose_dir, chosen_files = candidates[-1]
		rospy.logwarn(
			'[pose_file_localiser] No poses found directly under ~run_dir; using latest run: %s',
			chosen_sub)
		return chosen_sub, chosen_pose_dir, chosen_files, checked_dirs

	@staticmethod
	def _load_pose_files(files):
		poses = []
		for fpath in files:
			try:
				with open(fpath, 'r') as fh:
					data = json.loads(fh.read())

				x = float(data['position']['x'])
				y = float(data['position']['y'])
				qx = float(data['orientation']['x'])
				qy = float(data['orientation']['y'])
				qz = float(data['orientation']['z'])
				qw = float(data['orientation']['w'])

				_, _, yaw = tft.euler_from_quaternion([qx, qy, qz, qw])
				poses.append((x, y, yaw))
			except Exception as e:
				rospy.logwarn(
					'[pose_file_localiser] Skipping bad pose file %s (%s)',
					fpath, str(e))

		return poses

	def _odom_cb(self, msg):
		q = msg.pose.pose.orientation
		_, _, yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])

		self.rx = msg.pose.pose.position.x
		self.ry = msg.pose.pose.position.y
		self.ryaw = yaw
		self.have_odom = True

	def _nearest_index(self):
		best_i = 0
		best_d2 = 1e18

		for i, (x, y, _) in enumerate(self.poses):
			dx = x - self.rx
			dy = y - self.ry
			d2 = dx * dx + dy * dy
			if d2 < best_d2:
				best_d2 = d2
				best_i = i

		return best_i

	def _bearing_to_point(self, gx, gy):
		"""Signed bearing from robot heading to point in radians."""
		dy = gy - self.ry
		dx = gx - self.rx
		return _wrap(math.atan2(dy, dx) - self.ryaw)

	def _skip_ahead_if_goal_behind(self):
		"""Advance idx if current goal is behind the robot heading."""
		threshold = math.radians(self.behind_skip_threshold_deg)
		start_idx = self.idx
		best_idx = self.idx
		best_abs_bearing = float('inf')

		steps_limit = min(self.max_skip_ahead, len(self.poses))
		for steps in range(steps_limit):
			if self.loop_route:
				cand = (start_idx + steps) % len(self.poses)
			else:
				cand = start_idx + steps
				if cand >= len(self.poses):
					break

			gx, gy, _ = self.poses[cand]
			bearing = self._bearing_to_point(gx, gy)
			abs_bearing = abs(bearing)
			if abs_bearing < best_abs_bearing:
				best_abs_bearing = abs_bearing
				best_idx = cand
			if abs_bearing <= threshold:
				self.idx = cand
				break
		else:
			# No waypoint is comfortably in front of the robot; keep the best
			# candidate instead of getting stuck at the end of the list.
			self.idx = best_idx

		if self.idx != start_idx:
			rospy.logwarn_throttle(
				2.0,
				'[pose_file_localiser] Skipped ahead from idx %d to %d (goal was behind).',
				start_idx,
				self.idx)

	@staticmethod
	def _assign_goal_message(msg, gx, gy, gth):
		"""
		Fill goal message (geometry_msgs/Pose2D).
		"""
		if hasattr(msg, 'x') and hasattr(msg, 'y') and hasattr(msg, 'theta'):
			msg.x = gx
			msg.y = gy
			msg.theta = gth
			return True

		if (hasattr(msg, 'target_x') and hasattr(msg, 'target_y')
				and hasattr(msg, 'target_theta')):
			msg.target_x = gx
			msg.target_y = gy
			msg.target_theta = gth
			return True

		if hasattr(msg, 'pose'):
			q = tft.quaternion_from_euler(0.0, 0.0, gth)
			pose_obj = msg.pose
			# PoseStamped case
			if hasattr(pose_obj, 'pose'):
				pose_obj = pose_obj.pose

			if hasattr(pose_obj, 'position') and hasattr(pose_obj, 'orientation'):
				pose_obj.position.x = gx
				pose_obj.position.y = gy
				pose_obj.position.z = 0.0
				pose_obj.orientation.x = q[0]
				pose_obj.orientation.y = q[1]
				pose_obj.orientation.z = q[2]
				pose_obj.orientation.w = q[3]
				return True

		return False

	def run(self):
		rate = rospy.Rate(self.publish_hz)

		while not rospy.is_shutdown() and not self.have_odom:
			rospy.loginfo_throttle(2.0, '[pose_file_localiser] waiting for /odom...')
			rate.sleep()

		self.idx = self._nearest_index()
		rospy.loginfo('[pose_file_localiser] Starting from nearest index %d', self.idx)
		self._skip_ahead_if_goal_behind()

		while not rospy.is_shutdown():
			self._skip_ahead_if_goal_behind()
			gx, gy, gth = self.poses[self.idx]
			dx = gx - self.rx
			dy = gy - self.ry
			dist = math.sqrt(dx * dx + dy * dy)

			# Advance waypoint when close enough, keep final waypoint latched.
			if dist < self.goal_radius and self.idx < (len(self.poses) - 1):
				self.idx += 1
				gx, gy, gth = self.poses[self.idx]

			goal_msg = Pose2D()
			ok = self._assign_goal_message(goal_msg, gx, gy, gth)
			if not ok:
				rospy.logfatal(
						'[pose_file_localiser] Unsupported goal schema: %s',
					str(getattr(goal_msg, '__slots__', [])))
				return

			self.goal_pub.publish(goal_msg)
			rate.sleep()


if __name__ == '__main__':
	try:
		PoseFileLocaliser().run()
	except rospy.ROSInterruptException:
		pass

