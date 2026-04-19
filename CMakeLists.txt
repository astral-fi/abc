cmake_minimum_required(VERSION 3.0.2)
project(abc)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  std_srvs
  geometry_msgs
  nav_msgs
  sensor_msgs
  tf
  cv_bridge
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs std_srvs geometry_msgs nav_msgs sensor_msgs tf cv_bridge
)

# Mark Python scripts as executable so rosrun can find them
catkin_install_python(PROGRAMS
  scripts/jetracer_drive_to_pose_controller.py
  scripts/odom_integrator.py
  scripts/visual_pose_localiser.py
  scripts/data_collect.py
  scripts/data_save.py
  scripts/cmd_vel_to_jetracer.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

# Install config files
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
