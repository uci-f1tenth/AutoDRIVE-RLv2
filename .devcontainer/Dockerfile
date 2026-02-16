FROM docker.io/ros:kilted-ros-core

RUN apt-get update && apt-get install -y ros-$ROS_DISTRO-foxglove-bridge ros-$ROS_DISTRO-slam-toolbox ros-$ROS_DISTRO-tf-transformations python3-pip python3-colcon-common-extensions git
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc