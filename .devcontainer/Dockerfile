FROM ros:humble-perception

RUN sudo apt-get update
RUN sudo apt-get install -y ros-$ROS_DISTRO-foxglove-bridge ros-$ROS_DISTRO-slam-toolbox ros-$ROS_DISTRO-tf-transformations python3-pip
RUN pip install --upgrade pip