from __future__ import annotations

import sys
import threading
from typing import Any, Dict, List, Optional, SupportsFloat, Tuple, Union

import gymnasium as gym
import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from gym_unity.envs import UnityToGymWrapper
from gymnasium import spaces
from gymnasium.core import RenderFrame
from mlagents_envs.environment import UnityEnvironment
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header
from tf_transformations import quaternion_from_euler


class SlamToolboxBridge:
    def __init__(self) -> None:
        if not rclpy.ok():
            rclpy.init()
        self.slam_toolbox_bridge = rclpy.create_node("slam_toolbox_bridge")
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.lidar_publisher = self.slam_toolbox_bridge.create_publisher(LaserScan, "/scan", qos_profile)
        self.transformation_broadcaster = tf2_ros.TransformBroadcaster(self.slam_toolbox_bridge)
        self.static_transformation_broadcaster = tf2_ros.StaticTransformBroadcaster(self.slam_toolbox_bridge)
        self._shutdown_event = threading.Event()
        self._spin_thread = threading.Thread(target=lambda: rclpy.spin(self.slam_toolbox_bridge), daemon=True)
        self._spin_thread.start()

    def __del__(self) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        if self._shutdown_event.is_set():
            return
        self._shutdown_event.set()
        node = getattr(self, "slam_toolbox_bridge", None)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        thread = getattr(self, "_spin_thread", None)
        if thread is not None and thread.is_alive() and threading.current_thread() != thread:
            thread.join(timeout=1.0)

    def publish_lidar_scan(self, lidar_range_array):
        scan = LaserScan()
        scan.header.stamp = self.slam_toolbox_bridge.get_clock().now().to_msg()
        scan.header.frame_id = "lidar"
        scan.angle_min = -3 * np.pi / 4
        scan.angle_max = 3 * np.pi / 4
        scan.angle_increment = (scan.angle_max - scan.angle_min) / len(lidar_range_array)
        scan.range_min = 0.0
        scan.range_max = 50.0
        scan.ranges = lidar_range_array
        self.lidar_publisher.publish(scan)

    def publish_transforms(self, x: float, y: float, yaw: float):
        qx, qy, qz, qw = quaternion_from_euler(0.0, 0.0, float(yaw))

        t_odom_base = TransformStamped()
        t_odom_base.header.stamp = self.slam_toolbox_bridge.get_clock().now().to_msg()
        t_odom_base.header.frame_id = "odom"
        t_odom_base.child_frame_id = "base_link"
        t_odom_base.transform.translation.x = float(x)
        t_odom_base.transform.translation.y = float(y)
        t_odom_base.transform.translation.z = 0.0
        t_odom_base.transform.rotation.x = float(qx)
        t_odom_base.transform.rotation.y = float(qy)
        t_odom_base.transform.rotation.z = float(qz)
        t_odom_base.transform.rotation.w = float(qw)

        self.transformation_broadcaster.sendTransform(t_odom_base)

    def publish_laser_transforms(self) -> None:
        t_base_lidar = TransformStamped()
        t_base_lidar.header.stamp = self.slam_toolbox_bridge.get_clock().now().to_msg()
        t_base_lidar.header.frame_id = "base_link"
        t_base_lidar.child_frame_id = "lidar"

        t_base_lidar.transform.translation.x = 0.2733
        t_base_lidar.transform.translation.y = 0.0
        t_base_lidar.transform.translation.z = 0.096
        t_base_lidar.transform.rotation.x = 0.0
        t_base_lidar.transform.rotation.y = 0.0
        t_base_lidar.transform.rotation.z = 0.0
        t_base_lidar.transform.rotation.w = 1.0

        self.static_transformation_broadcaster.sendTransform(t_base_lidar)

    def create_laser_scan_msg(
        self,
        lidar_scan_rate: float,
        lidar_range_array: List[float],
        lidar_intensity_array: List[float],
    ) -> LaserScan:
        ls = LaserScan()
        ls.header = Header()
        ls.header.stamp = self.slam_toolbox_bridge.get_clock().now().to_msg()
        ls.header.frame_id = "lidar"
        ls.angle_min = -2.35619
        ls.angle_max = 2.35619
        ls.angle_increment = 0.004363323
        ls.time_increment = (1 / lidar_scan_rate) / 360
        ls.scan_time = ls.time_increment * 360
        ls.range_min = 0.06
        ls.range_max = 10.0
        ls.ranges = lidar_range_array
        ls.intensities = lidar_intensity_array
        return ls


class AutoDRIVEWrapper(gym.Wrapper):
    def __init__(self) -> None:
        if sys.platform == "darwin":
            binary_path = "autodrive_macos_build/autodrive.app"
        elif sys.platform == "win32":
            binary_path = r"autodrive_windows_build/autodrive/AutoDRIVE Simulator.exe"
        else:
            binary_path = "autodrive_linux_build/autodrive.x86_64"

        self.env = UnityToGymWrapper(UnityEnvironment(binary_path, no_graphics=True), allow_multiple_obs=True)

        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(55,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )  # (0, 1) for throttle and (-1, 1) for steering
        self.reward_range = (-np.inf, np.inf)
        self._render_mode: str = "rgb_array"
        self._metadata = {"render_fps": 60}
        self.slam_toolbox_bridge = SlamToolboxBridge()

    @property
    def render_mode(self) -> str:
        return self._render_mode

    def _convert_obs(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        return {"state": obs[0]}

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        self.slam_toolbox_bridge.publish_lidar_scan(obs[0][:-3])
        self.slam_toolbox_bridge.publish_transforms(obs[0][-3], obs[0][-2], obs[0][-1])
        self.slam_toolbox_bridge.publish_laser_transforms()

        return self._convert_obs(obs), reward, done, False, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs = self.env.reset()
        return self._convert_obs(obs), {}  # type: ignore

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        return self.env.render()  # type: ignore

    def close(self):
        self.env.close()
