from __future__ import annotations

import logging
import math
import os
import shlex
import signal
import subprocess
import sys
import threading
from typing import Any, Dict, List, Optional, Sequence, SupportsFloat, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import RenderFrame
from gym_unity.envs import UnityToGymWrapper  # type: ignore
from mlagents_envs.environment import UnityEnvironment  # type: ignore

LOGGER = logging.getLogger(__name__)


class SlamToolboxBridge:
    """Minimal helper to stream LiDAR scans into slam_toolbox and fetch poses."""

    def __init__(
        self,
        lidar_topic: str,
        pose_topic: str,
        frame_id: str,
        angle_min: float,
        angle_max: Optional[float],
        angle_increment: float,
        range_min: float,
        range_max: float,
        namespace: Optional[str] = None,
        slam_launch_cmd: Optional[Union[str, Sequence[str]]] = "ros2 run slam_toolbox sync_slam_toolbox_node",
        node_name: str = "autodrive_rl_slam_bridge",
    ) -> None:
        try:
            import rclpy
            from geometry_msgs.msg import PoseStamped
            from rclpy.executors import MultiThreadedExecutor
            from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
            from sensor_msgs.msg import LaserScan
        except ImportError as exc:  # pragma: no cover - executed only in ROS setups
            raise RuntimeError(
                "ROS 2 Python packages (rclpy, geometry_msgs, sensor_msgs) are required to enable slam_toolbox integration"
            ) from exc

        self._rclpy = rclpy
        self._PoseStamped = PoseStamped
        self._LaserScan = LaserScan
        self._executor_cls = MultiThreadedExecutor
        self._QoSProfile = QoSProfile
        self._QoSHistoryPolicy = QoSHistoryPolicy
        self._QoSReliabilityPolicy = QoSReliabilityPolicy
        self._lidar_topic = lidar_topic
        self._pose_topic = pose_topic
        self._frame_id = frame_id
        self._angle_min = angle_min
        self._angle_max = angle_max
        self._angle_increment = angle_increment
        self._range_min = range_min
        self._range_max = range_max
        self._pose_lock = threading.Lock()
        self._latest_pose: Optional[np.ndarray] = None
        self._last_pose_ts: float = 0.0
        self._slam_launch_cmd = self._normalize_cmd(slam_launch_cmd)
        self._ros_env = os.environ.copy()
        if namespace:
            self._ros_env["ROS_NAMESPACE"] = namespace

        self._slam_process: Optional[subprocess.Popen] = None
        self._node = None
        self._executor = None
        self._spin_thread: Optional[threading.Thread] = None
        self._pose_sub = None
        self._lidar_pub = None
        self._start_ros_node(node_name)
        if self._slam_launch_cmd:
            self._start_slam_toolbox()

    def _normalize_cmd(
        self, cmd: Optional[Union[str, Sequence[str]]]
    ) -> Optional[List[str]]:
        if cmd is None:
            return None
        if isinstance(cmd, str):
            return shlex.split(cmd)
        return list(cmd)

    def _start_ros_node(self, node_name: str) -> None:
        try:
            already_running = self._rclpy.ok()
        except Exception:
            already_running = False
        if not already_running:
            self._rclpy.init(args=None)
        self._node = self._rclpy.create_node(node_name)
        self._executor = self._executor_cls()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        sensor_qos = self._QoSProfile(
            depth=5,
            reliability=self._QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=self._QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
        )
        pose_qos = self._QoSProfile(
            depth=5,
            reliability=self._QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE,
            history=self._QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
        )
        self._lidar_pub = self._node.create_publisher(self._LaserScan, self._lidar_topic, sensor_qos)
        self._pose_sub = self._node.create_subscription(
            self._PoseStamped,
            self._pose_topic,
            self._pose_callback,
            pose_qos,
        )

    def _start_slam_toolbox(self) -> None:
        try:
            self._slam_process = subprocess.Popen(self._slam_launch_cmd, env=self._ros_env)
        except FileNotFoundError as exc:
            LOGGER.warning("slam_toolbox launch failed: %s", exc)
            self._slam_process = None

    def publish_scan(
        self,
        ranges: np.ndarray,
        intensities: Optional[np.ndarray],
        scan_rate_hz: float,
    ) -> None:
        if self._lidar_pub is None or self._node is None or ranges.size == 0:
            return
        ranges = np.asarray(ranges, dtype=np.float32).flatten()
        intensities_np = (
            np.asarray(intensities, dtype=np.float32).flatten() if intensities is not None else None
        )
        msg = self._LaserScan()
        msg.header.frame_id = self._frame_id
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.angle_min = self._angle_min
        if self._angle_max is None:
            msg.angle_max = self._angle_min + self._angle_increment * max(len(ranges) - 1, 0)
        else:
            msg.angle_max = self._angle_max
        msg.angle_increment = self._angle_increment
        num_points = max(len(ranges), 1)
        scan_period = 1.0 / max(scan_rate_hz, 1e-3)
        msg.scan_time = scan_period
        msg.time_increment = scan_period / num_points
        msg.range_min = self._range_min
        msg.range_max = self._range_max
        msg.ranges = np.clip(ranges, self._range_min, self._range_max).tolist()
        if intensities_np is not None and intensities_np.shape[0] == ranges.shape[0]:
            msg.intensities = intensities_np.tolist()
        else:
            msg.intensities = []
        self._lidar_pub.publish(msg)

    def _pose_callback(self, msg: Any) -> None:
        if self._node is None:
            return
        pose = msg.pose
        q = pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        with self._pose_lock:
            self._latest_pose = np.array(
                [pose.position.x, pose.position.y, yaw], dtype=np.float32
            )
            self._last_pose_ts = self._node.get_clock().now().nanoseconds * 1e-9

    def latest_pose(self) -> Optional[np.ndarray]:
        with self._pose_lock:
            if self._latest_pose is None:
                return None
            return self._latest_pose.copy()

    def close(self) -> None:
        if self._pose_sub is not None and self._node is not None:
            self._node.destroy_subscription(self._pose_sub)
            self._pose_sub = None
        if self._lidar_pub is not None and self._node is not None:
            self._node.destroy_publisher(self._lidar_pub)
            self._lidar_pub = None
        if self._executor and self._node is not None:
            self._executor.remove_node(self._node)
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        if self._spin_thread and self._spin_thread.is_alive():
            self._spin_thread.join(timeout=1.0)
        self._spin_thread = None
        try:
            if self._rclpy.ok():
                self._rclpy.shutdown()
        except Exception:  # pragma: no cover - defensive shutdown
            pass
        if self._slam_process is not None:
            self._terminate_process(self._slam_process)
            self._slam_process = None

    def _terminate_process(self, proc: subprocess.Popen) -> None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=1)


class AutoDRIVEWrapper(gym.Wrapper):
    def __init__(
        self,
        enable_slam: bool = True,
        slam_launch_cmd: Optional[Union[str, Sequence[str]]] = "ros2 run slam_toolbox sync_slam_toolbox_node",
        slam_pose_topic: str = "/slam_toolbox/pose",
        slam_namespace: Optional[str] = None,
        lidar_topic: str = "/autodrive/f1tenth_1/lidar",
        lidar_frame_id: str = "lidar",
        lidar_obs_index: Optional[int] = None,
        min_lidar_samples: int = 360,
        lidar_scan_rate_hz: float = 10.0,
        lidar_angle_min: float = -2.35619,
        lidar_angle_max: Optional[float] = 2.35619,
        lidar_angle_increment: float = 0.004363323,
        lidar_range_min: float = 0.06,
        lidar_range_max: float = 10.0,
        pose_position_scale: float = 20.0,
    ) -> None:
        if sys.platform == "darwin":
            binary_path = "autodrive_macos_build/autodrive.app"
        elif sys.platform == "win32":
            binary_path = r"autodrive_windows_build/autodrive/AutoDRIVE Simulator.exe"
        else:
            binary_path = "autodrive_linux_build/autodrive.x86_64"

        unity_env = UnityEnvironment(binary_path)
        self.env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)
        self._slam_enabled = enable_slam
        self._slam_bridge: Optional[SlamToolboxBridge] = None
        self._lidar_obs_index = lidar_obs_index
        self._min_lidar_samples = min_lidar_samples
        self._lidar_scan_rate = lidar_scan_rate_hz
        self._pose_scale = pose_position_scale
        self._last_pose = np.zeros(4, dtype=np.float32)

        if self._slam_enabled:
            try:
                self._slam_bridge = SlamToolboxBridge(
                    lidar_topic=lidar_topic,
                    pose_topic=slam_pose_topic,
                    frame_id=lidar_frame_id,
                    angle_min=lidar_angle_min,
                    angle_max=lidar_angle_max,
                    angle_increment=lidar_angle_increment,
                    range_min=lidar_range_min,
                    range_max=lidar_range_max,
                    namespace=slam_namespace,
                    slam_launch_cmd=slam_launch_cmd,
                )
            except RuntimeError as exc:
                LOGGER.warning("SLAM bridge disabled: %s", exc)
                self._slam_bridge = None

        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(55,), dtype=np.float32),
                "slam_pose": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            }
        )
        self.action_space = spaces.MultiDiscrete([3, 3])
        self.reward_range = (-np.inf, np.inf)
        self._render_mode: str = "rgb_array"
        self._metadata = {"render_fps": 60}

    @property
    def render_mode(self) -> str:
        return self._render_mode

    def _convert_obs(self, obs: Any, info: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
        state = self._extract_state(obs)
        slam_pose = self._process_lidar_and_pose(obs, info)
        return {"state": state, "slam_pose": slam_pose}

    def _extract_state(self, obs: Any) -> np.ndarray:
        if isinstance(obs, dict):
            if "state" in obs:
                return np.asarray(obs["state"], dtype=np.float32)
            first_key = next(iter(obs))
            return np.asarray(obs[first_key], dtype=np.float32)
        if isinstance(obs, (list, tuple)):
            return np.asarray(obs[0], dtype=np.float32)
        return np.asarray(obs, dtype=np.float32)

    def _process_lidar_and_pose(
        self,
        obs: Any,
        info: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        lidar_payload = self._extract_lidar(obs, info)
        if self._slam_bridge and lidar_payload is not None:
            ranges, intensities, scan_rate = lidar_payload
            try:
                self._slam_bridge.publish_scan(
                    ranges=ranges,
                    intensities=intensities,
                    scan_rate_hz=scan_rate or self._lidar_scan_rate,
                )
            except RuntimeError as exc:
                LOGGER.warning("Failed to publish LiDAR scan: %s", exc)
        pose = self._slam_bridge.latest_pose() if self._slam_bridge else None
        if pose is None:
            return self._last_pose.copy()
        processed = self._preprocess_pose(pose)
        self._last_pose = processed
        return processed

    def _extract_lidar(
        self,
        obs: Any,
        info: Optional[Dict[str, Any]],
    ) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], Optional[float]]]:
        ranges: Optional[np.ndarray] = None
        intensities: Optional[np.ndarray] = None
        scan_rate: Optional[float] = None

        if isinstance(obs, dict):
            lidar_value = obs.get("lidar")
            if isinstance(lidar_value, dict):
                ranges = np.asarray(lidar_value.get("ranges")) if lidar_value.get("ranges") is not None else None
                if lidar_value.get("intensities") is not None:
                    intensities = np.asarray(lidar_value["intensities"])
                scan_rate = lidar_value.get("scan_rate")
            elif lidar_value is not None:
                ranges = np.asarray(lidar_value)
        elif isinstance(obs, (list, tuple)):
            lidar_value = self._select_lidar_from_sequence(obs)
            if lidar_value is not None:
                ranges = np.asarray(lidar_value)

        if ranges is None and info is not None:
            if "lidar" in info and isinstance(info["lidar"], dict):
                lidar_dict = info["lidar"]
                ranges = np.asarray(lidar_dict.get("ranges")) if lidar_dict.get("ranges") is not None else None
                if lidar_dict.get("intensities") is not None:
                    intensities = np.asarray(lidar_dict["intensities"])
                scan_rate = lidar_dict.get("scan_rate")
            elif "lidar_ranges" in info:
                ranges = np.asarray(info["lidar_ranges"])
                if "lidar_intensities" in info:
                    intensities = np.asarray(info["lidar_intensities"])
                if "lidar_scan_rate" in info:
                    scan_rate = float(info["lidar_scan_rate"])

        if ranges is None:
            return None
        if ranges.ndim > 1:
            ranges = ranges.reshape(-1)
        if intensities is not None and intensities.ndim > 1:
            intensities = intensities.reshape(-1)
        return ranges.astype(np.float32), (
            intensities.astype(np.float32) if intensities is not None else None
        ), scan_rate

    def _select_lidar_from_sequence(self, obs_sequence: Union[List[Any], Tuple[Any, ...]]) -> Optional[np.ndarray]:
        if not obs_sequence:
            return None
        if self._lidar_obs_index is None:
            for idx, entry in enumerate(obs_sequence[1:], start=1):
                entry_array = np.asarray(entry)
                if entry_array.ndim == 1 and entry_array.shape[0] >= self._min_lidar_samples:
                    self._lidar_obs_index = idx
                    break
        if self._lidar_obs_index is None:
            return None
        if self._lidar_obs_index >= len(obs_sequence):
            return None
        return np.asarray(obs_sequence[self._lidar_obs_index])

    def _preprocess_pose(self, pose: np.ndarray) -> np.ndarray:
        processed = np.zeros(4, dtype=np.float32)
        processed[0] = np.clip(pose[0] / self._pose_scale, -1.0, 1.0)
        processed[1] = np.clip(pose[1] / self._pose_scale, -1.0, 1.0)
        processed[2] = math.sin(pose[2])
        processed[3] = math.cos(pose[2])
        return processed

    def step(self, action: Any) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        return self._convert_obs(obs, info), reward, done, False, info

    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        return self._convert_obs(obs, {}), {}

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        return self.env.render()

    def close(self):
        try:
            self.env.close()
        finally:
            if self._slam_bridge:
                self._slam_bridge.close()
