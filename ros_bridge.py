import threading
import http.server

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import LaserScan
from slam_toolbox.srv import Reset
import json


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
        self.lidar_publisher = self.slam_toolbox_bridge.create_publisher(
            LaserScan, "/scan", qos_profile
        )
        self.reset_client = self.slam_toolbox_bridge.create_client(
            Reset, "/slam_toolbox/reset"
        )
        self.transformation_broadcaster = tf2_ros.TransformBroadcaster(
            self.slam_toolbox_bridge
        )
        self.static_transformation_broadcaster = tf2_ros.StaticTransformBroadcaster(
            self.slam_toolbox_bridge
        )
        self._spin_thread = threading.Thread(
            target=lambda: rclpy.spin(self.slam_toolbox_bridge), daemon=True
        )
        self._spin_thread.start()

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass

    def shutdown(self) -> None:
        try:
            self.slam_toolbox_bridge.destroy_node()
        except Exception as e:
            print(f"Failed to destroy slam_toolbox_bridge node: {e}")
        try:
            rclpy.shutdown()
        except Exception as e:
            print(f"Failed to shutdown rclpy: {e}")
        try:
            self._spin_thread.join(timeout=1.0)
        except Exception as e:
            print(f"Failed to join spin thread: {e}")

    def reset(self) -> None:
        if not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.slam_toolbox_bridge.get_logger().warn("Reset service not available")
            return
        request = Reset.Request()
        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(
            self.slam_toolbox_bridge, future, timeout_sec=2.0
        )

    def publish(
        self, x: float, y: float, yaw: float, lidar_range_array: list[float]
    ) -> None:
        stamp = self.slam_toolbox_bridge.get_clock().now().to_msg()

        yaw_rad = np.deg2rad(90.0 - yaw)
        qw = np.cos(yaw_rad / 2)
        qx = 0.0
        qy = 0.0
        qz = np.sin(yaw_rad / 2)
        t_odom_base = TransformStamped()
        t_odom_base.header.stamp = stamp
        t_odom_base.header.frame_id = "odom"
        t_odom_base.child_frame_id = "base_footprint"
        t_odom_base.transform.translation.x = float(x)
        t_odom_base.transform.translation.y = float(y)
        t_odom_base.transform.translation.z = 0.0
        t_odom_base.transform.rotation.x = float(qx)
        t_odom_base.transform.rotation.y = float(qy)
        t_odom_base.transform.rotation.z = float(qz)
        t_odom_base.transform.rotation.w = float(qw)
        self.transformation_broadcaster.sendTransform(t_odom_base)

        scan = LaserScan()
        scan.header.stamp = stamp
        scan.header.frame_id = "lidar"
        scan.angle_min = 3 * np.pi / 4
        scan.angle_max = -3 * np.pi / 4
        scan.angle_increment = (scan.angle_max - scan.angle_min) / (
            len(lidar_range_array) - 1
        )
        scan.range_min = 0.0
        scan.range_max = 50.0
        scan.ranges = (
            np.asarray(lidar_range_array, dtype=np.float32)
            .clip(scan.range_min, scan.range_max)
            .tolist()
        )
        self.lidar_publisher.publish(scan)

        t_base_lidar = TransformStamped()
        t_base_lidar.header.stamp = stamp
        t_base_lidar.header.frame_id = "base_footprint"
        t_base_lidar.child_frame_id = "lidar"

        t_base_lidar.transform.translation.x = 0.2733
        t_base_lidar.transform.translation.y = 0.0
        t_base_lidar.transform.translation.z = 0.096
        t_base_lidar.transform.rotation.x = 0.0
        t_base_lidar.transform.rotation.y = 0.0
        t_base_lidar.transform.rotation.z = 0.0
        t_base_lidar.transform.rotation.w = 1.0

        self.static_transformation_broadcaster.sendTransform(t_base_lidar)


class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        try:
            msg = json.loads(post_data.decode("utf-8"))
            command = msg.get("command")
            match command:
                case "reset":
                    bridge.reset()
                case "publish":
                    bridge.publish(
                        msg["data"][-3],
                        msg["data"][-2],
                        msg["data"][-1],
                        msg["data"][:-3],
                    )
            self.send_response(200)
            self.end_headers()
        except Exception as e:
            print(f"Error processing message: {e}")
            self.send_response(400)
            self.end_headers()


bridge = SlamToolboxBridge()

with http.server.HTTPServer(("127.0.0.1", 9000), Handler) as server:
    server.serve_forever()
