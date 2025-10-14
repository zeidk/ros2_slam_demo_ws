
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np
from .helpers import pointcloud2_to_numpy, numpy_to_pointcloud2

def estimate_curvature(points: np.ndarray, k: int = 10) -> np.ndarray:
    if points.shape[0] < k:
        return np.zeros((points.shape[0],), dtype=np.float32)
    pad = k//2
    curv = np.zeros((points.shape[0],), dtype=np.float32)
    for i in range(points.shape[0]):
        a = max(0, i-pad); b = min(points.shape[0], i+pad+1)
        neigh = points[a:b]
        mu = neigh.mean(axis=0)
        dif = neigh - mu
        curv[i] = np.mean(np.sum(dif*dif, axis=1))
    return curv

class FeatureNode(Node):
    def __init__(self):
        super().__init__("feature_node")
        self.declare_parameter("input_topic", "/points_filtered")
        self.declare_parameter("corners_topic", "/keypoints/corners")
        self.declare_parameter("planes_topic", "/keypoints/planes")
        self.declare_parameter("corner_curvature_thresh", 0.1)
        self.declare_parameter("plane_curvature_thresh", 0.01)
        self.declare_parameter("frame_id", "base_link")
        self.in_topic = self.get_parameter("input_topic").value
        self.sub = self.create_subscription(PointCloud2, self.in_topic, self.cb_cloud, 10)
        self.pub_corners = self.create_publisher(PointCloud2, self.get_parameter("corners_topic").value, 10)
        self.pub_planes  = self.create_publisher(PointCloud2, self.get_parameter("planes_topic").value, 10)
        self.get_logger().info(f"FeatureNode: {self.in_topic} -> (/keypoints/corners,/keypoints/planes)")

    def cb_cloud(self, msg: PointCloud2):
        try:
            pts = pointcloud2_to_numpy(msg)
        except Exception as e:
            self.get_logger().error(f"parse error: {e}")
            return
        if pts.shape[0] == 0:
            return
        curv = estimate_curvature(pts, k=10)
        corner_t = float(self.get_parameter("corner_curvature_thresh").value)
        plane_t  = float(self.get_parameter("plane_curvature_thresh").value)
        corners = pts[curv >= corner_t]
        planes  = pts[curv <= plane_t]
        try:
            if corners.size:
                self.pub_corners.publish(numpy_to_pointcloud2(corners, frame_id=self.get_parameter("frame_id").value))
            if planes.size:
                self.pub_planes.publish(numpy_to_pointcloud2(planes, frame_id=self.get_parameter("frame_id").value))
        except Exception as e:
            self.get_logger().error(f"publish error: {e}")

def main():
    """Entry point for the ``feature_node`` executable.

    Initializes rclpy, instantiates :class:`FeatureNode`, and enters the ROS 2
    spin loop until shutdown. The function ensures that all resources are
    properly cleaned up even if an exception occurs during execution.
    """
    try:
        # Initialize the ROS 2 client library
        rclpy.init()

        # Create an instance of the feature extraction node
        node = FeatureNode()

        # Keep the node active and processing callbacks until a shutdown signal
        node.get_logger().info("FeatureNode is running. Press Ctrl+C to exit.")
        rclpy.spin(node)

    except KeyboardInterrupt:
        # Graceful exit when user interrupts with Ctrl+C
        print("\n[FeatureNode] Interrupted by user (Ctrl+C). Shutting down...")
    except Exception as e:
        # Log unexpected exceptions for debugging
        print(f"[FeatureNode] Unexpected error: {e}")
    finally:
        # Always destroy the node and shut down ROS cleanly
        try:
            node.destroy_node()
        except Exception:
            # Node might not exist if initialization failed
            pass

        # Shutdown the ROS 2 client library
        rclpy.shutdown()
        print("[FeatureNode] Shutdown complete.")

