import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import numpy as np
from .helpers import pointcloud2_to_numpy, numpy_to_pointcloud2


class PreprocessNode(Node):
    """ROS 2 node for preprocessing raw LiDAR point clouds.

    This node performs two main operations:

    1. **Pass-through filtering** along the Z-axis to remove points outside
       a user-defined vertical range.
    2. **Voxel down-sampling** to reduce the number of points while preserving
       geometric structure.

    The processed point cloud is published to an output topic for downstream
    nodes (e.g., feature extraction or scan matching).

    Parameters (ROS 2):
        input_topic (str): Topic name for incoming raw point clouds.
        output_topic (str): Topic name for publishing filtered clouds.
        voxel_size (float): Voxel size in meters for down-sampling.
        passthrough_z_min (float): Minimum Z value for pass-through filter.
        passthrough_z_max (float): Maximum Z value for pass-through filter.
        frame_id (str): Frame ID to embed in output messages.
    """

    def __init__(self):
        """Initialize the preprocessing node and its parameters."""
        super().__init__("preprocess_node")

        # Declare configurable parameters with default values
        self.declare_parameter("input_topic", "/points_raw")
        self.declare_parameter("output_topic", "/points_filtered")
        self.declare_parameter("voxel_size", 0.2)
        self.declare_parameter("passthrough_z_min", -3.0)
        self.declare_parameter("passthrough_z_max", 3.0)
        self.declare_parameter("frame_id", "base_link")

        # Retrieve topic names from parameters
        self.in_topic = self.get_parameter("input_topic").value
        self.out_topic = self.get_parameter("output_topic").value

        # Create ROS 2 subscriber and publisher
        self.sub = self.create_subscription(PointCloud2, self.in_topic, self.cb_cloud, 10)
        self.pub = self.create_publisher(PointCloud2, self.out_topic, 10)

        # Log startup information
        self.get_logger().info(f"PreprocessNode: {self.in_topic} -> {self.out_topic}")

    def cb_cloud(self, msg: PointCloud2):
        """Callback function to process incoming LiDAR point clouds.

        The function converts a ROS 2 ``PointCloud2`` message into a NumPy array,
        applies Z-axis pass-through filtering, optionally performs voxel
        down-sampling, and publishes the resulting filtered cloud.

        Args:
            msg (PointCloud2): Input ROS 2 message containing the point cloud.

        Raises:
            Exception: If parsing or publishing the point cloud fails.
        """
        try:
            # Convert PointCloud2 → NumPy array (Nx3)
            pts = pointcloud2_to_numpy(msg)
        except Exception as e:
            self.get_logger().error(f"Failed to parse PointCloud2: {e}")
            return

        # Skip processing if the input cloud is empty
        if pts.size == 0:
            return

        # Retrieve pass-through filter limits
        zmin = float(self.get_parameter("passthrough_z_min").value)
        zmax = float(self.get_parameter("passthrough_z_max").value)

        # Keep only points within the vertical range
        mask = (pts[:, 2] >= zmin) & (pts[:, 2] <= zmax)
        pts = pts[mask]

        # Retrieve voxel size parameter
        voxel = float(self.get_parameter("voxel_size").value)

        # Apply voxel grid down-sampling if enabled
        if voxel > 0 and pts.shape[0] > 0:
            # Divide by voxel size to normalize into discrete bins
            vox = (pts / voxel).astype(np.float32)
            # Quantize and round down to form voxel indices
            vox = np.floor(vox).astype(np.int32)
            # Keep one representative point per voxel
            _, idx = np.unique(vox, axis=0, return_index=True)
            pts = pts[idx]

        try:
            # Convert filtered points back to PointCloud2
            out = numpy_to_pointcloud2(pts, frame_id=self.get_parameter("frame_id").value)
            # Publish filtered cloud
            self.pub.publish(out)
        except Exception as e:
            self.get_logger().error(f"Failed to publish filtered cloud: {e}")


def main():
    """Entry point for the ``preprocess_node`` executable.

    Initializes rclpy, creates an instance of :class:`PreprocessNode`,
    and keeps it active until shutdown. Ensures graceful cleanup on exit.
    """
    try:
        # Initialize ROS 2 system
        rclpy.init()

        # Create node instance
        node = PreprocessNode()
        node.get_logger().info("PreprocessNode is running. Press Ctrl+C to exit.")

        # Spin until interrupted
        rclpy.spin(node)

    except KeyboardInterrupt:
        print("\n[PreprocessNode] Interrupted by user. Exiting...")
    except Exception as e:
        print(f"[PreprocessNode] Unexpected error: {e}")
    finally:
        # Always perform proper cleanup
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()
        print("[PreprocessNode] Shutdown complete.")
