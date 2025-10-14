import numpy as np
from typing import Optional, Tuple
import rclpy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

try:
    # The sensor_msgs_py library provides helper utilities for PointCloud2 creation and parsing
    from sensor_msgs_py import point_cloud2 as pc2
except Exception:
    pc2 = None


# Standard field definitions for XYZ-only PointCloud2 messages
FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]


def numpy_to_pointcloud2(points_xyz: np.ndarray, frame_id: str = "base_link") -> PointCloud2:
    """Convert a NumPy array of 3D points to a ROS 2 ``PointCloud2`` message.

    This function is typically used to publish point cloud data that was
    processed in Python (e.g., filtered, voxelized, or transformed) back
    into the ROS 2 network.  It creates a minimal ``PointCloud2`` message
    containing only the ``x``, ``y``, and ``z`` fields.

    Args:
        points_xyz (np.ndarray): Array of shape ``(N, 3)`` containing point
            coordinates in meters.
        frame_id (str, optional): Frame identifier to use in the message
            header. Defaults to ``"base_link"``.

    Returns:
        PointCloud2: A ROS 2 message containing the converted point cloud.

    Raises:
        RuntimeError: If ``sensor_msgs_py`` is unavailable.
        ValueError: If ``points_xyz`` cannot be converted to float32 or
            has invalid shape.

    Example:
        >>> pts = np.random.rand(100, 3)
        >>> msg = numpy_to_pointcloud2(pts, frame_id="lidar")
    """
    if pc2 is None:
        raise RuntimeError(
            "sensor_msgs_py not available "
            "(install with: sudo apt install ros-$ROS_DISTRO-sensor-msgs-py)"
        )

    # Build ROS 2 header with current timestamp and frame information
    header = Header()
    header.stamp = rclpy.clock.Clock().now().to_msg()
    header.frame_id = frame_id

    # Ensure the array is float32 and convert to a Python list for pc2.create_cloud
    pts = np.asarray(points_xyz, dtype=np.float32)

    # Create the PointCloud2 message (XYZ only)
    return pc2.create_cloud(header, FIELDS_XYZ, pts.tolist())


def pointcloud2_to_numpy(msg: PointCloud2) -> np.ndarray:
    """Convert a ROS 2 ``PointCloud2`` message into a NumPy ``(N, 3)`` array.

    The function extracts the ``x``, ``y``, and ``z`` fields from the input
    message and skips any points that contain NaN values.  The output is a
    dense array suitable for further numerical processing (e.g., filtering,
    segmentation, or registration).

    Args:
        msg (PointCloud2): Input point cloud message.

    Returns:
        np.ndarray: Array of shape ``(N, 3)`` containing the extracted XYZ
        coordinates in meters. Returns an empty array if no valid points
        are present.

    Raises:
        RuntimeError: If ``sensor_msgs_py`` is unavailable.

    Example:
        >>> def callback(msg):
        ...     pts = pointcloud2_to_numpy(msg)
        ...     print(f"Received {pts.shape[0]} points")
    """
    if pc2 is None:
        raise RuntimeError(
            "sensor_msgs_py not available "
            "(install with: sudo apt install ros-$ROS_DISTRO-sensor-msgs-py)"
        )

    # Extract the XYZ coordinates, skipping NaN values
    pts = np.array(
        list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)),
        dtype=np.float32,
    )

    # Return an empty array if the message contained no valid points
    if pts.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    # Ensure the output shape is (N, 3)
    return pts.reshape(-1, 3)
