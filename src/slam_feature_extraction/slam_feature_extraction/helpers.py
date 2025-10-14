import numpy as np
from typing import Optional, Tuple
import rclpy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

try:
    # sensor_msgs_py provides convenience functions for creating and reading PointCloud2
    from sensor_msgs_py import point_cloud2 as pc2
except Exception:
    pc2 = None

# Standard XYZ field specification for PointCloud2 messages
FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]


def numpy_to_pointcloud2(points_xyz: np.ndarray, frame_id: str = "base_link") -> PointCloud2:
    """Convert a NumPy array of 3-D points to a ROS 2 PointCloud2 message.

    Args:
        points_xyz (np.ndarray): Array of shape ``(N, 3)`` containing XYZ coordinates.
        frame_id (str, optional): Frame ID to embed in the message header.
            Defaults to ``"base_link"``.

    Returns:
        PointCloud2: A populated ROS 2 point cloud message containing the XYZ data.

    Raises:
        RuntimeError: If the ``sensor_msgs_py`` module is not available.
        ValueError: If the input array cannot be converted to float32 or has
            invalid dimensions.

    Notes:
        - The timestamp is generated from the current ROS 2 clock.
        - Intensity or color channels are not included; only XYZ coordinates
          are published.
    """
    if pc2 is None:
        raise RuntimeError(
            "sensor_msgs_py not available "
            "(install with: sudo apt install ros-$ROS_DISTRO-sensor-msgs-py)"
        )

    # Prepare header with current time and frame
    header = Header()
    header.stamp = rclpy.clock.Clock().now().to_msg()
    header.frame_id = frame_id

    # Ensure float32 array and convert to list of points
    pts = np.asarray(points_xyz, dtype=np.float32)

    # Create PointCloud2 message
    return pc2.create_cloud(header, FIELDS_XYZ, pts.tolist())


def pointcloud2_to_numpy(msg: PointCloud2) -> np.ndarray:
    """Convert a ROS 2 PointCloud2 message into a NumPy ``(N, 3)`` array.

    Args:
        msg (PointCloud2): Incoming ROS 2 point cloud message containing XYZ fields.

    Returns:
        np.ndarray: Array of shape ``(N, 3)`` with XYZ coordinates in meters.

    Raises:
        RuntimeError: If ``sensor_msgs_py`` is not available.

    Notes:
        - Only the fields ``x``, ``y``, and ``z`` are extracted.
        - Points with NaN values are automatically skipped.
        - Returns an empty array if the input message contains no valid points.
    """
    if pc2 is None:
        raise RuntimeError(
            "sensor_msgs_py not available "
            "(install with: sudo apt install ros-$ROS_DISTRO-sensor-msgs-py)"
        )

    # Extract point data as an iterator of (x, y, z) tuples
    pts = np.array(
        list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)),
        dtype=np.float32,
    )

    # Handle empty point clouds gracefully
    if pts.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    # Reshape to (N, 3)
    return pts.reshape(-1, 3)
