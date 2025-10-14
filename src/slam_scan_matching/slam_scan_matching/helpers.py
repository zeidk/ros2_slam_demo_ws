import numpy as np
from typing import Optional, Tuple
import rclpy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

try:
    # Provides utilities to create and parse PointCloud2 messages in Python
    from sensor_msgs_py import point_cloud2 as pc2
except Exception:
    pc2 = None


# ---------------------------------------------------------------------------
# Standard XYZ field specification for PointCloud2
# ---------------------------------------------------------------------------
FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]


def numpy_to_pointcloud2(points_xyz: np.ndarray, frame_id: str = "base_link") -> PointCloud2:
    """Convert a NumPy array of 3D points into a ROS 2 ``PointCloud2`` message.

    This helper wraps :func:`sensor_msgs_py.point_cloud2.create_cloud` to
    generate a minimal XYZ-only cloud from a NumPy array. It is typically used
    when publishing processed or filtered point clouds from Python nodes.

    Args:
        points_xyz (np.ndarray): Input array of shape ``(N, 3)`` containing
            XYZ coordinates in meters.
        frame_id (str, optional): Frame identifier for the published message.
            Defaults to ``"base_link"``.

    Returns:
        PointCloud2: A ROS 2 point cloud message containing the XYZ data.

    Raises:
        RuntimeError: If ``sensor_msgs_py`` is not available.
        ValueError: If the input array cannot be converted to ``float32`` or
            has invalid shape.

    Example:
        >>> pts = np.random.rand(100, 3)
        >>> cloud = numpy_to_pointcloud2(pts, frame_id="lidar")
    """
    if pc2 is None:
        raise RuntimeError(
            "sensor_msgs_py not available "
            "(install with: sudo apt install ros-$ROS_DISTRO-sensor-msgs-py)"
        )

    # Build a ROS 2 message header
    header = Header()
    header.stamp = rclpy.clock.Clock().now().to_msg()
    header.frame_id = frame_id

    # Convert to float32 and Python list (required by create_cloud)
    pts = np.asarray(points_xyz, dtype=np.float32)

    return pc2.create_cloud(header, FIELDS_XYZ, pts.tolist())


def pointcloud2_to_numpy(msg: PointCloud2) -> np.ndarray:
    """Convert a ROS 2 ``PointCloud2`` message into a NumPy array of XYZ points.

    The output contains one row per point and three columns for ``x``, ``y``,
    and ``z``. Points with NaN coordinates are skipped automatically.

    Args:
        msg (PointCloud2): Input ROS 2 point cloud message.

    Returns:
        np.ndarray: Array of shape ``(N, 3)`` containing XYZ coordinates.
        If no valid points exist, returns an empty ``(0, 3)`` array.

    Raises:
        RuntimeError: If ``sensor_msgs_py`` is not available.

    Example:
        >>> def callback(msg):
        ...     pts = pointcloud2_to_numpy(msg)
        ...     print(f"Received {pts.shape[0]} valid points")
    """
    if pc2 is None:
        raise RuntimeError(
            "sensor_msgs_py not available "
            "(install with: sudo apt install ros-$ROS_DISTRO-sensor-msgs-py)"
        )

    # Extract (x, y, z) tuples, skipping NaNs
    pts = np.array(
        list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)),
        dtype=np.float32,
    )

    # Return empty array if no points are present
    if pts.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    return pts.reshape(-1, 3)
