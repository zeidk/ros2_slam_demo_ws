import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
import numpy as np
from .helpers import pointcloud2_to_numpy, numpy_to_pointcloud2


def icp_point_to_point(src: np.ndarray,
                       dst: np.ndarray,
                       max_iters: int = 20,
                       tol: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """Run a minimal point-to-point ICP between two XYZ point sets.

    This implementation uses a **brute-force nearest neighbor** assignment
    (O(N²) per iteration) and SVD-based rigid alignment. It is intended for
    teaching and demo purposes, not for large-scale datasets.

    Args:
        src: Source points as an ``(N, 3)`` array.
        dst: Target points as an ``(M, 3)`` array.
        max_iters: Maximum ICP iterations.
        tol: Convergence tolerance on mean correspondence error (meters).

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(R, t)`` where
        - ``R`` is a ``(3, 3)`` rotation matrix,
        - ``t`` is a translation vector of shape ``(3,)``.
    """
    # Early exit if either cloud is empty
    if src.shape[0] == 0 or dst.shape[0] == 0:
        return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    R = np.eye(3, dtype=np.float32)
    t = np.zeros(3, dtype=np.float32)
    prev_err = 1e9

    for _ in range(max_iters):
        # Transform source by the current estimate
        src_tf = (src @ R.T) + t

        # Brute-force nearest neighbors from src_tf -> dst
        d2 = ((src_tf[:, None, :] - dst[None, :, :]) ** 2).sum(axis=2)
        idx = np.argmin(d2, axis=1)
        matched_dst = dst[idx]

        # Center both sets
        mu_src = src_tf.mean(axis=0)
        mu_dst = matched_dst.mean(axis=0)
        X = src_tf - mu_src
        Y = matched_dst - mu_dst

        # Compute optimal rotation via SVD
        U, S, Vt = np.linalg.svd(X.T @ Y)
        R_iter = Vt.T @ U.T
        if np.linalg.det(R_iter) < 0:
            # Enforce a proper rotation (no reflection)
            Vt[2, :] *= -1
            R_iter = Vt.T @ U.T

        # Translation to align centroids
        t_iter = mu_dst - mu_src @ R_iter.T

        # Compose increments
        R = R_iter @ R
        t = t @ R_iter.T + t_iter

        # Mean correspondence error
        err = np.mean(np.sqrt(((src_tf - matched_dst) ** 2).sum(axis=1)))
        if abs(prev_err - err) < tol:
            break
        prev_err = err

    return R, t


def R_t_to_quat_trans(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a rotation matrix and translation to (x, y, z, w) quaternion + translation.

    Args:
        R: Rotation matrix of shape ``(3, 3)``.
        t: Translation vector of shape ``(3,)``.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(q, t)`` where ``q`` is the quaternion
        ``[x, y, z, w]`` and ``t`` is the unchanged translation.
    """
    # Numerical guard in trace
    qw = np.sqrt(max(1.0 + np.trace(R), 1e-6)) / 2.0
    qx = (R[2, 1] - R[1, 2]) / (4 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw)
    return np.array([qx, qy, qz, qw], dtype=np.float32), t


class ICPNode(Node):
    """ROS 2 node that performs frame-to-frame ICP and publishes odometry.

    The node subscribes to a filtered LiDAR cloud, aligns the current frame to
    the previous one using point-to-point ICP, integrates the pose over time,
    and publishes:

    - ``/points_aligned``: the current cloud aligned to the previous frame
    - ``/odom``: incremental odometry pose

    Parameters (ROS 2):
        input_topic (str): Input cloud topic (default: ``/points_filtered``).
        aligned_topic (str): Output aligned cloud topic (default: ``/points_aligned``).
        odom_topic (str): Output odometry topic (default: ``/odom``).
        max_iters (int): ICP maximum iterations (default: 20).
        frame_id (str): Child frame of the platform (default: ``base_link``).
        odom_frame_id (str): Fixed odom frame (default: ``odom``).
    """

    def __init__(self) -> None:
        """Initialize node, parameters, I/O, and internal pose buffers."""
        super().__init__("icp_node")

        # Declare parameters
        self.declare_parameter("input_topic", "/points_filtered")
        self.declare_parameter("aligned_topic", "/points_aligned")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("max_iters", 20)
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("odom_frame_id", "odom")

        # Internal state for incremental pose
        self.prev_pts = None
        self.pose_R = np.eye(3, dtype=np.float32)
        self.pose_t = np.zeros(3, dtype=np.float32)

        # I/O
        self.sub = self.create_subscription(
            PointCloud2,
            self.get_parameter("input_topic").value,
            self.cb_cloud,
            10,
        )
        self.pub_aligned = self.create_publisher(
            PointCloud2, self.get_parameter("aligned_topic").value, 10
        )
        self.pub_odom = self.create_publisher(
            Odometry, self.get_parameter("odom_topic").value, 10
        )

        self.get_logger().info("ICPNode ready.")

    def cb_cloud(self, msg: PointCloud2) -> None:
        """Process an incoming cloud, run ICP to the previous frame, and publish outputs.

        Steps:
        1) Convert :class:`PointCloud2` → ``(N, 3)`` NumPy array.
        2) If this is the first frame, cache and return.
        3) Run ICP to estimate ``(R, t)`` aligning current → previous.
        4) Update the running pose ``(pose_R, pose_t)``.
        5) Publish the aligned cloud and :class:`nav_msgs.msg.Odometry`.

        Args:
            msg: Input filtered cloud (XYZ).
        """
        # Convert to numpy for processing
        pts = pointcloud2_to_numpy(msg)
        if pts.shape[0] == 0:
            return

        # First frame: just cache and wait for next
        if self.prev_pts is None:
            self.prev_pts = pts
            return

        # ICP against previous frame
        R, t = icp_point_to_point(
            pts,
            self.prev_pts,
            max_iters=int(self.get_parameter("max_iters").value),
        )

        # Integrate pose (compose transforms)
        self.pose_R = R @ self.pose_R
        self.pose_t = (self.pose_t @ R.T) + t

        # Aligned current cloud (to previous)
        aligned = (pts @ R.T) + t
        try:
            self.pub_aligned.publish(
                numpy_to_pointcloud2(aligned, frame_id=self.get_parameter("frame_id").value)
            )
        except Exception as e:
            self.get_logger().error(f"aligned publish error: {e}")

        # Publish odometry (odom -> base_link)
        q, trans = R_t_to_quat_trans(self.pose_R, self.pose_t)
        odom = Odometry()
        odom.header.stamp = msg.header.stamp
        odom.header.frame_id = self.get_parameter("odom_frame_id").value
        odom.child_frame_id = self.get_parameter("frame_id").value
        odom.pose.pose.position.x = float(trans[0])
        odom.pose.pose.position.y = float(trans[1])
        odom.pose.pose.position.z = float(trans[2])
        (
            odom.pose.pose.orientation.x,
            odom.pose.pose.orientation.y,
            odom.pose.pose.orientation.z,
            odom.pose.pose.orientation.w,
        ) = [float(x) for x in q]
        self.pub_odom.publish(odom)

        # Slide window
        self.prev_pts = pts


def main() -> None:
    """Entry point for the ``icp_node`` executable.

    Initializes rclpy, creates :class:`ICPNode`, spins until shutdown, and
    ensures a clean teardown on exit.
    """
    try:
        rclpy.init()
        node = ICPNode()
        node.get_logger().info("ICPNode is running. Press Ctrl+C to exit.")
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[ICPNode] Interrupted by user. Exiting...")
    except Exception as e:
        print(f"[ICPNode] Unexpected error: {e}")
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()
        print("[ICPNode] Shutdown complete.")
