
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Use installed parameter files if available; fall back to package share.
    pp = os.path.join(get_package_share_directory('slam_preprocessing'), 'params.yaml')
    fe = os.path.join(get_package_share_directory('slam_feature_extraction'), 'params.yaml')
    icp = os.path.join(get_package_share_directory('slam_scan_matching'), 'params.yaml')
    return LaunchDescription([
        Node(package='slam_preprocessing', executable='preprocess_node', name='preprocess',
             parameters=[pp], output='screen'),
        Node(package='slam_feature_extraction', executable='feature_node', name='features',
             parameters=[fe], output='screen'),
        Node(package='slam_scan_matching', executable='icp_node', name='icp',
             parameters=[icp], output='screen'),
    ])
