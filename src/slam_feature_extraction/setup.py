
from setuptools import setup
package_name = 'slam_feature_extraction'
setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'params.yaml']),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='You',
    maintainer_email='zeidk@umd.edu',
    description='Feature extraction from filtered LiDAR scans.',
    license='MIT',
    entry_points={'console_scripts': ['feature_node = slam_feature_extraction.feature_node:main']},
)
