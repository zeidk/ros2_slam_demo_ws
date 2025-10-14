
from setuptools import setup
package_name = 'slam_scan_matching'
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
    maintainer_email='you@example.com',
    description='Scan matching (ICP) and odometry publisher.',
    license='MIT',
    entry_points={'console_scripts': ['icp_node = slam_scan_matching.icp_node:main']},
)
