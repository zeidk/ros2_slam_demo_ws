
from setuptools import setup
package_name = 'slam_preprocessing'
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
    description='LiDAR preprocessing: filtering & deskewing.',
    license='MIT',
    entry_points={'console_scripts': ['preprocess_node = slam_preprocessing.preprocess_node:main']},
)
