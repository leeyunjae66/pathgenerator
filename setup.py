from setuptools import setup

package_name = 'path_planner3'

setup(
    name=package_name,
    version='0.0.2',  # package.xml과 통일
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy'],  # numpy 추가
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='ROS2 Bezier path planner using /clicked_point and /map with obstacle avoidance',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'path_generator = path_planner3.path_generator:main',
        ],
    },
)
