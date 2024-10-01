from setuptools import find_packages, setup

package_name = 'gesture_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'opencv-python', 'mediapipe', 'numpy', 'scipy'],
    zip_safe=True,
    maintainer='om',
    maintainer_email='stillwake27@gmail.com',
    description='Gesture control for robot using hand gestures',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gesture_publisher = gesture_control.gesture_publisher:main',  # Add this line
        ],
    },
)
