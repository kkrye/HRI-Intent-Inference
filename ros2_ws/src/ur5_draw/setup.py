import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'ur5_draw'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        (os.path.join('share', package_name), glob(f'{package_name}/launch/*launch.py')),
        # (os.path.join('lib', package_name), glob(f'{package_name}/scripts/*.py')),
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='jmirenzi717@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'draw = ur5_draw.draw_node:main',
            'action = ur5_draw.draw_action:main',
            'test = ur5_draw.test_action_client:main',
        ],
    },
)
