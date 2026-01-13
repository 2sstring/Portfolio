from setuptools import find_packages, setup

package_name = 'pv_edge'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/pv_edge.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='so',
    maintainer_email='shiljcf@naver.com',
    description='PV edge nodes',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'pv_simulator = pv_edge.pv_simulator_node:main',
        ],
    },
)
