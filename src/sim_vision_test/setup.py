from setuptools import find_packages, setup

package_name = 'sim_vision_test'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='akenitoy',
    maintainer_email='akenitoy@todo.todo',
    description='Paquete de procesamiento de vision con OpenCV en simulacion para ROS 2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_sim_node = sim_vision_test.vision_sim_node:main'
        ],
    },
)
