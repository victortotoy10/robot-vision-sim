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
            'vision_sim_node = sim_vision_test.vision_sim_node:main',
            'data_recorder_node = sim_vision_test.data_recorder:main',
            'neural_pilot_node = sim_vision_test.neural_pilot_node:main',
            'evolutionary_trainer = sim_vision_test.evolutionary_trainer:main',
            'evolutionary_pilot = sim_vision_test.evolutionary_pilot:main',
            'train_sb3 = sim_vision_test.train_sb3:main',
            'sb3_pilot = sim_vision_test.sb3_pilot:main'
        ],
    },
)
