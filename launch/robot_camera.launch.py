import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # Resolve URDF / world relative to this launch file's location.
    # Works both when running from the source tree (no colcon needed) and
    # after `colcon build` because the installed share/<pkg>/launch/ has
    # urdf/ and worlds/ as siblings — same layout as the source tree.
    launch_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_share = os.path.dirname(launch_dir)
    urdf_file = os.path.join(pkg_share, 'urdf', 'my_robot.urdf')

    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    # Launch argument for headless mode
    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='Ejecutar Gazebo en modo headless (servidor sin GUI 3D)'
    )

    # Launch argument for selecting the world file
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='camera_world',
        description='Nombre del mundo a cargar (camera_world, racetrack, racetrack_decorated)'
    )

    # Robot State Publisher — broadcasts TF from URDF
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description,
                     'use_sim_time': True}],
        output='screen'
    )

    # Gazebo with our selected world, optionally headless
    # If headless is 'true', we pass '-s -r <world>', otherwise just '-r <world>'
    gz_args = PythonExpression([
        "'-s -r ' + '" + pkg_share + "/worlds/' + '",
        LaunchConfiguration('world'),
        "' + '.sdf' if '",
        LaunchConfiguration('headless'),
        "' == 'true' else '-r ' + '" + pkg_share + "/worlds/' + '",
        LaunchConfiguration('world'),
        "' + '.sdf'"
    ])

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch', 'gz_sim.launch.py'
            )
        ),
        launch_arguments={'gz_args': gz_args}.items()
    )

    # Spawn the robot from /robot_description
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'my_robot',
            '-topic', '/robot_description',
            '-z', '0.1'
        ],
        output='screen'
    )

    # Bridge: Gazebo <-> ROS2
    # Maps cmd_vel, odom, joint_states, scan, camera/image_raw, camera_info, clock, and TF.
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry',
            '/joint_states@sensor_msgs/msg/JointState@gz.msgs.Model',
            '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            '/camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/model/my_robot/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
        ],
        remappings=[
            ('/model/my_robot/tf', '/tf'),
        ],
        output='screen'
    )

    return LaunchDescription([
        headless_arg,
        world_arg,
        robot_state_publisher,
        gazebo,
        spawn_robot,
        bridge,
    ])
