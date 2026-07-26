import os
import sys
import glob
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # Configurar variables de entorno para aceleración EGL / NVIDIA GPU en Ignition Gazebo
    os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
    os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
    
    egl_json = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
    if os.path.exists(egl_json):
        os.environ['__EGL_VENDOR_LIBRARY_FILENAMES'] = egl_json

    # Detección automática de OGRE2_RESOURCE_PATH
    ogre2_path = '/usr/lib/x86_64-linux-gnu/OGRE-2.3.0'
    if not os.path.exists(ogre2_path):
        matches = glob.glob('/usr/lib/**/OGRE-2.*', recursive=True)
        if matches:
            ogre2_path = matches[0]
    if os.path.exists(ogre2_path):
        os.environ['OGRE2_RESOURCE_PATH'] = ogre2_path

    # Resolve URDF / world relative to this launch file's location.
    launch_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_share = os.path.dirname(launch_dir)
    urdf_file = os.path.join(pkg_share, 'urdf', 'racer_robot.urdf')

    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    # Reemplazar rutas de package:// en el URDF por rutas absolutas locales
    meshes_dir = os.path.join(pkg_share, 'meshes')
    robot_description = robot_description.replace('package://racer_description/meshes/', meshes_dir + '/')

    # Patch world SDF files to use absolute mesh paths at runtime.
    import tempfile
    worlds_dir = os.path.join(pkg_share, 'worlds')

    def get_patched_world_path(world_name):
        world_file = os.path.join(worlds_dir, world_name + '.sdf')
        with open(world_file, 'r') as f:
            content = f.read()
        content = content.replace('../meshes/', meshes_dir + '/')
        tmp = tempfile.NamedTemporaryFile(
            mode='w', suffix='.sdf', delete=False, prefix='gz_world_'
        )
        tmp.write(content)
        tmp.flush()
        return tmp.name

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

    world_name_arg = None
    for i, a in enumerate(sys.argv):
        if a.startswith('world:='):
            world_name_arg = a.split(':=', 1)[1]
            break
    if world_name_arg is None:
        world_name_arg = 'camera_world'

    headless_val = False
    for a in sys.argv:
        if a == 'headless:=true':
            headless_val = True
            break

    patched_world = get_patched_world_path(world_name_arg)
    if headless_val:
        gz_args_str = f'-s -r {patched_world}'
    else:
        gz_args_str = f'-r {patched_world}'

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch', 'gz_sim.launch.py'
            )
        ),
        launch_arguments={'gz_args': gz_args_str}.items()
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
