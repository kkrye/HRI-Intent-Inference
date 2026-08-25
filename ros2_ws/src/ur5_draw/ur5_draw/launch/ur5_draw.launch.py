from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch.substitutions import PathJoinSubstitution, Command, FindExecutable
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def get_robot_description(ur_type,robot_ip):
    joint_limit_params = PathJoinSubstitution(
        [FindPackageShare("ur_description"), "config",ur_type, "joint_limits.yaml"]
    )
    kinematics_params = PathJoinSubstitution(
        [FindPackageShare("ur_description"), "config", ur_type, "default_kinematics.yaml"]
    )
    physical_params = PathJoinSubstitution(
        [FindPackageShare("ur_description"), "config", ur_type, "physical_parameters.yaml"]
    )
    visual_params = PathJoinSubstitution(
        [FindPackageShare("ur_description"), "config", ur_type, "visual_parameters.yaml"]
    )
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("ur_description"), "urdf", "ur.urdf.xacro"]),
            " ",
            "robot_ip:=",
            robot_ip,
            " ",
            "joint_limit_params:=",
            joint_limit_params,
            " ",
            "kinematics_params:=",
            kinematics_params,
            " ",
            "physical_params:=",
            physical_params,
            " ",
            "visual_params:=",
            visual_params,
            " ",
           "safety_limits:=",
            "true",
            " ",
            "safety_pos_margin:=",
            "0.15",
            " ",
            "safety_k_position:=",
            "20",
            " ",
            "name:=",
            "ur",
            " ",
            "ur_type:=",
            ur_type,
            " ",
            "prefix:=",
            '""',
            " ",
        ]
    )


    robot_description = {"robot_description": robot_description_content}
    return robot_description

def get_robot_description_semantic():
    # MoveIt Configuration
    robot_description_semantic_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("ur_moveit_config"), "srdf", "ur.srdf.xacro"]),
            " ",
            "name:=",
            # Also ur_type parameter could be used but then the planning group names in yaml
            # configs has to be updated!
            "ur",
            " ",
            "prefix:=",
            '""',
            " ",
        ]
    )
    robot_description_semantic = {
        "robot_description_semantic": robot_description_semantic_content
    }
    return robot_description_semantic


def generate_launch_description():
    # 1. Launch Arguments
    enable_sim = LaunchConfiguration('enable_sim')
    robot_ip = LaunchConfiguration('robot_ip')
    ur_type = LaunchConfiguration('ur_type')

    # Argument to switch between real robot and simulation
    enable_sim_arg = DeclareLaunchArgument(
        'enable_sim',
        default_value='false',
        description='Set to true to launch in simulation mode (using fake hardware) or false for real robot.',
    )

    # Robot IP (only needed when enable_sim is false)
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        default_value='192.168.56.50',
        description='The IP address of the real UR robot (only used if enable_sim is false).',
    )

    # UR Type (used for both)
    ur_type_arg = DeclareLaunchArgument(
        'ur_type',
        default_value='ur5',
        description='The type of UR robot (e.g., ur5, ur10).',
    )


    # Find the UR Driver package share directory
    ur_driver_pkg = FindPackageShare('ur_robot_driver')

    # Define common arguments for the ur_control.launch.py
    # Note: The ur_robot_driver launch file handles both real and fake hardware
    # using 'use_fake_hardware' argument.
    ur_control_launch = PathJoinSubstitution([
        ur_driver_pkg,
        'launch',
        'ur_control.launch.py'
    ])

    # Conditional Launch for UR Driver (Real Hardware)
    # When enable_sim is false, use the provided robot_ip
    ur_driver_real = IncludeLaunchDescription(
        ur_control_launch,
        condition=UnlessCondition(enable_sim),
        launch_arguments={
            'ur_type': ur_type,
            'robot_ip': robot_ip,
            'reverse_ip': '192.168.56.101',
            'use_fake_hardware': 'false',
            # 'headless_mode': 'true',
            'use_sim_time': 'false', # Important for real robot
            'launch_rviz': 'false'
        }.items(),
    )

    # Conditional Launch for UR Driver (Simulation/Fake Hardware)
    # When enable_sim is true, set use_fake_hardware to true
    ur_sim_launch = PathJoinSubstitution([
        FindPackageShare('ur_simulation_gz'),
        'launch',
        'ur_sim_control.launch.py'
    ])
    ur_driver_sim = IncludeLaunchDescription(
            ur_sim_launch,
            condition=IfCondition(enable_sim),
            launch_arguments={
                'ur_type': ur_type,
                'gazebo_gui': 'false',
                # 'initial_joint_controller': 'joint_position_controller',
                'initial_joint_controller': 'joint_trajectory_controller',
                'launch_rviz': 'false',
            }.items(),
    )

    moveit_config = IncludeLaunchDescription(
        PathJoinSubstitution([ FindPackageShare('ur_moveit_config'), 'launch', 'ur_moveit.launch.py' ]),
        launch_arguments={
            'ur_type': ur_type,
            'use_sim_time': enable_sim,
            'launch_rviz': 'true',
        }.items(),
    )

    # metal cart
    static_table_frame_pub = Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name='static_table_frame_pub',
                arguments=[ 
                    '--x', '0.0', '--y', '0.0', '--z', '-0.108', # lower value to lower the table
                    '--yaw', '-0.785398', '--pitch', '0.0', '--roll',
                    '0', '--frame-id', 'world', '--child-frame-id', 'table_frame']
            )

    # image frame on cart
    static_img_frame_pub = Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name='static_img_frame_pub',
                arguments=[ 
                    '--x', '0.5', '--y', '-0.12', '--z', '0.014', # lower value brings pen closer to table
                    '--yaw', '0.0', '--pitch', '0', '--roll',
                    '0', '--frame-id', 'table_frame', '--child-frame-id', 'image_frame']
            )

    robot_description = get_robot_description(ur_type,robot_ip)
    robot_description_semantic = get_robot_description_semantic()

    moveit_service = Node(
        package="ur_draw_cmake",
        executable="moveit_service",
        name="moveit_service",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
        ],
    )

    draw_action = Node(
        package="ur5_draw",
        executable="action",
        name="draw_action_server",
        output="screen",
    )

    return LaunchDescription([
        enable_sim_arg,
        robot_ip_arg,
        ur_type_arg,
        static_table_frame_pub,
        static_img_frame_pub,
        moveit_service,
        draw_action,

        # Launches the appropriate UR driver setup
        ur_driver_real,
        ur_driver_sim,
        moveit_config,

    ])