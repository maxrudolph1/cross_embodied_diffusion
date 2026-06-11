import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node


def generate_launch_description():
    # Get package path
    pkg_share = get_package_share_directory(
        "left_sharpa_ha4"
    )  # <-- rename to your ROS2 package name

    # Launch arguments
    gui_arg = DeclareLaunchArgument(
        "gui",
        default_value="false",
        description="Flag to enable joint_state_publisher_gui",
    )

    # URDF file
    urdf_file = os.path.join(pkg_share, "left_sharpa_ha4_v2_1.urdf")

    # Nodes
    joint_state_publisher_node = Node(
        package="joint_state_publisher_gui",
        executable="joint_state_publisher_gui",
        name="joint_state_publisher_gui",
        condition=None,  # could use IfCondition(LaunchConfiguration('gui'))
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[{"robot_description": open(urdf_file).read()}],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", os.path.join(pkg_share, "config/sharpa.rviz")],
    )

    return LaunchDescription(
        [gui_arg, joint_state_publisher_node, robot_state_publisher_node, rviz_node]
    )
