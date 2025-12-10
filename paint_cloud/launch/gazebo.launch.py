import os
from ament_index_python.packages import get_package_share_directory # Import this
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share = get_package_share_directory('paint_cloud')
    sdf_file = os.path.join(pkg_share, 'worlds', 'realsense_world.sdf')
    rviz_config_file = os.path.join(pkg_share, 'rviz', 'paint_cloud.rviz')

    ign_gazebo = ExecuteProcess(
        cmd=['ign', 'gazebo', '-r', sdf_file],
        output='screen'
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/realsense/image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/realsense/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/realsense/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked',
            '/realsense/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
        ],
        output='screen'
    )

    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '1.5', '0', '1.5708', '0', 'world', 'realsense_camera/link/realsense_d435']
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen'
    )

    paint_cloud = Node(
        package='paint_cloud',
        executable='paint_cloud',
        output='screen'
    )

    return LaunchDescription([
        ign_gazebo,
        bridge,
        static_tf,
        paint_cloud,
        rviz
    ])
