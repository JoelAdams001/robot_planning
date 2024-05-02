from launch_ros.substitutions import FindPackageShare

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node

def generate_launch_description():
    parameters=[{
          'frame_id':'body',
          'voxel_size':0.2,
          'clip_distance':1.1,
          'cloud_topic':"/depth_registered/frontleft/points"}
          ]
    
    return LaunchDescription([
        Node(
            package='robot_planning', executable='crop_scan_section', output='screen',
            parameters=parameters,
            arguments=[])
    ])

