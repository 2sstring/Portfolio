from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='pv_ai',
            executable='pv_forecast_node',
            name='pv_forecast_node',
            output='screen',
        ),
    ])
