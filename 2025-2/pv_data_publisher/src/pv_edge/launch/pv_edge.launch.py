from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='pv_edge', executable='pv_simulator', name='pv_simulator',
            output='screen', parameters=[{'plant_id':'PLANT_A'},{'inverter':'INV01'},{'period_sec':1.0}]),
        Node(package='pv_edge', executable='pv_logger', name='pv_logger',
            output='screen', parameters=[{'csv_path':'pv_sample.csv'}, {'db_path':'pv_sample.sqlite'}]),
        ])
