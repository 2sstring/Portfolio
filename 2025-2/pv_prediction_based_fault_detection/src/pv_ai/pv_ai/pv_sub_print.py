#!/usr/bin/env python3
import json, rclpy
from rclpy.node import Node
from std_msgs.msg import String

class PVSubPrint(Node):
    def __init__(self):
        super().__init__('pv_sub_print')
        self.sub = self.create_subscription(String, '/pv/sample', self.cb, 10)

    def cb(self, msg: String):
        try:
            d = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warning(f'JSON parse fail: {e}')
            return
        p = d.get('ac_power') or d.get('power')
        self.get_logger().info(
            f"{d.get('ts')} INV={d.get('inverter_id','INV1')} "
            f"P={p} DHI={d.get('dhi')} DNI={d.get('dni')} "
            f"WS={d.get('ws')} RH={d.get('rh')} status={d.get('status')}"
        )

def main():
    rclpy.init(); rclpy.spin(PVSubPrint()); rclpy.shutdown()

if __name__ == '__main__':
    main()
