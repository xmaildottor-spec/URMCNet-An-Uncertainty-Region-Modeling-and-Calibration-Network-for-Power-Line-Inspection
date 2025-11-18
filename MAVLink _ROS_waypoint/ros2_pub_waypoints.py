'''
PART 3 —— ROS2 Waypoint Publisher（Nav Stack）
'''

import json
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import numpy as np
import math

def load_waypoints(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

class WaypointPublisher(Node):
    def __init__(self, json_path="uav_waypoints.json"):
        super().__init__("waypoint_publisher")
        self.publisher_ = self.create_publisher(PoseStamped, "uav_waypoints", 10)
        self.waypoints = load_waypoints(json_path)
        self.index = 0
        self.timer = self.create_timer(0.5, self.timer_callback)

    def timer_callback(self):
        if self.index >= len(self.waypoints):
            return

        wp = self.waypoints[self.index]
        msg = PoseStamped()
        msg.header.frame_id = "map"

        # XYZ
        msg.pose.position.x = wp["x"]
        msg.pose.position.y = wp["y"]
        msg.pose.position.z = wp["z"]

        # yaw → quaternion
        yaw = wp["yaw"]
        qz = math.sin(yaw/2)
        qw = math.cos(yaw/2)
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw

        self.publisher_.publish(msg)
        print(f"Published waypoint {self.index}: {wp}")

        self.index += 1


def main():
    rclpy.init()
    node = WaypointPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
