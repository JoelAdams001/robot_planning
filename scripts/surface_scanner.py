#!/usr/bin/env python3

from typing import Optional

import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from bdai_ros2_wrappers.action_client import ActionClientWrapper
from bosdyn.api import geometry_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME
from bosdyn.client.robot_command import RobotCommandBuilder
from utilities.simple_spot_commander import SimpleSpotCommander
from utilities.tf_listener_wrapper import TFListenerWrapper

import spot_driver.conversions as conv
from spot_msgs.action import RobotCommand  # type: ignore
from sensor_msgs.msg import PointCloud2
import struct

class SurfaceScanner(Node):
    def __init__(self):
        super().__init__("surface_scanner")
        self.declare_parameter('cloud_topic', 'scan_section')
        cloud_topic = self.get_parameter('cloud_topic').get_parameter_value().string_value
        self.subscription = self.create_subscription(
            PointCloud2,
            cloud_topic,
            self.scan,
            qos_profile_sensor_data
        )

    def scan(self, msg):
        tf_listener = TFListenerWrapper(
        "surface_scanner_tf", wait_for_transform=[ODOM_FRAME_NAME, "body"]
        )

        robot = SimpleSpotCommander()
        robot_command_client = ActionClientWrapper(RobotCommand, "robot_command", "surface_scanner_action_node")

        # Claim robot
        self.get_logger().info("Claiming robot")
        result = robot.command("claim")
        if not result.success:
            self.get_logger().error("Unable to claim robot message was " + result.message)
            return False
        self.get_logger().info("Claimed robot")

        point_step = msg.point_step
        format_str = 'fff'
        for i in range(0, len(msg.data), point_step):
            point = struct.unpack_from(format_str, msg.data, offset=i)
            # Make the arm pose RobotCommand
            # Build a position to move the arm to (in meters, relative to and expressed in the gravity aligned body frame).
            x, y, z = point[0], point[1], point[2]
            hand_pos = geometry_pb2.Vec3(x=x, y=y, z=z+0.3)

            # Rotation as a quaternion
            qw = 0.707
            qx = 0
            qy = 0.707
            qz = 0
            hand_Q = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)

            hand_T_body = geometry_pb2.SE3Pose(position=hand_pos, rotation=hand_Q)

            hand_T_grav_body = tf_listener.lookup_a_tform_b(ODOM_FRAME_NAME, "body")

            hand_T_odom = hand_T_grav_body * math_helpers.SE3Pose.from_obj(hand_T_body)

            # duration in seconds
            seconds = 5

            arm_command = RobotCommandBuilder.arm_pose_command(
                hand_T_odom.x,
                hand_T_odom.y,
                hand_T_odom.z,
                hand_T_odom.rot.w,
                hand_T_odom.rot.x,
                hand_T_odom.rot.y,
                hand_T_odom.rot.z,
                ODOM_FRAME_NAME,
                seconds,
            )

            follow_arm_command = RobotCommandBuilder.follow_arm_command()
            command = RobotCommandBuilder.build_synchro_command(follow_arm_command, arm_command)


            # Convert to a ROS message
            action_goal = RobotCommand.Goal()
            conv.convert_proto_to_bosdyn_msgs_robot_command(command, action_goal.command)
            # Send the request and wait until the arm arrives at the goal
            self.get_logger().info("Moving arm to position 1.")
            robot_command_client.send_goal_and_wait("arm_move_one", action_goal)
     

        #Stow arm
        action_goal = RobotCommand.Goal()
        command = RobotCommandBuilder.arm_stow_command()
        conv.convert_proto_to_bosdyn_msgs_robot_command(command, action_goal.command)
        self.get_logger().info("Stowing arm")
        robot_command_client.send_goal_and_wait("stow arm", action_goal)

        tf_listener.shutdown()

        return True


def main(args=None):
    rclpy.init(args=args)
    surface_scanner = SurfaceScanner()
    rclpy.spin(surface_scanner)
    surface_scanner.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
