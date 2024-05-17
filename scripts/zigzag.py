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
from spot_examples.simple_spot_commander import SimpleSpotCommander
from bdai_ros2_wrappers.tf_listener_wrapper import TFListenerWrapper

import bosdyn_msgs.conversions as conv
from spot_msgs.action import RobotCommand  # type: ignore
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
import struct
import numpy as np

class ZigzagScanner(Node):
    def __init__(self):
        super().__init__("zigzag_scanner")
        self.declare_parameter('H', "2.432")
        self.declare_parameter('n', "2")
        self.declare_parameter('w', "0.35")
        self.declare_parameter('frame_id', "body")

        self.publisher_ = self.create_publisher(PoseArray, 'zigzag_marker', 10)
        #self.timer_ = self.create_timer(0.5, self.scan)
        self.scan()

    def create_array(H, n, w):
        if(not isinstance(n, int)):
            print("n must be an integer")
            return

        points = np.array([], dtype=np.float64).reshape(0, 2)
        x = 0
        y = 0
        h = H/n
        #First point
        x += .8
        points = np.vstack([points, [x, y]])
        for _ in range(0, n):
            #Middle
            for _ in range(0,2):
                y -= w/2
                points = np.vstack([points, [x,y]])
            for _ in range(0,2):
                x += h/2
                points = np.vstack([points, [x,y]])
            for _ in range(0,4):
                y += w/2
                points = np.vstack([points, [x,y]])
            for _ in range(0,2):
                x += h/2
                points = np.vstack([points, [x,y]])
            for _ in range(0,2):
                y -= w/2
                points = np.vstack([points, [x,y]])

        #Last 2
        for _ in range(0, 2):
            x += h/2
            points = np.vstack([points, [x, y]])
        return points

    def scan(self):
        H = self.get_parameter('H').get_parameter_value().double_value
        n = self.get_parameter('n').get_parameter_value().integer_value
        w = self.get_parameter('w').get_parameter_value().double_value
        frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        tf_listener = TFListenerWrapper(
        "zigzag_scanner_tf", wait_for_transform=[ODOM_FRAME_NAME, "body"]
        )
        hand_T_grav_body = tf_listener.lookup_a_tform_b(ODOM_FRAME_NAME, "body")

        robot = SimpleSpotCommander()
        robot_command_client = ActionClientWrapper(RobotCommand, "robot_command", "surface_scanner_action_node")

        #Claim robot
        self.get_logger().info("Claiming robot")
        result = robot.command("claim")
        if not result.success:
            self.get_logger().error("Unable to claim robot message was " + result.message)
            return False
        self.get_logger().info("Claimed robot")

        points = ZigzagScanner.create_array(0.6, 1, 0.85)
        p_array = PoseArray()
        p_array.header.frame_id = frame_id
        for point in points:
            pose = Pose()
            pose.position = Point(x = point[0], y = point[1], z = -0.39)
            pose.orientation = Quaternion(x = 0.0 , y = 0.707 , z = 0.0 , w = 0.707)
            p_array.poses.append(pose)
        self.publisher_.publish(p_array)

        
        for pose in p_array.poses:
            # Make the arm pose RobotCommand
            # Build a position to move the arm to (in meters, relative to and expressed in the gravity aligned body frame).
            x, y, z = pose.position.x, pose.position.y, pose.position.z
            hand_pos = geometry_pb2.Vec3(x=x, y=y, z=z)

            # Rotation as a quaternion
            qw = pose.orientation.w
            qx = pose.orientation.x
            qy = pose.orientation.y
            qz = pose.orientation.z
            hand_Q = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)

            hand_T_body = geometry_pb2.SE3Pose(position=hand_pos, rotation=hand_Q)

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
    zigzag_scanner = ZigzagScanner()
    rclpy.spin(zigzag_scanner)
    zigzag_scanner.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()