#!/usr/bin/env python3

import argparse
from typing import Optional

import rclpy
from rclpy.action import ActionServer
from bdai_ros2_wrappers.action_client import ActionClientWrapper
from bosdyn.api import geometry_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME
from bosdyn.client.robot_command import RobotCommandBuilder
from rclpy.node import Node
from utilities.simple_spot_commander import SimpleSpotCommander
from utilities.tf_listener_wrapper import TFListenerWrapper

import spot_driver.conversions as conv
from spot_msgs.action import RobotCommand  # type: ignore

from robot_planning_interfaces.action import Unstow


class Unstow(Node):

    def __init__(self):
        super().__init__('unstow_action_server')
        self._action_server = ActionServer(
            self,
            Unstow,
            'unstow',
            self.execute_callback)
        
    def execute_callback(self, goal_handle):
        self.get_logger().info('Unstowing Arm')
        result = Unstow.Result()
        return result
    
    # Set up basic ROS2 utilities for communicating with the driver
    node = Node("unstow_arm")
    name = ""
    namespace = ""
    tf_listener = TFListenerWrapper(
        "arm_simple_tf", wait_for_transform=[name + ODOM_FRAME_NAME, name + GRAV_ALIGNED_BODY_FRAME_NAME]
    )

    robot = SimpleSpotCommander(namespace)
    robot_command_client = ActionClientWrapper(
        RobotCommand, "robot_command", "arm_simple_action_node", namespace=namespace
    )

    # Claim robot
    node.get_logger().info("Claiming robot")
    result = robot.command("claim")
    if not result.success:
        node.get_logger().error("Unable to claim robot message was " + result.message)
        return False
    node.get_logger().info("Claimed robot")

    # Stand the robot up.
    node.get_logger().info("Powering robot on")
    result = robot.command("power_on")
    if not result.success:
        node.get_logger().error("Unable to power on robot message was " + result.message)
        return False
    node.get_logger().info("Standing robot up")
    result = robot.command("stand")
    if not result.success:
        node.get_logger().error("Robot did not stand message was " + result.message)
        return False
    node.get_logger().info("Successfully stood up.")

    # Move the arm to a spot in front of the robot, and open the gripper.

    # Make the arm pose RobotCommand
    # Build a position to move the arm to (in meters, relative to and expressed in the gravity aligned body frame).
    x = 0.75
    y = 0
    z = 0.25
    hand_ewrt_flat_body = geometry_pb2.Vec3(x=x, y=y, z=z)

    # Rotation as a quaternion
    qw = 1
    qx = 0
    qy = 0
    qz = 0
    flat_body_Q_hand = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)

    flat_body_T_hand = geometry_pb2.SE3Pose(position=hand_ewrt_flat_body, rotation=flat_body_Q_hand)

    odom_T_flat_body = tf_listener.lookup_a_tform_b(name + ODOM_FRAME_NAME, name + GRAV_ALIGNED_BODY_FRAME_NAME)

    odom_T_hand = odom_T_flat_body * math_helpers.SE3Pose.from_obj(flat_body_T_hand)

    # duration in seconds
    seconds = 2

    arm_command = RobotCommandBuilder.arm_pose_command(
        odom_T_hand.x,
        odom_T_hand.y,
        odom_T_hand.z,
        odom_T_hand.rot.w,
        odom_T_hand.rot.x,
        odom_T_hand.rot.y,
        odom_T_hand.rot.z,
        ODOM_FRAME_NAME,
        seconds,
    )

    # Make the open gripper RobotCommand
    gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)

    # Combine the arm and gripper commands into one RobotCommand
    command = RobotCommandBuilder.build_synchro_command(gripper_command, arm_command)

    # Convert to a ROS message
    action_goal = RobotCommand.Goal()
    conv.convert_proto_to_bosdyn_msgs_robot_command(command, action_goal.command)
    # Send the request and wait until the arm arrives at the goal
    node.get_logger().info("Moving arm to position 1.")
    robot_command_client.send_goal_and_wait("arm_move_one", action_goal)

    # #Stow arm
    # action_goal = RobotCommand.Goal()
    # command = RobotCommandBuilder.arm_stow_command()
    # conv.convert_proto_to_bosdyn_msgs_robot_command(command, action_goal.command)
    # node.get_logger().info("Stowing arm")
    # robot_command_client.send_goal_and_wait("stow arm", action_goal)

    tf_listener.shutdown()

    return True


def main() -> None:
    rclpy.init()
    unstow_action_server = Unstow()
    rclpy.spin(unstow_action_server)

if __name__ == "__main__":
    main()
