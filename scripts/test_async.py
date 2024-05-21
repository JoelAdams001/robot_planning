#!/usr/bin/env python3

from typing import Optional

import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
import bdai_ros2_wrappers.process as ros_process
import bdai_ros2_wrappers.scope as ros_scope
from bdai_ros2_wrappers.action_client import ActionClientWrapper
from bdai_ros2_wrappers.action_handle import ActionHandle

from bosdyn.api import geometry_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME
from bosdyn.client.robot_command import RobotCommandBuilder
from spot_examples.simple_spot_commander import SimpleSpotCommander
from bdai_ros2_wrappers.tf_listener_wrapper import TFListenerWrapper

from bosdyn_msgs.conversions import convert
from spot_msgs.action import RobotCommand  # type: ignore
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion, Vector3Stamped
import time
import threading
import math

#/status/end_effector_force

class AsyncTest(Node):
    def __init__(self):
        super().__init__("async_test")
        self.spot_node_ = ros_scope.node()
        if self.spot_node_ is None:
                raise ValueError("no ROS 2 node available (did you use bdai_ros2_wrapper.process.main?)")
        self.force_sub = self.create_subscription(Vector3Stamped, "/status/end_effector_force", self.force_callback, 10)
        self.force = Vector3Stamped()
        self.test_thread = threading.Thread(target=self.test)
        self.test_thread.start()

    def force_callback(self, force_msg):
         self.force = force_msg
    
    def feedback_callback(self, feedback_msg):
        response = feedback_msg.feedback
        self.trajectory_status = (
            response.feedback.command.synchronized_feedback.arm_command_feedback.feedback.arm_cartesian_feedback.measured_pos_distance_to_goal
        )
        self.vec_mag = math.sqrt(self.force.vector.x**2 + self.force.vector.y**2 + self.force.vector.z**2)
        
    def test(self):
        tf_listener = TFListenerWrapper(self.spot_node_)
        tf_listener.wait_for_a_tform_b("odom", GRAV_ALIGNED_BODY_FRAME_NAME)
        hand_T_grav_body = tf_listener.lookup_a_tform_b("odom", GRAV_ALIGNED_BODY_FRAME_NAME)
        hand_T_grav_body_se3 = math_helpers.SE3Pose(
            hand_T_grav_body.transform.translation.x,
            hand_T_grav_body.transform.translation.y,
            hand_T_grav_body.transform.translation.z,
            math_helpers.Quat(
                hand_T_grav_body.transform.rotation.w,
                hand_T_grav_body.transform.rotation.x,
                hand_T_grav_body.transform.rotation.y,
                hand_T_grav_body.transform.rotation.z,
            ),
        )

        robot = SimpleSpotCommander(node=self.spot_node_)
        robot_command_client = ActionClientWrapper(RobotCommand, "robot_command", self.spot_node_)

        #Claim robot
        self.get_logger().info("Claiming robot")
        result = robot.command("claim")
        if not result.success:
            self.get_logger().error("Unable to claim robot message was " + result.message)
            return False
        self.get_logger().info("Claimed robot")

    
        # Make the arm pose RobotCommand
        # Build a position to move the arm to (in meters, relative to and expressed in the gravity aligned body frame).
        hand_pos = geometry_pb2.Vec3(x=0.75, y=0, z=0.25)

        # Rotation as a quaternion
        qw = 0.707
        qx = 0.707
        qy = 0
        qz = 0
        hand_Q = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)

        hand_T_body = geometry_pb2.SE3Pose(position=hand_pos, rotation=hand_Q)

        hand_T_odom = hand_T_grav_body_se3 * math_helpers.SE3Pose.from_proto(hand_T_body)

        # duration in seconds
        seconds = 10

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

        #command = RobotCommandBuilder.build_synchro_command(arm_command)

        # Convert to a ROS message
        action_goal = RobotCommand.Goal()
        convert(arm_command, action_goal.command)
        action_handle = ActionHandle(RobotCommand)
        action_handle.set_feedback_callback(self.feedback_callback)
        send_goal_future = robot_command_client.send_goal_async(action_goal, feedback_callback=action_handle.get_feedback_callback)
        self.trajectory_status = 100.0
        self.vec_mag = 0.0
        while (self.trajectory_status > 0.001):
             if self.vec_mag > 40.0:
                  print("Maximum force exceeded!")
                  break
     
        #Stow arm
        action_goal = RobotCommand.Goal()
        command = RobotCommandBuilder.arm_stow_command()
        convert(command, action_goal.command)
        robot_command_client.send_goal_and_wait("stow arm", action_goal)

        tf_listener.shutdown()

        return True

@ros_process.main()
def main(args=None):
    aync_test = AsyncTest()
    rclpy.spin(aync_test)
    aync_test.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()