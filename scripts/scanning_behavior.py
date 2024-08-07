import rclpy
from rclpy.node import Node
import bdai_ros2_wrappers.process as ros_process
import bdai_ros2_wrappers.scope as ros_scope
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from spot_examples.simple_spot_commander import SimpleSpotCommander
from bosdyn.client.robot_command import RobotCommandBuilder
from bdai_ros2_wrappers.tf_listener_wrapper import TFListenerWrapper
from bosdyn_msgs.conversions import convert
from spot_msgs.action import RobotCommand
from bdai_ros2_wrappers.action_client import ActionClientWrapper
from bdai_ros2_wrappers.action_handle import ActionHandle
from bosdyn.api import geometry_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME
from geometry_msgs.msg import Pose, Point, Point32, Quaternion, Vector3Stamped
from robot_planning_interfaces.action import FindNextPoint
from robot_planning_interfaces.action import Kromek
from robot_planning_interfaces.srv import UpdateModel

import threading
from geometry_msgs.msg import Vector3Stamped
import math

class ScanningBehavior(Node):
    def __init__(self):
        super().__init__("scanning_behavior")
        # Parameters
        self.declare_parameter('frame_id', "map")
        self.declare_parameter('max_arm_force', 25.0)
        self.declare_parameter('num_samples', 22)
        self.declare_parameter('measurement_time', 5)
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        # Subscriber, action clients, and service clients
        self.force_sub = self.create_subscription(Vector3Stamped, "/status/end_effector_force", self.force_callback, 1)
        self.find_next_point_client = ActionClient(self, FindNextPoint, 'find_next_point_server')
        self.kromek_client = ActionClient(self, Kromek, 'kromek')
        self.update_model_client = self.create_client(UpdateModel, "update_model_server")

        self.spot_node_ = ros_scope.node()
        if self.spot_node_ is None:
                raise ValueError("no ROS 2 node available (did you use bdai_ros2_wrapper.process.main?)")
        
        # Ensure the action servers are available
        while not self.find_next_point_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Waiting for FindNextPoint action server...')
        while not self.kromek_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Waiting for Kromek action server...')

        #Initialize some important member variables
        self.force = Vector3Stamped()
        self.fb_time = self.get_clock().now()
        self.fail_count = 0

        #Set up some robot thigns
        self.robot = SimpleSpotCommander(node=self.spot_node_)
        self.robot_command_client = ActionClientWrapper(RobotCommand, "robot_command", self.spot_node_)
       
        self.begin()

    def begin(self):
         n = self.get_parameter('num_samples').get_parameter_value().integer_value
         self.get_logger().info(f'Number of samples: {n}')
         find_next_pt_goal = FindNextPoint.Goal()
         for i in range(n): #Sample iteratively requested number of points
             # Use find point action client to obtain next point to sample
             find_next_pt_goal.fail_count = self.fail_count
             self.find_next_point_client.wait_for_server()
             send_goal_future = self.find_next_point_client.send_goal_async(find_next_pt_goal)
             rclpy.spin_until_future_complete(self, send_goal_future)
             goal_handle = send_goal_future.result()

             if not goal_handle.accepted:
                  self.get_logger().info('Goal rejected :o')
                  self.find_next_point_client.destroy()
                  break
             
             self.get_logger().info('Goal accepted')
             get_result_future = goal_handle.get_result_async()
             rclpy.spin_until_future_complete(self, get_result_future)

             point = get_result_future.result().result.point
             status = get_result_future.result().status
             if status == GoalStatus.STATUS_SUCCEEDED:
                  self.get_logger().info('Goal succeeded!')
                  self.move_to_goal(point)
             else:
                  self.get_logger().info('Goal failed with status code: {0}'.format(status))
                  self.find_next_point_client.destroy()
                  self.destroy_node()
                  return
         self.stow_arm()
         self.find_next_point_client.destroy()
         self.destroy_node()
         return

    def force_callback(self, force_msg):
         self.force = force_msg
         self.vec_mag = math.sqrt(self.force.vector.x**2 + self.force.vector.y**2 + self.force.vector.z**2)
    
    def feedback_callback(self, feedback_msg):
        self.fb_time = self.get_clock().now()
        response = feedback_msg.feedback
        self.trajectory_status = (
            response.feedback.command.synchronized_feedback.arm_command_feedback.feedback.arm_cartesian_feedback.measured_pos_distance_to_goal
        )

    def move_to_goal(self, point):
        if not self.claim_robot():
            return False
        self.get_logger().info("move_to_goal!")
        tf_listener = TFListenerWrapper(self.spot_node_)
        tf_listener.wait_for_a_tform_b(ODOM_FRAME_NAME, "odom")
        action_goal = self.construct_pose_command(point, tf_listener)
        action_handle = ActionHandle(RobotCommand)
        action_handle.set_feedback_callback(self.feedback_callback)
        fb_start_time = self.get_clock().now()
        send_goal_future = self.robot_command_client.send_goal_async(action_goal, feedback_callback=action_handle.get_feedback_callback)
        self.trajectory_status = 100.0
        self.vec_mag = 0.0
        max_force = self.get_parameter('max_arm_force').get_parameter_value().double_value
        while (self.trajectory_status > 0.03):
            rclpy.spin_once(self, timeout_sec=0.1)
            #self.get_logger().info(f'Current force: {self.vec_mag}')
            if (self.vec_mag > max_force and (self.fb_time.nanoseconds - fb_start_time.nanoseconds) > 3e+9):
                print(self.vec_mag)
                self.get_logger().info("Max force exceeded, stowing arm and dropping current point")
                self.stow_arm()
                self.fail_count += 1
                return False
        self.fail_count = 0
            
        #Take measurement
        y = self.take_measurement()
        if not y:
            return False
        
        #Update model
        update_model_req = UpdateModel.Request()
        update_model_req.x = point
        update_model_req.y = float(y)
        while not self.update_model_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        future = self.update_model_client.call_async(update_model_req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result:
            self.get_logger().info('Updated model successfully')
    
    def stow_arm(self):
         if self.claim_robot():
            action_goal = RobotCommand.Goal()
            command = RobotCommandBuilder.arm_stow_command()
            convert(command, action_goal.command)
            self.get_logger().info("Stowing arm")
            self.robot_command_client.send_goal_and_wait("bayesian_optimization", action_goal)
    
    def claim_robot(self):
        #Claim robot
        self.get_logger().info("Claiming robot")
        result = self.robot.command("claim")
        if not result.success:
            self.get_logger().error("Unable to claim robot message was " + result.message)
            return False
        self.get_logger().info("Claimed robot")
        return True

    def construct_pose_command(self, point, tf_listener):
        #Get tranform for arm
        hand_T_grav_body = tf_listener.lookup_a_tform_b(ODOM_FRAME_NAME, "odom")
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
        pose = Pose()
        pose.position = point
        pose.orientation = Quaternion(x = 0.0 , y = 0.707 , z = 0.0 , w = 0.707)
        x, y, z = pose.position.x, pose.position.y, -0.05
        hand_pos = geometry_pb2.Vec3(x=x, y=y, z=z)

        # Rotation as a quaternion
        qw = pose.orientation.w
        qx = pose.orientation.x
        qy = pose.orientation.y
        qz = pose.orientation.z
        hand_Q = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)

        hand_T_body = geometry_pb2.SE3Pose(position=hand_pos, rotation=hand_Q)

        hand_T_odom = hand_T_grav_body_se3 * math_helpers.SE3Pose.from_proto(hand_T_body)

        # duration in seconds
        seconds = 7

        arm_command = RobotCommandBuilder.arm_pose_command(
            hand_T_odom.x,
            hand_T_odom.y,
            hand_T_odom.z,
            hand_T_odom.rot.w,
            hand_T_odom.rot.x,
            hand_T_odom.rot.y,
            hand_T_odom.rot.z,
            ODOM_FRAME_NAME,
            seconds
        )

        follow_arm_command = RobotCommandBuilder.follow_arm_command()
        command = RobotCommandBuilder.build_synchro_command(follow_arm_command, arm_command)

        # Convert to a ROS message
        action_goal = RobotCommand.Goal()
        convert(command, action_goal.command)
        return action_goal
    
    def take_measurement(self):
        # Action call to kromek_ros' action server
        time = self.get_parameter('measurement_time').get_parameter_value().integer_value
        kromek_pt_goal = Kromek.Goal()
        kromek_pt_goal.time = time
        self.kromek_client.wait_for_server()
        send_goal_future = self.kromek_client.send_goal_async(kromek_pt_goal)
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :o')
            self.kromek_client.destroy()
            return
        self.get_logger().info('Goal accepted')
        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)
        counts_per_sec = get_result_future.result().result.count
        status = get_result_future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Goal succeeded!')
            return counts_per_sec
        else:
            self.get_logger().info('Goal failed with status code: {0}'.format(status))
            self.kromek_client.destroy()
            self.destroy_node()
            rclpy.shutdown()
            return False

@ros_process.main()
def main(args=None):
    scanning_behavior = ScanningBehavior()
    rclpy.spin(scanning_behavior)
    scanning_behavior.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()