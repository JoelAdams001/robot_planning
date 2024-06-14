import rclpy
import bdai_ros2_wrappers.process as ros_process
import bdai_ros2_wrappers.scope as ros_scope
from rclpy.action import ActionClient
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
from robot_planning_interfaces.action import FindNextPoint
from robot_planning_interfaces.srv import UpdateModel

from geometry_msgs.msg import Vector3Stamped
import math

class ScanningBehavior(rclpy.node.Node):
    def __init__(self):
        super().__init__("scanning_behavior")
        # Parameters
        self.declare_parameter('frame_id', "map")
        self.declare_parameter('max_arm_force', 30.0)
        self.declare_parameter('num_samples', 10)
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        # Subscriber and action client
        self.force_sub = self.create_subscription(Vector3Stamped, "/status/end_effector_force", self.force_callback, 1)
        self.find_next_point_client = ActionClient(self, FindNextPoint, 'find_next_point')

        self.spot_node_ = ros_scope.node()
        if self.spot_node_ is None:
                raise ValueError("no ROS 2 node available (did you use bdai_ros2_wrapper.process.main?)")

        #Initialize some important member variables
        self.force = Vector3Stamped()
        self.fb_time = self.get_clock().now()
        self.fail_count = 0

        #self.timer = self.create_timer(1.0, self.timer_callback)

        #Set up some robot thigns
        self.robot = SimpleSpotCommander(node=self.spot_node_)
        self.robot_command_client = ActionClientWrapper(RobotCommand, "robot_command", self.spot_node_)
        self.tf_listener = TFListenerWrapper(self.spot_node_)

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
        #Get point through service call
        #goal_msg = FindNextPoint.Goal(self.fail_count)
        #self.find_next_point_client.wait_for_server()
        #self.send_goal_future = self.find_next_point_client.send_goal_async(goal_msg)

        self.claim_robot()
        #Get tranform for arm
        hand_T_grav_body = self.tf_listener.lookup_a_tform_b(ODOM_FRAME_NAME, "odom")
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
    
    def stow_arm(self):
         self.claim_robot()
    
    def claim_robot(self):
        #Claim robot
        self.get_logger().info("Claiming robot")
        result = self.robot.command("claim")
        if not result.success:
            self.get_logger().error("Unable to claim robot message was " + result.message)
            return False
        self.get_logger().info("Claimed robot")

@ros_process.main()
def main(args=None):
    scanning_behavior = ScanningBehavior()
    rclpy.spin(scanning_behavior)
    scanning_behavior.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()