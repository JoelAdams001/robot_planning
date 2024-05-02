#!/usr/bin/env python3

import struct
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from geometry_msgs.msg import PolygonStamped, Pose, Point32, Quaternion

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
from scipy.stats import norm

from bdai_ros2_wrappers.action_client import ActionClientWrapper
from bosdyn.api import geometry_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME
from bosdyn.client.robot_command import RobotCommandBuilder
from utilities.simple_spot_commander import SimpleSpotCommander
from utilities.tf_listener_wrapper import TFListenerWrapper
import spot_driver.conversions as conv
from spot_msgs.action import RobotCommand  # type: ignore

class BayesianScanner(Node):
    def __init__(self):
        super().__init__("bayesian_scanning")
        self.declare_parameter('box_length', "2.5")
        self.declare_parameter('box_width', "2.5")
        self.declare_parameter('frame_id', "map")

        self.box_pub_ = self.create_publisher(PolygonStamped, 'bounding_box', 10)
        self.ground_truth_pub_ = self.create_publisher(PointCloud2, 'ground_truth', 10)
        self.box_timer_ = self.create_timer(0.5, self.bounding_box)
        self.ground_truth_timer_ = self.create_timer(0.5, self.ground_truth)
        self.samples_pub_ = self.create_publisher(PointCloud2, 'sampled_points', 10)
        self.scan()

    def bounding_box(self):
        frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        l = self.get_parameter('box_length').get_parameter_value().double_value
        w = self.get_parameter('box_width').get_parameter_value().double_value
        bounding_box = PolygonStamped()
        bounding_box.header = Header(stamp=self.get_clock().now().to_msg(), frame_id=frame_id)
        points = [
        Point32(x=0.3, y=-w, z=0.0),
        Point32(x=0.3 + l, y=-w, z=0.0),
        Point32(x=0.3 + l, y=w, z=0.0),
        Point32(x=0.3, y=w, z=0.0)
        ]

        bounding_box.polygon.points = points
        self.box_pub_.publish(bounding_box)

    def function(self, x):
        return (np.exp(-(x[0] - 2)**2 - (x[1] - 2)**2) +
                np.exp(-(x[0] - 4)**2 - (x[1] - 4)**2))
    
    def ground_truth(self):
        frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        # Plot the ground truth function
        x = np.linspace(0, 2.5, 300)
        y = np.linspace(0, 2.5, 300)
        X, Y = np.meshgrid(x, y)
        Z = np.array([self.function([xx, yy]) for xx, yy in zip(np.ravel(X), np.ravel(Y))])
        Z = Z.reshape(X.shape)
        tuples = np.dstack((X, Y, Z)).reshape(-1,3)
        
        pc2 = self.create_pointcloud(tuples, frame_id)
        self.ground_truth_pub_.publish(pc2)

    def create_pointcloud(tuples, frame_id):
        #Construct a ROS2 PointCloud2 message using a numpy array of tuples
        buf = []
        for pt in tuples:
            buf += struct.pack('fff', *pt)

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        
        pc2 = PointCloud2(
            header= Header(frame_id=frame_id),
            height=1,
            width=len(tuples),
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=12,  # Each point consists of three float32s, each float32 is 4 bytes
            row_step=12 * len(tuples),
            data=bytearray(buf)
        )

        return pc2

    def probability_of_improvement(self, X, gp, y_max, xi= 0.01):
        """Computes the PI at points X based on existing samples using a Gaussian process surrogate model."""
        mu, sigma = gp.predict(X, return_std=True)
        with np.errstate(divide='warn'):
            # Improvement threshold, adjust xi to trade-off exploration vs. exploitation.
            imp_thresh = y_max + xi
            Z = (mu - imp_thresh) / sigma
            if sigma == 0.0:
                return 0.0
            return norm.cdf(Z)

    def select_next(self, acquisition, gp, bounds, y_max, n_restarts=25):
        dim = gp.kernel_.n_dims
        min_val = 1
        min_x = None

        def min_obj(X):
            """Minimization objective is the negative acquisition function"""
            return -acquisition(X.reshape(-1, dim), gp=gp, y_max=y_max)
        
        # Find the best optimum by starting from n_restart different random points.
        for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
            res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun
                min_x = res.x
                
        return min_x


    def scan(self):
        frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        l = self.get_parameter('box_length').get_parameter_value().double_value
        w = self.get_parameter('box_width').get_parameter_value().double_value

        # Create Gaussian Process Surrogate
        kernel = C(1.0, (1e-4, 1e1)) * RBF([1.0, 1.0], (1e-4, 1e1))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

        #Sample first point
        #Note here x_sample refers to the array of coords (x,y), y_sample is array of rad readings
        X_sample = np.array([[0.3, 0.3]])
        rad_reading = self.function(X_sample)
        y_sample = np.array([[rad_reading]])
        gp.fit(X_sample, y_sample)
        y_max = rad_reading

        for _ in range(10):
            x_next = self.select_next(self.probability_of_improvement, gp, [(-w, w) , (-l,l)], y_max)
            y_next = self.function(x_next)
            if y_next > y_max:
               y_max = y_next
            X_sample = np.vstack((X_sample, x_next))
            y_sample = np.append(y_sample, y_next)
            gp.fit(X_sample, y_sample)

            tuples = [(x[0], x[1], y) for x, y in zip(X_sample, y_sample)]
            pc2 = self.create_pointcloud(tuples, frame_id)
            self.samples_pub_.publish(pc2)

        # frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        # tf_listener = TFListenerWrapper(
        # "zigzag_scanner_tf", wait_for_transform=[ODOM_FRAME_NAME, "body"]
        # )
        # hand_T_grav_body = tf_listener.lookup_a_tform_b(ODOM_FRAME_NAME, "body")

        # robot = SimpleSpotCommander()
        # robot_command_client = ActionClientWrapper(RobotCommand, "robot_command", "surface_scanner_action_node")

        # #Claim robot
        # self.get_logger().info("Claiming robot")
        # result = robot.command("claim")
        # if not result.success:
        #     self.get_logger().error("Unable to claim robot message was " + result.message)
        #     return False
        # self.get_logger().info("Claimed robot")

        # points = ZigzagScanner.create_array(0.8, 2, 0.35)
        # p_array = PoseArray()
        # p_array.header.frame_id = frame_id
        # for point in points:
        #     pose = Pose()
        #     pose.position = Point(x = point[0], y = point[1], z = -0.2)
        #     pose.orientation = Quaternion(x = 0.0 , y = 0.707 , z = 0.0 , w = 0.707)
        #     p_array.poses.append(pose)
        # self.publisher_.publish(p_array)

        
        # for pose in p_array.poses:
        #     # Make the arm pose RobotCommand
        #     # Build a position to move the arm to (in meters, relative to and expressed in the gravity aligned body frame).
        #     x, y, z = pose.position.x, pose.position.y, pose.position.z
        #     hand_pos = geometry_pb2.Vec3(x=x, y=y, z=z)

        #     # Rotation as a quaternion
        #     qw = pose.orientation.w
        #     qx = pose.orientation.x
        #     qy = pose.orientation.y
        #     qz = pose.orientation.z
        #     hand_Q = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)

        #     hand_T_body = geometry_pb2.SE3Pose(position=hand_pos, rotation=hand_Q)

        #     hand_T_odom = hand_T_grav_body * math_helpers.SE3Pose.from_obj(hand_T_body)

        #     # duration in seconds
        #     seconds = 5

        #     arm_command = RobotCommandBuilder.arm_pose_command(
        #         hand_T_odom.x,
        #         hand_T_odom.y,
        #         hand_T_odom.z,
        #         hand_T_odom.rot.w,
        #         hand_T_odom.rot.x,
        #         hand_T_odom.rot.y,
        #         hand_T_odom.rot.z,
        #         ODOM_FRAME_NAME,
        #         seconds,
        #     )

        #     follow_arm_command = RobotCommandBuilder.follow_arm_command()
        #     command = RobotCommandBuilder.build_synchro_command(follow_arm_command, arm_command)


        #     # Convert to a ROS message
        #     action_goal = RobotCommand.Goal()
        #     conv.convert_proto_to_bosdyn_msgs_robot_command(command, action_goal.command)
        #     # Send the request and wait until the arm arrives at the goal
        #     self.get_logger().info("Moving arm to position 1.")
        #     robot_command_client.send_goal_and_wait("arm_move_one", action_goal)
     

        # #Stow arm
        # action_goal = RobotCommand.Goal()
        # command = RobotCommandBuilder.arm_stow_command()
        # conv.convert_proto_to_bosdyn_msgs_robot_command(command, action_goal.command)
        # self.get_logger().info("Stowing arm")
        # robot_command_client.send_goal_and_wait("stow arm", action_goal)

        # tf_listener.shutdown()

        return True


def main(args=None):
    rclpy.init(args=args)
    bayesian_scanner = BayesianScanner()
    rclpy.spin(bayesian_scanner)
    bayesian_scanner.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()