#!/usr/bin/env python3

import struct
import numpy as np
import rclpy
import time
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
import bdai_ros2_wrappers.process as ros_process
import bdai_ros2_wrappers.scope as ros_scope
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import PolygonStamped, Pose, Point, Point32, Quaternion
from visualization_msgs.msg import Marker, MarkerArray

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
from scipy.stats import norm

from bdai_ros2_wrappers.action_client import ActionClientWrapper
from bosdyn.api import geometry_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME
from bosdyn.client.robot_command import RobotCommandBuilder
from spot_examples.simple_spot_commander import SimpleSpotCommander
from bdai_ros2_wrappers.tf_listener_wrapper import TFListenerWrapper
from bosdyn_msgs.conversions import convert
from spot_msgs.action import RobotCommand  # type: ignore

class BayesianScanner(Node):
    def __init__(self):
        super().__init__("bayesian_scanning")
        self.declare_parameter('box_length', 2.0)
        self.declare_parameter('box_width', 2.5)
        self.declare_parameter('box_resolution', 0.05)
        self.declare_parameter('frame_id', "map")
        self.declare_parameter('cloud_topic', "cloud_map")

        self.l = self.get_parameter('box_length').get_parameter_value().double_value
        self.w = self.get_parameter('box_width').get_parameter_value().double_value
        self.r = self.get_parameter('box_resolution').get_parameter_value().double_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.cloud_topic = self.get_parameter('cloud_topic').get_parameter_value().string_value

        self.box_pub_ = self.create_publisher(PolygonStamped, 'bounding_box', 10)
        self.ground_truth_pub_ = self.create_publisher(PointCloud2, 'ground_truth', 10)
        self.box_timer_ = self.create_timer(0.5, self.bounding_box)
        self.ground_truth_timer_ = self.create_timer(0.5, self.ground_truth)
        self.samples_pub_ = self.create_publisher(PointCloud2, 'sampled_points', 10)
        self.gp_pub_ = self.create_publisher(PointCloud2, 'radiation_map', 10)
        self.grid_pub = self.create_publisher(MarkerArray, 'indication_function', 10)
        self.cloud_sub = self.create_subscription(PointCloud2, self.cloud_topic, self.indication_function, 10)

        #Initialize indication function grid
        self.grid_width = int(self.l / self.r)
        self.grid_height = int(self.w / self.r)
        self.indication_func = np.zeros((2 * self.grid_height, self.grid_width), dtype=int)

        self.spot_node_ = ros_scope.node()
        if self.spot_node_ is None:
                raise ValueError("no ROS 2 node available (did you use bdai_ros2_wrapper.process.main?)")

        self.ground_truth()
        #self.scan()

    def indication_function(self, cloud):
        # Convert PointCloud2 to array of points
        points = np.array(list(pc2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True)))

        # Clear grid
        self.indication_func.fill(1)

        # for p in points:
        #     x = p[0]
        #     y = p[1]
        #     if -(self.grid_width * self.r) <= x < self.grid_width * self.r and -(self.grid_height * self.r) <= y < self.grid_height * self.r:
        #         grid_idx_x = int(x // self.r)
        #         grid_idx_y = int(y // self.r)
        #         self.indication_func[grid_idx_y, grid_idx_x] = 1


        #Publish to ROS as marker_array
        marker_array = MarkerArray()
        marker_id = 0

        for y in range(2*self.grid_height):
            for x in range(self.grid_width):
                if self.indication_func[y, x] == 1:
                    marker = Marker()
                    marker.header.frame_id = self.frame_id
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.ns = "grid"
                    marker.id = marker_id
                    marker_id += 1
                    marker.type = Marker.CUBE
                    marker.action = Marker.ADD
                    marker.pose.position.x = 0.3 + x * self.r
                    marker.pose.position.y = -y * self.r
                    marker.pose.position.z = 0.0
                    marker.pose.orientation.x = 0.0
                    marker.pose.orientation.y = 0.0
                    marker.pose.orientation.z = 0.0
                    marker.pose.orientation.w = 1.0
                    marker.scale.x = self.r
                    marker.scale.y = self.r
                    marker.scale.z = 0.1  # height of the marker
                    marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.5)
                    marker_array.markers.append(marker)
        self.grid_pub.publish(marker_array)


    def bounding_box(self):
        bounding_box = PolygonStamped()
        bounding_box.header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.frame_id)
        points = [
        Point32(x=0.3, y=-self.w, z=0.0),
        Point32(x=0.3 + self.l, y=-self.w, z=0.0),
        Point32(x=0.3 + self.l, y=self.w, z=0.0),
        Point32(x=0.3, y=self.w, z=0.0)
        ]

        bounding_box.polygon.points = points
        self.box_pub_.publish(bounding_box)

    def function(self, x):
        return (np.exp(-(x[0] - 2)**2 - (x[1] - 2)**2) +
                np.exp(-(x[0] - 4)**2 - (x[1] - 4)**2))
    
    def ground_truth(self):
        # Plot the ground truth function
        x = np.linspace(0.3, self.l+0.3, 100)
        y = np.linspace(-self.w, self.w, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.array([self.function([xx, yy]) for xx, yy in zip(np.ravel(X), np.ravel(Y))])
        Z = Z.reshape(X.shape)
        tuples = np.dstack((X, Y, Z)).reshape(-1,3)
        
        pc2 = self.create_pointcloud(tuples, self.frame_id)
        self.ground_truth_pub_.publish(pc2)

    def create_pointcloud(self, tuples, frame_id):
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
        dim = 2
        min_val = 1
        min_x = None

        def min_obj(X):
            """Minimization objective is the negative acquisition function"""
            return -acquisition(X.reshape(-1, dim), gp=gp, y_max=y_max)
        
        mins = [b[0] for b in bounds]  # Minimum values for each dimension
        maxs = [b[1] for b in bounds]  # Maximum values for each dimension
        
        # Find the best optimum by starting from n_restart different random points.
        for x0 in np.random.uniform(mins, maxs, size=(n_restarts, dim)):
            res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun
                min_x = res.x
                
        return min_x


    def scan(self):
        # Create Gaussian Process Surrogate
        kernel = C(1.0, (1e-4, 1e1)) * RBF([1.0, 1.0], (1e-4, 1e1))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

        #Sample first point
        #Note here x_sample refers to the array of coords (x,y), y_sample is array of rad readings
        X_sample = np.array([[0.3, 0.3]])
        rad_reading = self.function(X_sample[0])
        y_sample = np.array([[rad_reading]])
        gp.fit(X_sample, y_sample)
        y_max = rad_reading

        for _ in range(10):
            tf_listener = TFListenerWrapper(self.spot_node_)
            tf_listener.wait_for_a_tform_b(ODOM_FRAME_NAME, "odom")

            #tf_listener = TFListenerWrapper(
            #"bo_scanner_tf", wait_for_transform=[ODOM_FRAME_NAME, "odom"]
            #)
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

            x_next = self.select_next(self.probability_of_improvement, gp, [(0, self.w) , (0,self.l)], y_max)
            y_next = self.function(x_next)
            if y_next > y_max:
               y_max = y_next
            X_sample = np.vstack((X_sample, x_next))
            y_sample = np.append(y_sample, y_next)
            gp.fit(X_sample, y_sample)

            tuples = [(x[0], x[1], y) for x, y in zip(X_sample, y_sample)]
            pc2 = self.create_pointcloud(tuples, self.frame_id)
            self.samples_pub_.publish(pc2)

            #Publish current surrogate model
            x = np.linspace(0, self.l, 100)
            y = np.linspace(0, self.w, 100)
            X, Y = np.meshgrid(x, y)
            X_flat = X.ravel()
            Y_flat = Y.ravel()
            XY = np.vstack([X_flat, Y_flat]).T
            Z_pred, Z_std = gp.predict(XY, return_std=True)
            Z_pred = Z_pred.reshape(X.shape)
            Z_std = Z_std.reshape(X.shape)
            tuples = np.dstack((X, Y, Z_pred)).reshape(-1,3)
            pc2 = self.create_pointcloud(tuples, self.frame_id)
            self.gp_pub_.publish(pc2)       

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
            pose = Pose()
            pose.position = Point(x = x_next[0], y = x_next[1], z = 0.1)
            pose.orientation = Quaternion(x = 0.0 , y = 0.707 , z = 0.0 , w = 0.707)
            x, y, z = pose.position.x, pose.position.y, pose.position.z
            hand_pos = geometry_pb2.Vec3(x=x, y=y, z=z)

            # Rotation as a quaternion
            qw = pose.orientation.w
            qx = pose.orientation.x
            qy = pose.orientation.y
            qz = pose.orientation.z
            hand_Q = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)

            hand_T_body = geometry_pb2.SE3Pose(position=hand_pos, rotation=hand_Q)

            hand_T_odom = hand_T_grav_body_se3 * math_helpers.SE3Pose.from_obj(hand_T_body)

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
                seconds,
            )

            follow_arm_command = RobotCommandBuilder.follow_arm_command()
            command = RobotCommandBuilder.build_synchro_command(follow_arm_command, arm_command)

            # Convert to a ROS message
            action_goal = RobotCommand.Goal()
            convert(command, action_goal.command)
            # Send the request and wait until the arm arrives at the goal
            robot_command_client.send_goal_and_wait("bayesian_optimization", action_goal)
     

        #Stow arm
        action_goal = RobotCommand.Goal()
        command = RobotCommandBuilder.arm_stow_command()
        convert(command, action_goal.command)
        self.get_logger().info("Stowing arm")
        robot_command_client.send_goal_and_wait("bayesian_optimization", action_goal)

        tf_listener.shutdown()

        return True

@ros_process.main()
def main(args=None):
    bayesian_scanner = BayesianScanner()
    rclpy.spin(bayesian_scanner)
    bayesian_scanner.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()