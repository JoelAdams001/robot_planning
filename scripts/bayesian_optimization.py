#!/usr/bin/env python3

import struct
import numpy as np
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from robot_planning_interfaces.action import FindNextPoint
from robot_planning_interfaces.srv import UpdateModel
from std_srvs.srv import Trigger
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.optimize import minimize
from scipy.stats import norm
import math

class BayesianOptimization(Node):
    def __init__(self):
        super().__init__("bayesian_optimization")
        # Parameters
        self.declare_parameter('box_length', 1.7)
        self.declare_parameter('box_width', 1.7)
        self.declare_parameter('frame_id', "map")
        self.l = self.get_parameter('box_length').get_parameter_value().double_value
        self.w = self.get_parameter('box_width').get_parameter_value().double_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        # Action servers, service servers and publishers
        self._find_next_point_server = ActionServer(
            self,
            FindNextPoint,
            'find_next_point',
            self.find_next_point)
        self.update_model_service = self.create_service(UpdateModel, 'update_model', self.update_model)
        self.pub_sample_pts_service = self.create_service(Trigger, 'publish_sample_pts', self.publish_sample_pts)
        self.samples_pub_ = self.create_publisher(PointCloud2, 'sampled_points', 10)
        self.gp_pub_ = self.create_publisher(PointCloud2, 'radiation_map', 10)

        # Create Gaussian Process Surrogate
        kernel = C(1.0, (1e-4, 1e1)) * RBF([1.0, 1.0], (1e-4, 1e1))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

        # Sample first point
        # Note here x_sample refers to the array of coords (x,y), y_sample is array of rad readings
        self.X_sample = np.array([[0.4, 0.4]])
        rad_reading = self.function(self.X_sample[0])
        self.y_sample = np.array([[rad_reading]])
        self.gp.fit(self.X_sample, self.y_sample)
        self.y_max = rad_reading

    def find_next_point(self, goal_handle):
        self.get_logger().info("Finding next point to sample...")

        n_restarts=25
        dim = 2
        min_val = 1
        min_x = None

        def min_obj(X):
            """Minimization objective is the negative acquisition function"""
            return -self.probability_of_improvement(X.reshape(-1, dim), goal_handle.request.fail_count)
        
        bounds = [(-self.w, self.w) , (0.8, self.l+0.8)]
        mins = [b[0] for b in bounds]  # Minimum values for each dimension
        maxs = [b[1] for b in bounds]  # Maximum values for each dimension

        # Find the best optimum by starting from n_restart different random points.
        for x0 in np.random.uniform(mins, maxs, size=(n_restarts, dim)):
            res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun
                min_x = res.x
        result = FindNextPoint.Result()
        result.point = Point(x = min_x[0], y = min_x[1], z = 0.2)
        return result
    
    def probability_of_improvement(self, X, fail_count):
        # Calculate xi based on number of times consecutively the robot failed to reach its goal point
        if (fail_count >= 0 and fail_count <= 5):
            xi = (1/5)*math.exp(0.5*fail_count)
        else:
            xi = 2.5
        #print(self.fail_count)
        print("Current xi: ", str(xi))
        """Computes the PI at points X based on existing samples using a Gaussian process surrogate model."""
        mu, sigma = self.gp.predict(X, return_std=True)
        with np.errstate(divide='warn'):
            # Improvement threshold, adjust xi to trade-off exploration vs. exploitation.
            imp_thresh = self.y_max + xi
            Z = (mu - imp_thresh) / sigma
            if sigma == 0.0:
                return 0.0
            return norm.cdf(Z)
        
    def update_model(self, req, res):
        if req.y > self.y_max:
            self.y_max = req.y
        # Update GP model
        X_sample = np.vstack((self.X_sample, req.x))
        y_sample = np.append(self.y_sample, req.y)
        self.gp.fit(self.X_sample, self.y_sample)
        res.success = True
        return res
    
    def publish_sample_pts(self, req, res):
        tuples = [(x[0], x[1], y) for x, y in zip(self.X_sample, self.y_sample)]
        pc2 = self.create_pointcloud(tuples, self.frame_id)
        self.samples_pub_.publish(pc2)
        res.success = True
        return res
    
    def publish_surrogate_model(self, req, res):
        x = np.linspace(-self.w, self.w, 200)
        y = np.linspace(0.8, self.l+0.8, 100)
        X, Y = np.meshgrid(x, y)
        X_flat = X.ravel()
        Y_flat = Y.ravel()
        XY = np.vstack([X_flat, Y_flat]).T
        Z_pred, Z_std = self.gp.predict(XY, return_std=True)
        Z_pred = Z_pred.reshape(X.shape)
        Z_std = Z_std.reshape(X.shape)
        tuples = np.dstack((X, Y, Z_pred)).reshape(-1,3)
        pc2 = self.create_pointcloud(tuples, self.frame_id)
        self.gp_pub_.publish(pc2) 

    def create_pointcloud(self, tuples, frame_id):
        # Construct a ROS2 PointCloud2 message using a numpy array of tuples
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
            point_step=12, # Each point consists of three float32s, each float32 is 4 bytes
            row_step=12 * len(tuples),
            data=bytearray(buf)
        )
        return pc2


def main(args=None):
    rclpy.init(args=args)
    bayesian_optimization = BayesianOptimization()
    rclpy.spin(bayesian_optimization)
    bayesian_optimization.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()