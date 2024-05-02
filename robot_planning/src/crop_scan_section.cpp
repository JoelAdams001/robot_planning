#define BOOST_BIND_NO_PLACEHOLDERS
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include "pcl_conversions/pcl_conversions.h"
#include <string>
#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/point_cloud2.hpp>
#include "tf2_ros/transform_listener.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2/exceptions.h"
#include "tf2_ros/buffer.h"
#include "tf2_sensor_msgs/tf2_sensor_msgs.hpp"


using std::placeholders::_1;
class CropScanSection : public rclcpp::Node
{
  public:
    CropScanSection()
    : Node("crop_scan_section")
    {
      this->declare_parameter<std::string>("cloud_topic", "/cloud_full");
      this->get_parameter("cloud_topic", cloud_topic_);
      
      this->declare_parameter<std::string>("frame_id", "body");
      this->get_parameter("frame_id", frame_id_);

      this->declare_parameter<double>("clip_distance", 3);
      this->get_parameter("clip_distance", clip_distance_);

      this->declare_parameter<float>("voxel_size", 0.1);
      this->get_parameter("voxel_size", voxel_size_);

      this->declare_parameter<std::string>("published_topic_name", "scan_section");
      this->get_parameter("published_topic_name", published_topic_name_);

      tf_buffer_ =
      std::make_unique<tf2_ros::Buffer>(this->get_clock());
      tf_listener_ =
      std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
      
      sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      cloud_topic_, rclcpp::SensorDataQoS(), std::bind(&CropScanSection::callback, this, _1));

      
      pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(published_topic_name_, rclcpp::SensorDataQoS());
    }

  private:
    void callback(const sensor_msgs::msg::PointCloud2& cloud_in)
        {
            RCLCPP_INFO(this->get_logger(), "Callback");
            pcl::PCLPointCloud2::Ptr cloud_clipped (new pcl::PCLPointCloud2 ());
            pcl::PCLPointCloud2::Ptr cloud_voxelized (new pcl::PCLPointCloud2 ());
            sensor_msgs::msg::PointCloud2 cloud_ros;
            sensor_msgs::msg::PointCloud2 cloud_transformed;

            //Transform to body frame
            // geometry_msgs::msg::TransformStamped transform;
            // try {
            //     transform = tf_buffer_->lookupTransform(
            //     frame_id_, cloud_in.header.frame_id, cloud_in.header.stamp);
            // } catch (const tf2::TransformException & ex) {
            //     RCLCPP_INFO(
            //     this->get_logger(), "Could not transform %s to %s: %s",
            //     frame_id_, cloud_in.header.frame_id, ex.what());
            // return;
            // }
            // tf2::doTransform(cloud_ros, cloud_transformed, transform);

            // Convert cloud from ROS msg to PCL msg
            pcl_conversions::toPCL(cloud_ros, *cloud_clipped);

            pcl::PassThrough<pcl::PCLPointCloud2> pass;
            // x axis
            pass.setInputCloud (cloud_clipped);
            pass.setFilterFieldName ("x");
            pass.setFilterLimits (0.6, 0.6+clip_distance_);
            pass.filter (*cloud_clipped);

            // y axis
            pass.setFilterFieldName ("y");
            pass.setFilterLimits (-clip_distance_, clip_distance_);
            pass.filter (*cloud_clipped);

            // z axis
            pass.setFilterFieldName ("z");
            pass.setFilterLimits (-0.7, clip_distance_);
            pass.filter (*cloud_clipped);

            // Create the filtering object
            pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
            sor.setInputCloud (cloud_clipped);
            sor.setLeafSize (voxel_size_, voxel_size_, voxel_size_);
            sor.filter (*cloud_voxelized);
            
            // Convert back to ROS msg
            pcl_conversions::fromPCL(*cloud_voxelized, cloud_ros);
            cloud_ros.header.frame_id = frame_id_;

            // Publish cloud
            pub_->publish(cloud_ros);
        }
    std::string cloud_topic_, frame_id_, published_topic_name_;
    double clip_distance_;
    float voxel_size_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CropScanSection>());
  rclcpp::shutdown();
  return 0;
}