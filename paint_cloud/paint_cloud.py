import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

import numpy as np
import numpy as np
import open3d as o3d
import struct


class PaintCloud(Node):
    def __init__(self):
        super().__init__('paint_cloud')

        # Publishers
        self.pcd_pub = self.create_publisher(PointCloud2, 'curved_cloud', 10)
        self.normal_pub = self.create_publisher(
            MarkerArray, 'surface_normals', 10)

        self.get_logger().info("Started the paint_cloud node")

        timer_period = 5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        self.generate_mock_pc()
        self.estimate_surface_normals()
        self.publish_message()
        self.get_logger().info("Published the message")

    def generate_mock_pc(self):
        """
        Generates a dummy pointcloud, to mock the PointCloud from depth sensor in Gazebo
        The PC generated will have slight curvature
        """

        # @TODO - Pick these parameters from a config file
        length, width, spacing = 1.5, 1.0, 0.05

        # --- Generate Synthetic Data using NumPy ---
        # 1.5m x 1.0m with 5cm spacing
        x_range = np.arange(0, length + spacing, spacing)  # 0 to 1.5m
        y_range = np.arange(0, length + spacing, spacing)  # 0 to 1.0m

        xx, yy = np.meshgrid(x_range, y_range)

        # Add curvature: z = A * sin(B * x)
        amplitude = 0.2
        frequency = 3.0
        zz = amplitude * np.sin(frequency * xx)

        # Flatten and stack to (N, 3) array
        self.points = np.column_stack(
            (xx.flatten(), yy.flatten(), zz.flatten()))

        self.get_logger().info(f"Generated {len(self.points)} mock points \
                with length: {length}, width: {width}, spacing: {spacing}")

    def estimate_surface_normals(self):
        """
        Estimate the surface normals of from the pointcloud collected
        """

        # --- Create Open3D PointCloud object ---
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.points)

        # Estimate normals
        # Radius 0.15m ensures we catch neighbors since spacing is 0.05m
        self.pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.15, max_nn=30)
        )

        # Force them to point towards positive Z [0, 0, 1]
        # This flips any normal where the dot product with Z is negative
        self.pcd.orient_normals_to_align_with_direction(
            orientation_reference=np.array([0.0, 0.0, 1.0]))

        self.get_logger().info(
            f"Generated {len(self.points)} points with normals.")

    def publish_message(self):
        """
        Takes in the points, and corresponding surface normals, and publishes 
        them as ros messages for visualization in rviz 
        """

        timestamp = self.get_clock().now().to_msg()
        header = Header()
        header.stamp = timestamp
        header.frame_id = "map"  # Fixed frame for RViz

        # --- Publish PointCloud2 ---
        pc2_msg = self.create_pointcloud2_msg(self.points, header)
        self.pcd_pub.publish(pc2_msg)

        # --- Publish Normals as Arrows ---
        marker_array = self.create_normal_markers(self.pcd, header)
        self.normal_pub.publish(marker_array)

    def create_pointcloud2_msg(self, points, header):
        """
        Takes in numpy points, and converts them to PointCloud2 msg of ros
        """

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = points.shape[0]
        msg.is_bigendian = False
        msg.is_dense = True

        # Define fields: x, y, z (float32)
        msg.fields = [
            PointField(name='x', offset=0,
                       datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,
                       datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,
                       datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 12
        msg.row_step = msg.point_step * points.shape[0]

        # Pack data
        buffer = []
        for p in points:
            buffer.append(struct.pack('fff', p[0], p[1], p[2]))
        msg.data = b''.join(buffer)

        return msg

    def create_normal_markers(self, pcd, header):
        """
        Create a MarkerArray of arrows representing surface normals.
        """
        marker_array = MarkerArray()

        # Access Open3D data as numpy arrays
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)

        # Downsample for visualization if needed (showing all 600+ is fine, but this loops safely)
        for i in range(len(points)):
            marker = Marker()
            marker.header = header
            marker.ns = "normals"
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD

            # Start point (the point on the surface)
            start = Point()
            start.x = float(points[i][0])
            start.y = float(points[i][1])
            start.z = float(points[i][2])

            # End point (start + normal_vector * scale)
            scale = 0.1  # Length of the arrow
            end = Point()
            end.x = start.x + float(normals[i][0]) * scale
            end.y = start.y + float(normals[i][1]) * scale
            end.z = start.z + float(normals[i][2]) * scale

            marker.points = [start, end]

            # Arrow properties
            marker.scale.x = 0.01  # Shaft diameter
            marker.scale.y = 0.02  # Head diameter
            marker.scale.z = 0.05  # Head length

            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0  # Yellow arrows
            marker.color.a = 1.0

            marker.lifetime.sec = 0  # 0 means forever (until overwritten)
            marker.frame_locked = True

            marker_array.markers.append(marker)

        return marker_array


def main(args=None):
    rclpy.init(args=args)

    paint_cloud_node = PaintCloud()

    rclpy.spin(paint_cloud_node)

    try:
        rclpy.spin(paint_cloud_node)
    except KeyboardInterrupt:
        pass
    finally:
        paint_cloud_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
