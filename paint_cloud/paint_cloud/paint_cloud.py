import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

import numpy as np
import open3d as o3d
import struct
from sensor_msgs_py import point_cloud2
import tf2_ros
import tf_transformations
from geometry_msgs.msg import TransformStamped, PoseArray, Pose, PolygonStamped, Point32
from paint_cloud_msgs.srv import GetPaintPath
from scipy.spatial import cKDTree

def quaternion_from_normal(normal):
    """
    Computes a quaternion (x, y, z, w) that aligns the Z-axis with the given normal vector.
    Uses tf_transformations for the final quaternion generation.
    """
    normal = np.array(normal)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return [0.0, 0.0, 0.0, 1.0] # Identity
    
    normal = normal / norm
    
    # We want to rotate the standard Z axis (0, 0, 1) to the normal
    start_vec = np.array([0.0, 0.0, 1.0])
    
    # Dot product and cross product
    dot = np.dot(start_vec, normal)
    cross = np.cross(start_vec, normal)
    
    # Check for parallel/anti-parallel
    if dot > 0.999999:
        return [0.0, 0.0, 0.0, 1.0]
    elif dot < -0.999999:
        # Rotate 180 deg around X
        return list(tf_transformations.quaternion_about_axis(np.pi, [1.0, 0.0, 0.0]))
    
    # Calculate angle and axis
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    axis = cross / np.linalg.norm(cross)
    
    return list(tf_transformations.quaternion_about_axis(angle, axis))


class PaintCloud(Node):
    def __init__(self):
        super().__init__('paint_cloud')

        # Publishers
        self.pcd_pub = self.create_publisher(PointCloud2, 'curved_cloud', 10)
        self.normal_pub = self.create_publisher(
            MarkerArray, 'surface_normals', 10)
        self.path_pub = self.create_publisher(PoseArray, 'paint_path', 10)
        self.polygon_pub = self.create_publisher(PolygonStamped, 'surface_polygon', 10)

        # Services
        self.srv = self.create_service(GetPaintPath, 'get_paint_path', self.get_paint_path_callback)

        # Subscribers
        self.create_subscription(
            PointCloud2,
            '/realsense/points',
            self.pointcloud_callback,
            10
        )

        self.declare_parameter('downsample_factor', 100)
        self.declare_parameter('brush_radius', 0.1)
        self.declare_parameter('overlap_factor', 1.0)
        self.declare_parameter('surface_length_x', 1.0)
        self.declare_parameter('surface_width_y', 0.8)

        self.points = None
        self.pcd_header = None
        self.generated_path = None
        
        # TF Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # timer_period = 5  # seconds
        # self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info("Started the paint_cloud node")

    def timer_callback(self):
        self.scan_pc_generate_path()

    def get_paint_path_callback(self, request, response):
        """
        Callback for the GetPaintPath service.
        Triggers path generation and returns the result.
        """
        self.get_logger().info("Incoming request for Paint Path")
        
        path = self.scan_pc_generate_path()
        
        if path:
            response.poses = path
            self.get_logger().info(f"Returning path with {len(path.poses)} poses")
        else:
            self.get_logger().warn("Path generation failed or returned None")
            response.poses = PoseArray() 
            if self.pcd_header:
                response.poses.header = self.pcd_header
            
        return response
    
    def scan_pc_generate_path(self):
        if self.points is None:
            self.get_logger().info("Waiting for pointcloud...")
            return

        self.estimate_surface_normals()
        self.generate_path()
        self.publish_message()
        self.get_logger().info("Published the message")
        return self.generated_path

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

    def pointcloud_callback(self, msg):
        new_points = None
        
        # Optimize for the specific Realsense format provided
        # x(0), y(4), z(8), rgb(16), point_step=24, Little Endian (usually)
        if msg.point_step == 24:
            try:
                # Structured dtype matching the memory layout
                dtype_list = [
                    ('x', '<f4'),
                    ('y', '<f4'),
                    ('z', '<f4'),
                    ('skip1', 'V4'), # Offset 12-16
                    ('rgb', '<f4'),
                    ('skip2', 'V4')  # Offset 20-24
                ]
                
                # If big endian, swap byte order in dtype
                if msg.is_bigendian:
                    dtype_list = [(n, f.replace('<', '>')) for n, f in dtype_list]
                
                raw_data = np.frombuffer(msg.data, dtype=dtype_list)
                
                # Stack x, y, z
                points = np.column_stack((raw_data['x'], raw_data['y'], raw_data['z']))
                
                # Filter NaNs
                # Check for NaNs in any coordinate
                mask = ~np.isnan(points).any(axis=1)
                new_points = points[mask]

            except Exception as e:
                self.get_logger().warn(f"Fast parse failed: {e}. Falling back to standard reader.")
        
        if new_points is None:
            # Fallback for other formats
            gen = point_cloud2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))
            new_points = np.array(list(gen))

        if new_points.size > 0:
            # Downsample
            factor = self.get_parameter('downsample_factor').get_parameter_value().integer_value
            if factor > 1:
                new_points = new_points[::factor]

            # Transform to world frame
            try:
                # Lookup transform from 'world' to sensor frame (msg.header.frame_id)
                # We want to transform points IN sensor frame TO world frame.
                # So we need transform T_world_sensor
                trans = self.tf_buffer.lookup_transform(
                    'world', 
                    msg.header.frame_id, 
                    # Use time 0 for latest to avoid sync issues, or msg.header.stamp
                    # Using stamp is safer if available, but might fail if lag. 
                    # Let's use latest for robustness in this simple node.
                    rclpy.time.Time() 
                )
                
                # Apply transform
                # translation
                t = trans.transform.translation
                translation = np.array([t.x, t.y, t.z])
                
                # rotation (quaternion)
                r = trans.transform.rotation
                quaternion = [r.x, r.y, r.z, r.w]
                rotation_matrix = tf_transformations.quaternion_matrix(quaternion)[:3, :3]
                
                # Formula: P_world = R * P_sensor + T
                new_points = np.dot(new_points, rotation_matrix.T) + translation
                
                # Update header to world
                self.pcd_header = msg.header
                self.pcd_header.frame_id = 'world'
                self.points = new_points

            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                self.get_logger().warn(f"TF Lookup failed: {e}")
                return

            # self.get_logger().info(f"Received {len(self.points)} points")

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

    def generate_path(self):
        """
        Take the following inputs
          - Surface normals (take into account that the normals are wrt sensor frame x axis)
          - paint brush radius
          - Surface size (length x width) (if length is 1m and width is 0.8m, the booundaries of painting box are x(-0.5m to 0.5m) and y(-0.4m to 0.4m))

        How to generate the path?
          - Generate zig lines (ends spanning outside the width of surface)
          - The distance between these lines should be overlap_factor x (2x brush_radius)
          - Higher the overlap factor, more closer the lines will be, ideally it should be 1.0
        
        With the overlap factor as 1.0, paint brush radius as 0.1m, the path generated will be as follows:
          (-0.6, -0.5), (-0.6, 0.5), (-0.4, 0.5), (-0.4, -0.5), (-0.2, -0.5), (-0.2, 0.5), (0.0, 0.5), (0.0, -0.5), (0.2, -0.5), (0.2, 0.5), (0.4, 0.5), (0.4, -0.5), (0.6, -0.5), (0.6, 0.5)
          (The path starts 0.1m from the left and right edges of the surface)
          Save this points waypoints, in the same zig zag order

        Once these waypoints are generated, interpolate sub points in between these points, with a distance of 0.1m (parameter)
        These sub points will be stored as geometry_msgs/PoseArray message
        The orientation will be 0, 0, 0 for now, and Z will be 0.0 for now
        """
        if self.points is None:
            return

        # Fetch Parameters
        brush_radius = self.get_parameter('brush_radius').get_parameter_value().double_value
        surface_length_x = self.get_parameter('surface_length_x').get_parameter_value().double_value
        surface_width_y = self.get_parameter('surface_width_y').get_parameter_value().double_value
        overlap_factor = self.get_parameter('overlap_factor').get_parameter_value().double_value

        # Calculate limits based on surface dimensions (centered at 0,0)
        x_half = surface_length_x / 2.0
        y_half = surface_width_y / 2.0

        # Expand boundaries by brush radius as per example
        path_x_min = -(x_half + brush_radius)
        path_x_max = (x_half + brush_radius)
        path_y_min = -(y_half + brush_radius)
        path_y_max = (y_half + brush_radius)
        
        # Calculate step size in X
        # As per example: overlap=1.0, radius=0.1 => step=0.2. formula: 2*r/overlap
        if overlap_factor <= 0:
            self.get_logger().warn("Overlap factor must be > 0. Defaulting to 1.0")
            overlap_factor = 1.0
            
        step_x = (2.0 * brush_radius) / overlap_factor

        # Generate Waypoints
        waypoints = []
        current_x = path_x_min
        
        # Use a epsilon for float comparison to include the upper bound
        epsilon = 1e-6
        direction_up = True # Start moving Up (from y_min to y_max)
        
        # We need to iterate until we cover path_x_max
        while current_x <= path_x_max + epsilon:
            if direction_up:
                waypoints.append((current_x, path_y_min))
                waypoints.append((current_x, path_y_max))
            else:
                waypoints.append((current_x, path_y_max))
                waypoints.append((current_x, path_y_min))
            
            direction_up = not direction_up
            current_x += step_x

        # Interpolation
        final_points = []
        interp_step = 0.1
        
        for i in range(len(waypoints) - 1):
            p1 = waypoints[i]
            p2 = waypoints[i+1]
            
            # Add the start point
            final_points.append(p1)
            
            # Calculate distance and vector
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist > 0:
                vx = dx / dist
                vy = dy / dist
                
                # Add sub-points
                current_dist = interp_step
                while current_dist < dist:
                    sub_x = p1[0] + vx * current_dist
                    sub_y = p1[1] + vy * current_dist
                    final_points.append((sub_x, sub_y))
                    current_dist += interp_step
        
        # Add the very last point
        if waypoints:
            final_points.append(waypoints[-1])

        # Create PoseArray
        self.generated_path = PoseArray()
        # header will be set in publish_message
        
        # Build KDTree for nearest neighbor search
        # We project the 3D points to 2D (x, y) for the search
        if self.points is not None and len(self.points) > 0:
            # Ahhh, this can be optimized, but it works for now. I could have used depth image to get the points, but it was not working.
            self.kd_tree = cKDTree(self.points[:, :2])
            
            normals = np.asarray(self.pcd.normals)
            
            for p in final_points:
                # Query nearest point in the cloud
                dist, idx = self.kd_tree.query([p[0], p[1]])
                
                # Get the Z height from the surface point
                surface_point = self.points[idx]
                surface_z = surface_point[2]
                
                # Get the normal
                normal = normals[idx]
                
                # Calculate orientation
                orientation = quaternion_from_normal(normal)
                
                pose = Pose()
                pose.position.x = float(p[0])
                pose.position.y = float(p[1])
                
                # Adjust Z: surface height offset along the normal by brush radius
                # Position = surface_point + normal * brush_radius
                offset_pos = surface_point + normal * brush_radius
                
                pose.position.x = float(offset_pos[0])
                pose.position.y = float(offset_pos[1])
                pose.position.z = float(offset_pos[2])
                
                pose.orientation.x = orientation[0]
                pose.orientation.y = orientation[1]
                pose.orientation.z = orientation[2]
                pose.orientation.w = orientation[3]
                
                self.generated_path.poses.append(pose)
            
            self.get_logger().info(f"Generated path with {len(self.generated_path.poses)} poses using KDTree projection.")
        else:
             self.get_logger().warn("No points available to build KDTree.")

    def publish_message(self):
        """
        Takes in the points, and corresponding surface normals, and publishes 
        them as ros messages for visualization in rviz 
        """
        if self.pcd_header is None:
            self.get_logger().warn("PC header not set. Skipping publish")
            return

        header = self.pcd_header
        
        # --- Publish PointCloud2 ---
        pc2_msg = self.create_pointcloud2_msg(self.points, header)
        self.pcd_pub.publish(pc2_msg)

        # --- Publish Normals as Arrows ---
        marker_array = self.create_normal_markers(self.pcd, header)
        self.normal_pub.publish(marker_array)
        
        # --- Publish Path ---
        if self.generated_path:
            self.generated_path.header = header # Sync header
            self.path_pub.publish(self.generated_path)

        # --- Publish Surface Polygon ---
        self.publish_surface_polygon(header)

    def publish_surface_polygon(self, header):
        """
        Publishes a rectangle polygon representing the surface boundaries
        """
        surface_length_x = self.get_parameter('surface_length_x').get_parameter_value().double_value
        surface_width_y = self.get_parameter('surface_width_y').get_parameter_value().double_value
        
        x_half = surface_length_x / 2.0
        y_half = surface_width_y / 2.0
        
        polygon_msg = PolygonStamped()
        polygon_msg.header = header
        
        # Define 4 corners (Counter Closkwise)
        p1 = Point32(x=float(-x_half), y=float(-y_half), z=0.0)
        p2 = Point32(x=float(x_half), y=float(-y_half), z=0.0)
        p3 = Point32(x=float(x_half), y=float(y_half), z=0.0)
        p4 = Point32(x=float(-x_half), y=float(y_half), z=0.0)
        
        polygon_msg.polygon.points = [p1, p2, p3, p4]
        
        self.polygon_pub.publish(polygon_msg)

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
