import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from paint_cloud_msgs.srv import GetPaintPath 
import numpy as np
import math
import time
import matplotlib.pyplot as plt

class JacobianIKSolver(Node):

    def __init__(self):
        super().__init__('jacobian_ik_publisher')
        
        self.publisher_ = self.create_publisher(
            JointTrajectory, 
            '/joint_trajectory_controller/joint_trajectory', 
            10
        )

        # Create the Service Client
        self.cli = self.create_client(GetPaintPath, '/get_paint_path')
        
        # UR5 Standard DH Parameters (a, alpha, d, theta_offset)
        self.dh_params = [
            [0,         np.pi/2,  0.089159, 0],       # Joint 1
            [-0.425,    0,        0,        0],       # Joint 2
            [-0.39225,  0,        0,        0],       # Joint 3
            [0,         np.pi/2,  0.10915,  0],       # Joint 4
            [0,        -np.pi/2,  0.09465,  0],       # Joint 5
            [0,         0,        0.0823,   0]        # Joint 6
        ]
        
        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

    def get_transform_matrix(self, theta, d, a, alpha):
        """Standard DH Transformation Matrix"""
        return np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0,              np.sin(alpha),                np.cos(alpha),               d            ],
            [0,              0,                            0,                           1            ]
        ])

    def forward_kinematics(self, joints):
        """
        Calculates the transformation matrices for all frames.
        Returns: 
           final_pos: [x, y, z] of end effector
           transforms: List of T matrices for all links (needed for Jacobian)
        """
        transforms = []
        T = np.eye(4)
        
        for i, params in enumerate(self.dh_params):
            a, alpha, d, offset = params
            theta = joints[i] + offset
            
            T_i = self.get_transform_matrix(theta, d, a, alpha)
            T = np.dot(T, T_i)
            transforms.append(T)
            
        final_pos = T[:3, 3]
        return final_pos, transforms

    def compute_jacobian(self, transforms):
        """Computes the Geometric Jacobian (6x6 matrix)."""
        J = np.zeros((6, 6))
        
        # End-effector position
        p_e = transforms[-1][:3, 3]
        
        # Base frame
        z_prev = np.array([0, 0, 1]) # Z-axis of base
        p_prev = np.array([0, 0, 0]) # Origin of base
        
        for i in range(6):
            # Geometric Jacobian formula for revolute joints
            J[:3, i] = np.cross(z_prev, (p_e - p_prev)) # Linear part
            J[3:, i] = z_prev                           # Angular part
            
            # Update for next iteration
            z_prev = transforms[i][:3, 2] 
            p_prev = transforms[i][:3, 3] 
            
        return J

    def solve_ik(self, target_pos, initial_joints, tolerance=0.01, max_iter=100):
        """Newton-Raphson method using Pseudo-Inverse."""
        q = np.array(initial_joints, dtype=float)
        
        for _ in range(max_iter):
            current_pos, transforms = self.forward_kinematics(q)
            
            # Calculate Error (Target - Current)
            error_pos = target_pos - current_pos
            
            # Check convergence
            if np.linalg.norm(error_pos) < tolerance:
                return q
            
            # Get Jacobian
            J = self.compute_jacobian(transforms)
            J_pos = J[:3, :] 
            
            # Calculate Pseudo-Inverse (Moore-Penrose)
            J_pinv = np.linalg.pinv(J_pos)
            
            # Update joints
            delta_q = np.dot(J_pinv, error_pos)
            q += delta_q * 0.5 
            
        self.get_logger().warn("IK did not converge!")
        return q

    def fetch_waypoints(self):
        """Calls the service and returns a list of [x,y,z] lists"""
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')

        req = GetPaintPath.Request()
        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            response = future.result()
            points = []
            for pose in response.poses.poses:
                points.append([pose.position.x, pose.position.y, pose.position.z])
            
            self.get_logger().info(f"Received {len(points)} waypoints from service.")
            return points
        else:
            self.get_logger().error('Service call failed')
            return []

    def execute_path(self):
        # 1. Start Configuration (Home-ish)
        current_joints = [0.0, -1.57, 1.57, 0.0, 0.0, 0.0]
        
        # 2. Get Waypoints from Service
        self.get_logger().info("Requesting waypoints from /get_paint_path...")
        waypoints = self.fetch_waypoints()

        if not waypoints:
            self.get_logger().warn("No waypoints received. Aborting execution.")
            return

        traj_msg = JointTrajectory()
        traj_msg.joint_names = self.joint_names
        time_from_start = 2.0 

        # === PLOTTING DATA CONTAINERS ===
        joint_history = []
        ee_pos_history = []
        # ================================

        for i, point in enumerate(waypoints):
            self.get_logger().info(f"Solving IK for point: {point}")
            
            # Solve IK for this point
            target = np.array(point)
            new_joints = self.solve_ik(target, current_joints)
            
            # Compute FK for the *solved* joints to verify/plot actual path
            actual_pos, _ = self.forward_kinematics(new_joints)
            
            # Store data for plotting
            joint_history.append(new_joints)
            ee_pos_history.append(actual_pos)
            
            # Update current_joints so next iteration starts from here
            current_joints = new_joints
            
            # Create Message Point
            p = JointTrajectoryPoint()
            p.positions = new_joints.tolist()
            p.time_from_start = Duration(sec=int((i+1)*time_from_start), nanosec=0)
            traj_msg.points.append(p)

        self.get_logger().info("Publishing Trajectory...")
        self.publisher_.publish(traj_msg)
        
        # === PLOTTING SECTION ===
        # We pass 'waypoints' (the target input) to the plotter
        self.plot_results(joint_history, ee_pos_history, waypoints)

    def plot_results(self, joint_hist, pos_hist, target_path):
        """Generates plots for Joint Angles and End Effector Path vs Target"""
        joint_hist = np.array(joint_hist)
        pos_hist = np.array(pos_hist)
        target_path = np.array(target_path) # Convert input list to array
        steps = np.arange(len(joint_hist))

        # 1. Plot End Effector X-Y Plane
        plt.figure(figsize=(8, 8))
        
        # Plot Target Input (Green dashed line with 'x' markers)
        plt.plot(target_path[:, 0], target_path[:, 1], 'g--x', label='Target Input')
        
        # Plot Computed IK Result (Blue solid line with 'o' markers)
        plt.plot(pos_hist[:, 0], pos_hist[:, 1], 'b-o', alpha=0.6, label='Computed IK')
        
        plt.title("Trajectory Tracking: Target vs Computed (X-Y Plane)")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        
        # 2. Plot Individual Joint Positions (3x2 Subplots)
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        fig.suptitle("Joint Angles over Waypoints")
        
        for i, ax in enumerate(axes.flat):
            if i < 6:
                ax.plot(steps, joint_hist[:, i], 'r-')
                ax.set_title(self.joint_names[i])
                ax.set_ylabel("Radians")
                ax.set_xlabel("Waypoint Index")
                ax.grid(True)
        
        plt.tight_layout()
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = JacobianIKSolver()
    
    # Wait for connections (publishers)
    time.sleep(1)
    
    # Execute the path (handles service call internally)
    node.execute_path()
    
    # Spin briefly to ensure messages are sent out
    rclpy.spin_once(node, timeout_sec=5)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()