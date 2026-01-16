# Adaptable Real-Time Surface Painting Robot

A ROS 2 package for generating robot manipulator painting paths from 3D perception data.

[![Project Demo](https://img.youtube.com/vi/ujLB43V4m5M/0.jpg)](https://www.youtube.com/watch?v=ujLB43V4m5M)

## Project Overview

**Goal**: Develop a real-time system that scans an arbitrary object, plans a coverage path adaptively, and executes the painting trajectory using a UR5-based manipulator.

**Methodology**:
1.  **Sense**: Use 3D LiDAR/Depth camera to scan the environment and generate a point cloud.
2.  **Plan**: Compute surface normals (PCA), generate a 2D coverage pattern, and project it onto the 3D surface using a KD-Tree.
3.  **Execute**: Solve Inverse Kinematics (Jacobian-based) to guide the end-effector along the planned path while maintaining surface-normal orientation.

## Features

- **PointCloud Processing**: Downsamples and transforms raw point cloud data from the camera frame to the world frame.
- **Surface Normal Estimation**: Uses Open3D to estimate normals for every point in the cloud.
- **Coverage Path Planning**: Generates a zig-zag path based on configurable surface dimensions, brush radius, and overlap factor.
- **3D Projection**: Uses a KD-Tree to project the 2D planned path onto the actual 3D curved surface.
- **Orientation Alignment**: Computes the orientation for each waypont to align the tool with the surface normal.
- **Service Interface**: Provides a ROS 2 service to trigger path generation and retrieve the path.
- **Visualization**: Publishes markers (normals), point clouds, and path poses for visualization in RViz.

## Installation

### Dependencies

- ROS 2 (humble/jazzy)
- `ros_gz` (ROS 2 - Gazebo Bridge)
- **Python Libraries**:
  - `open3d`
  - `scipy`
  - `numpy`
  - `transforms3d` (or `tf_transformations`)

### Building

1. Create a workspace (if you haven't already):
   ```bash
   mkdir -p ~/paint_ws
   cd ~/paint_ws
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/AakashDammala/paint_cloud.git src/
   ```

3. Install dependencies:
   ```bash
   cd ~/paint_ws
   rosdep install --from-paths src --ignore-src -r -y
   ```
   *Note: If you have to manually install python packages:*
   ```bash
   pip3 install open3d scipy
   ```

4. Build the workspace:
   ```bash
   colcon build
   source install/setup.bash
   ```

## Usage

### Launching the Simulation

This launches Gazebo with a LiDAR scanner in the environment, RViz2, and URDF and controllers.

```bash
ros2 launch paint_cloud classic.launch.py
```

Launch the path planning node
```bash
ros2 run paint_cloud paint_cloud
```

Trace the path using IK solver, which can calls the planning node to get the path
```bash
ros2 run my_ik_controller jacobian_mover
```

### Parameters

You can configure the following parameters in `paint_cloud.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `downsample_factor` | `100` | Factor to downsample the input point cloud, higher the factor, less dense the point cloud. |
| `brush_radius` | `0.1` | Radius of the painting brush (meters). |
| `overlap_factor` | `1.0` | Overlap between painting strokes (1.0 - no overlap, 0.5 - half overlap). |
| `surface_length_x` | `1.0` | Length of the target surface area (meters). |
| `surface_width_y` | `0.8` | Width of the target surface area (meters). |

### Services

| Service | Type | Description |
|---------|------|-------------|
| `get_paint_path` | `paint_cloud_msgs/GetPaintPath` | Triggers path generation and returns the path poses. |

### Outcomes
* Architected an autonomous painting system using a 6-DoF UR5 articulated arm, to reconstruct unknown surface geometries and autonomously generate adaptive zig-zag trajectories in real-time.
* Developed a perception-driven control framework in ROS2 and C++, implementing a numerical Jacobian pseudo-inverse solver for inverse kinematics and Principal Component Analysis (PCA) to extract surface normals for high-precision tool orientation.
