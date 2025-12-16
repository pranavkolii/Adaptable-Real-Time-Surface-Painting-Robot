import numpy as np
import matplotlib.pyplot as plt

class UR5KinematicsValidator:
    def __init__(self):
        # UR5 Standard DH Parameters (a, alpha, d, theta_offset)
        self.dh_params = [
            [0,         np.pi/2,  0.089159, 0],       # Joint 1
            [-0.425,    0,        0,        0],       # Joint 2
            [-0.39225,  0,        0,        0],       # Joint 3
            [0,         np.pi/2,  0.10915,  0],       # Joint 4
            [0,        -np.pi/2,  0.09465,  0],       # Joint 5
            [0,         0,        0.0823,   0]        # Joint 6
        ]

    def get_transform_matrix(self, theta, d, a, alpha):
        """Standard DH Transformation Matrix"""
        return np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0,              np.sin(alpha),                np.cos(alpha),               d            ],
            [0,              0,                            0,                           1            ]
        ])

    def forward_kinematics_full(self, joints):
        """
        Calculates positions of ALL joints (for plotting) and returns
        the final Transformation Matrix T.
        """
        transforms = []
        T = np.eye(4)
        
        # Start with base at 0,0,0
        joint_positions = [[0, 0, 0]]
        
        for i, params in enumerate(self.dh_params):
            a, alpha, d, offset = params
            theta = joints[i] + offset
            
            T_i = self.get_transform_matrix(theta, d, a, alpha)
            T = np.dot(T, T_i)
            transforms.append(T)
            
            # Extract position from current T matrix
            joint_positions.append(T[:3, 3].tolist())
            
        return T, joint_positions

def validate_and_print():
    ur5 = UR5KinematicsValidator()
    
    # === DEFINING THE TEST CASES ===
    # Using the CORRECTED expected values
    test_cases = [
        {
            "name": "Zero Configuration (Flat)",
            "joints": [0, 0, 0, 0, 0, 0],
            "expected": [-0.81725, -0.19145, -0.00549] 
        },
        {
            "name": "Upright Configuration",
            "joints": [0, -np.pi/2, 0, 0, 0, 0],
            "expected": [-0.09465, -0.19145, 0.90641]
        },
        {
            "name": "L-Shape Reach",
            "joints": [0, -np.pi/2, np.pi/2, 0, 0, 0],
            "expected": [-0.39225, -0.19145, 0.41951]
        }
    ]

    # Prepare Plot
    fig = plt.figure(figsize=(15, 6))

    for i, test in enumerate(test_cases):
        # 1. Compute Forward Kinematics
        T_matrix, joint_positions = ur5.forward_kinematics_full(test['joints'])
        
        # 2. Print Matrix and Verification
        print("="*75)
        print(f"TEST CASE {i+1}: {test['name']}")
        print(f"Joints: {test['joints']}")
        print("-" * 75)
        print("Final Transformation Matrix (T_base_ee):")
        
        # Pretty print matrix
        print(np.array2string(T_matrix, formatter={'float_kind':lambda x: "%.5f" % x}, separator='  '))
        
        print("\n> Coordinate Verification:")
        coords = T_matrix[:3, 3]
        expected = test['expected']
        
        for idx, axis in enumerate(['X', 'Y', 'Z']):
            status = "MATCH" if abs(coords[idx] - expected[idx]) < 1e-4 else "FAIL"
            print(f"  {axis}: {coords[idx]:.5f} (Expected: {expected[idx]:.5f}) -> {status}")
            
        print("="*75 + "\n")

        # 3. Plotting
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        xs = [p[0] for p in joint_positions]
        ys = [p[1] for p in joint_positions]
        zs = [p[2] for p in joint_positions]
        
        # Plot the links
        ax.plot(xs, ys, zs, 'o-', linewidth=3, markersize=6, color='blue', label='Links')
        ax.scatter(0, 0, 0, color='k', s=100, marker='^', label='Base')
        ax.scatter(xs[-1], ys[-1], zs[-1], color='r', s=100, marker='*', label='End Effector')
        
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([0, 1.2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(test['name'])
        
        if i == 0: ax.view_init(elev=30, azim=10)
        if i == 1: ax.view_init(elev=10, azim=10)
        if i == 2: ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    validate_and_print()