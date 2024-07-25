import numpy as np
from scipy.spatial.transform import Rotation as R



def quaternion_to_euler(quaternion):
    """Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw)."""
    r = R.from_quat(quaternion)
    return r.as_euler('xyz', degrees=False)  # Return Euler angles in radians



def create_transformation_matrix(roll, pitch, yaw, tx, ty, tz):
    """Create a 4x4 transformation matrix from Euler angles and translation vector."""
    # Rotation matrices around the x, y, and z axis
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])


    # Combined rotation matrix
    R = np.dot(np.dot(R_z, R_y), R_x)
    transformation_matrix = np.hstack( (R, (np.array ([[tx], [ty], [tz]]) )) ) # Combine rotation matrix and translation vector
    transformation_matrix = np.vstack([transformation_matrix, [0, 0, 0, 1]])
    return transformation_matrix