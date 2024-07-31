import numpy as np
from scipy.spatial import KDTree

def extract_road_points(point_clouds):
    """
    Extract road points from point clouds using a neighbor graph for density estimation.

    Args:
    point_clouds (np.array): Array of 3D points.

    Returns:
    np.array: Filtered road points.
    """
    tree = KDTree(point_clouds)
    indices = tree.query_ball_point(point_clouds, r=0.5)  # radius can be adjusted

    # Define a threshold for density
    road_indices = [idx for idx, points in enumerate(indices) if len(points) > 10]  # density threshold
    road_points = np.array([point_clouds[i] for i in road_indices])

    return road_points
