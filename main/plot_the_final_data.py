import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import logging
import datetime
from scipy.spatial import KDTree
from mpl_toolkits.mplot3d import Axes3D
from feature_extraction import extract_road_points

def plot3d_point_clouds(transformed_point_clouds, lidar_positions, current_folder):
    print('Extracting road points...')
    temp_clouds = []
    print("transformed_point_clouds", len(transformed_point_clouds))
    print("transformed_point_clouds[0]", transformed_point_clouds[0])
    
    road_points = extract_road_points(np.array(temp_clouds.append(clouds) for clouds in transformed_point_clouds))

    print('Plotting point clouds...')
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(road_points[:, 0], road_points[:, 1], road_points[:, 2], alpha=0.5, color='b')

    x_lidar = [pos[0] for pos in lidar_positions]
    y_lidar = [pos[1] for pos in lidar_positions]
    z_lidar = [0 for _ in lidar_positions]  # Assuming LiDAR positions are at ground level
    ax.scatter(x_lidar, y_lidar, z_lidar, color='red', s=10)

    ax.set_title('Transformed Road Point Clouds')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(['Road Points', 'LiDAR Positions'])

    plot_filename = os.path.join(current_folder, '3d_transformed_road_point_clouds_plot.png')
    plt.savefig(plot_filename)
    plt.close()
    print(f"Road point cloud plots saved to {plot_filename}")

    # Optionally display the plot
    plt.show()

# Additional code for 2D plots goes here, unchanged


def plot2d_lidar_positions(actual, predicted, current_folder):
    plt.figure(figsize=(10, 6))
    x_actual = [a[0] for a in actual]
    y_actual = [a[1] for a in actual]
    plt.scatter(x_actual, y_actual, color='blue', label='Actual')

    x_pred = [p[0][0] for p in predicted]  # Assuming predictions are array of arrays
    y_pred = [p[1][0] for p in predicted]
    plt.scatter(x_pred, y_pred, color='red', label='Predicted')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('2D Lidar Positions')
    plt.legend()
    plt.savefig(os.path.join(current_folder, '2d_lidar_positions.png'))
    plt.close()
    print(f"2D Lidar position plot saved to {os.path.join(current_folder, '2d_lidar_positions.png')}")

    # Optionally display the plot
    plt.show()
