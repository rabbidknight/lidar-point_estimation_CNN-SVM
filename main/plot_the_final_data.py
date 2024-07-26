import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import logging
import datetime

def plot3d_point_clouds(transformed_point_clouds, current_folder):
    """
    Plot transformed 3D point clouds, excluding points that are too distant.

    Args:
    transformed_point_clouds (list of np.array): List of point clouds after transformation.
    current_folder (str): Path to the directory where plot should be saved.
    distance_threshold (float): Maximum allowed distance from the origin for points to be plotted.
    """

    distance_threshold_xmax = 1100
    distance_threshold_ymax = 100
    distance_threshold_zmax = 500
    distance_threshold_xmin = 550
    distance_threshold_ymin = -200
    distance_threshold_zmin = -200

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    x_coords = []
    y_coords = []
    z_coords = []
    index1=-1
    for point in transformed_point_clouds:
        x, y, z = point[0], point[1], point[2]
        if index1<975000 and index1>950880:
            if (x) <= distance_threshold_xmax and (y) <= distance_threshold_ymax and (z) <= distance_threshold_zmax and x >= distance_threshold_xmin and y >= distance_threshold_ymin and z >= distance_threshold_zmin:
                x_coords.append(x)
                y_coords.append(y)
                z_coords.append(z)
        index1+=1
    ax.scatter(x_coords, y_coords, z_coords, alpha=0.1, color='b')

    print('Plotting done')

    ax.set_title('Transformed Point Clouds X-Y-Z Scatter')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(['Transformed Point Clouds'])

    # Save the plot to the directory
    plot_filename = os.path.join(current_folder, '3d_transformed_point_clouds_plot.png')
    plt.savefig(plot_filename)
    plt.close()  # Close the plot to free up memory and save the file
    print(f"3D point cloud plots saved to {plot_filename}")

    # Optionally display the plot
    plt.show()



def plot2d_lidar_positions(actual, predicted, current_folder):
    plt.figure(figsize=(10, 6))
    for act in actual:
        plt.scatter(act[0], act[1], color='blue', label='Actual' if act is actual[0] else "")  # Only label the first point to avoid duplicate labels
    
    for pred in predicted:
        if pred.ndim > 1 and pred.shape[1] >= 2:  # Ensure pred is at least 2D and has at least two columns
            plt.scatter(pred[0, 0], pred[0, 1], color='red', label='Predicted' if pred is predicted[0] else "")  # pred[0, 0] and pred[0, 1] for first row's x and y
        elif pred.ndim == 1 and len(pred) >= 2:  # If it's 1D but has at least two elements
            plt.scatter(pred[0], pred[1], color='red', label='Predicted' if pred is predicted[0] else "")
        else:
            print(f"Unexpected prediction shape or size: {pred.shape}")

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('2D Lidar Positions')
    plt.legend()

    # Save the plot in the unique folder
    plt.savefig(os.path.join(current_folder, f'lidar_positions.png'))

    plt.show()
    plt.close()  # Close the plot to free up memory