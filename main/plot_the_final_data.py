import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import logging
import datetime

def plot3d_point_clouds(transformed_point_clouds, lidar_positions, current_folder):
    """
    Plot transformed 3D point clouds, excluding points that are too distant.

    Args:
    transformed_point_clouds (list of np.array): List of point clouds after transformation.
    current_folder (str): Path to the directory where plot should be saved.
    distance_threshold (float): Maximum allowed distance from the origin for points to be plotted.
    """
    '''
     distance_threshold_xmax = 1200
    distance_threshold_ymax = 100
    distance_threshold_zmax = 500
    distance_threshold_xmin = 550
    distance_threshold_ymin = -200
    distance_threshold_zmin = -200
    '''
    print('Plotting point clouds...')
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)

    x_coords = []
    y_coords = []
    z_coords = []
    x_lidar = []
    y_lidar = []
    index1=-1
    for point in transformed_point_clouds:
        x, y, z = point[0], point[1], point[2]
        #if (x) <= distance_threshold_xmax and (y) <= distance_threshold_ymax and (z) <= distance_threshold_zmax and x >= distance_threshold_xmin and y >= distance_threshold_ymin and z >= distance_threshold_zmin:
        x_coords.append(x)
        y_coords.append(y)
        #z_coords.append(0)
        index1+=1
    ax.scatter(x_coords, y_coords, alpha=0.01, color='b')

    print('Plotting point clouds done')

    print('Plotting lidar...')
    # Scatter LiDAR positions
    for pos in lidar_positions:
        px, py = pos[0], pos[1]
        x_lidar.append(px)
        y_lidar.append(py)
    ax.scatter(x_lidar,y_lidar, color='red', s=10)  # Red color and larger size for LiDAR positions


    print('Plotting lidar done')

    ax.set_title('Transformed Point Clouds X-Y-Z Scatter')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    #ax.set_zlabel('Z')
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
    print('Plotting 2D comparison now:')
    x_pred = []
    y_pred = []
    x_actual = []
    y_actual = []

    print("Plotting the actuals...:", actual)
    for act in actual:
        x_actual.append(act[0])
        y_actual.append(act[1])
    plt.scatter(x_actual, y_actual, color='blue', label='Actual' if act is actual[0] else "")  # Only label the first point to avoid duplicate labels
    
    print("Plotting the predictions...:", predicted)
    for pred in predicted:

        x_pred.append(pred[0, 0])
        y_pred.append(pred[0, 1])
                
    plt.scatter(x_pred, y_pred, color='red', label='Predicted' if pred is predicted[0] else "")

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('2D Lidar Positions')
    plt.legend()
    print
    # Save the plot in the unique folder
    plt.savefig(os.path.join(current_folder, f'lidar_positions.png'))
    print("PLotting finished. 2D Lidar position plot saved to:", os.path.join(current_folder, f'lidar_positions.png'))
    plt.show()
    plt.close()  # Close the plot to free up memory