import os
import scipy
from scipy.spatial import distance
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
    ax = fig.add_subplot(111, projection='3d')

    x_coords = []
    y_coords = []
    z_coords = []
    x_lidar = []
    y_lidar = []
    index1=-1
    for point in transformed_point_clouds:
        x, y, z = point[0], point[1], point[2]
        #if (x) <= distance_threshold_xmax and (y) <= distance_threshold_ymax and (z) <= distance_threshold_zmax and x >= distance_threshold_xmin and y >= distance_threshold_ymin and z >= distance_threshold_zmin:
        x_coords.append(0)
        y_coords.append(y)
        z_coords.append(z)
        index1+=1
    ax.scatter(x_coords, y_coords, z_coords, alpha=0.01, color='b')

    print('Plotting point clouds done')

    """
    print('Plotting lidar...')
    # Scatter LiDAR positions
    for pos in lidar_positions:
        px, py = pos[0], pos[1]
        x_lidar.append(px)
        y_lidar.append(py)
    ax.scatter(x_lidar,y_lidar, color='red', s=10)  # Red color and larger size for LiDAR positions


    print('Plotting lidar done')
    """
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
    print('Plotting 2D comparison now:')
    x_pred = []
    y_pred = []
    x_actual = []
    y_actual = []

    print("Plotting the actuals...:")

    print("length of actual:", len(actual))
    print("length of actual[0]:", len(actual[0]))
    for act in actual:
        for prd in act:
            x_actual.append(prd[0])
            y_actual.append(prd[1])
    plt.scatter(x_actual, y_actual, color='blue', label='Actual' if act is actual[0] else "")  # Only label the first point to avoid duplicate labels
    
    print("Plotting the predictions...:", len(predicted))
    print("length of predicted[0]:", len(predicted[0]))
    for pred in predicted:
        for lsb in pred:
            x_pred.append(lsb[0])
            y_pred.append(lsb[1])
                
    plt.scatter(x_pred, y_pred, color='red', label='Predicted' if pred is predicted[0] else "")

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('2D Lidar Positions')
    plt.legend()
    # Save the plot in the unique folder
    plt.savefig(os.path.join(current_folder, f'lidar_positions.png'))
    print("PLotting finished. 2D Lidar position plot saved to:", os.path.join(current_folder, f'lidar_positions.png'))
    plt.show()
    plt.close()  # Close the plot to free up memory

    x_eucledian = []
    y_eucledian = []
    
    # Calculate Euclidean distances and statistical data
    for act, pred in zip(actual, predicted):
        for prd, lsb in zip(act, pred):
            print("prd:", prd)
            print("lsb:", lsb)
            x_eucledian.append(distance.euclidean(prd[0][0], lsb[0]))
            y_eucledian.append(distance.euclidean(prd[1][0], lsb[1]))
                
    mean_distance_x = np.mean(x_eucledian)
    std_deviation_x = np.std(x_eucledian)
    max_distance_x = np.max(x_eucledian)
    min_distance_x = np.min(x_eucledian)
    mean_distance_y = np.mean(y_eucledian)
    std_deviation_y = np.std(y_eucledian)
    max_distance_y = np.max(y_eucledian)
    min_distance_y = np.min(y_eucledian)

    print("Euclidean Distance Statistics:")
    print(f"Mean Distance X: {mean_distance_x:.2f}")
    print(f"Standard Deviation X: {std_deviation_x:.2f}")
    print(f"Max Distance X: {max_distance_x:.2f}")
    print(f"Min Distance X: {min_distance_x:.2f}")
    print(f"Mean Distance Y: {mean_distance_y:.2f}")
    print(f"Standard Deviation Y: {std_deviation_y:.2f}")
    print(f"Max Distance Y: {max_distance_y:.2f}")
    print(f"Min Distance Y: {min_distance_y:.2f}")
