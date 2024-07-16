import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import rosbag
import pandas as pd
import joblib
from joblib import load
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import load_model
from sklearn.svm import SVR
import os
import datetime
from datetime import datetime
import logging
import re
import subprocess

def launch_plotjuggler(bag_file_path):
    try:
        subprocess.run(["plotjuggler-ros", bag_file_path], check=True)
    except subprocess.CalledProcessError as e:
        print("Failed to launch PlotJuggler:", e)



def find_latest_folder(base_path, pattern="test_results_"):
    latest_time = None
    latest_folder = None
    for folder in os.listdir(base_path):
        if folder.startswith(pattern):
            # Extract the timestamp from the folder name
            timestamp_str = folder[len(pattern):]
            try:
                folder_time = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
                if latest_time is None or folder_time > latest_time:
                    latest_time = folder_time
                    latest_folder = folder
            except ValueError:
                continue  # If the timestamp is not valid, skip this folder
    return latest_folder if latest_folder else None

# Assuming the base directory where your test results folders are saved
base_directory = os.getcwd()  # or set this to where your folders are

# Find the latest folder
current_folder = find_latest_folder(base_directory)
if not current_folder:
    print("No valid test result folder found.")
    sys.exit(1)  # Exit if no folder found

# Use current_folder for loading models and saving plots

def extract_data_from_bag(bag_file):
    bag = rosbag.Bag(bag_file)
    point_cloud_data = []
    lidar_transform_data = []

    for topic, msg, t in bag.read_messages():
        if topic == "/lidar_localizer/aligned_cloud":
            data_array = np.frombuffer(msg.data, dtype=np.uint8)
            point_cloud_data.append(data_array)
        elif topic == "/lidar_localizer/lidar_pose":
            position = msg.pose.pose.position
            orientation = msg.pose.pose.orientation
            lidar_transform_data.append([
                position.x, position.y, position.z,
                orientation.x, orientation.y, orientation.z, orientation.w
            ])

    bag.close()

    # Pad point cloud data to ensure uniform length
    max_length = max(len(x) for x in point_cloud_data)
    padded_point_clouds = pad_sequences(point_cloud_data, maxlen=max_length, dtype='uint8', padding='post')

    return np.array(padded_point_clouds), pd.DataFrame(lidar_transform_data, columns=['pos_x', 'pos_y', 'pos_z', 'ori_x', 'ori_y', 'ori_z', 'ori_w'])


def calculate_mean_percentage_error(actual, predicted):
    non_zero_mask = actual != 0
    percentage_errors = np.where(non_zero_mask, 100 * (actual - predicted) / actual, 0)
    mean_percentage_errors = np.mean(percentage_errors, axis=0)
    return mean_percentage_errors


# Load the pre-trained models
slfn_model = load_model('slfn_model.h5')
#svm_model = joblib.load('model_09-07-24.joblib')  # Make sure you've saved the SVM model using joblib

# Load test data
X_test, y_test = extract_data_from_bag('Issue_ID_4_2024_06_13_07_47_15.bag')

 
 # Ensure y_test is a NumPy array and properly reshaped
if isinstance(y_test, pd.DataFrame):
     y_test = y_test.values  # Convert to NumPy array as it's a DataFrame

# Use the CNN model to extract features
cnn_features = slfn_model.predict(X_test)

# Use the pre-trained SVM to make predictions
#predicted_points = svm_model.predict(cnn_features.reshape(-1, 1))
#predicted_points = predicted_points.reshape(-1, 7)  # Assuming each prediction consists of 7 elements

predicted_points = cnn_features
actual_points = y_test

# Output predictions
print("Predicted LiDAR Points:", predicted_points)
print("Actual LiDAR Points:", actual_points)

# Plotting the x, y, z coordinates
x_actual = actual_points[:, 0]
y_actual = actual_points[:, 1]
z_actual = actual_points[:, 2]
x_pred = predicted_points[:, 0]
y_pred = predicted_points[:, 1]
z_pred = predicted_points[:, 2]


def plot_and_save(x_actual, y_actual, z_actual, x_pred, y_pred, z_pred, output_dir):
    plt.figure(figsize=(24, 8))

    # Plot X Coordinates
    plt.subplot(1, 4, 1)
    plt.scatter(x_actual, x_pred, c='blue')
    plt.title('X Coordinates')
    plt.xlabel('Actual X')
    plt.ylabel('Predicted X')
    plt.grid(True)

    # Plot Y Coordinates
    plt.subplot(1, 4, 2)
    plt.scatter(y_actual, y_pred, c='red')
    plt.title('Y Coordinates')
    plt.xlabel('Actual Y')
    plt.ylabel('Predicted Y')
    plt.grid(True)

    # Plot Z Coordinates
    plt.subplot(1, 4, 3)
    plt.scatter(z_actual, z_pred, c='green')
    plt.title('Z Coordinates')
    plt.xlabel('Actual Z')
    plt.ylabel('Predicted Z')
    plt.grid(True)

    # Plot X-Y Trajectories
    plt.subplot(1, 4, 4)
    plt.scatter(x_actual, y_actual, c='blue', label='Actual Trajectory', alpha=0.9)
    plt.scatter(x_pred, y_pred, c='red', label='Predicted Trajectory', alpha=0.3)
    plt.title('X-Y Trajectories')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'coordinate_comparisons.png'))
    plt.close()  # Close the plot to free up memory

# Example usage of the plot and save function
plot_and_save(x_actual, y_actual, z_actual, x_pred, y_pred, z_pred, current_folder)

#launch_plotjuggler('Issue_ID_4_2024_06_13_07_47_15.bag')

# Calculate mean percentage errors for each element
mean_percentage_errors = calculate_mean_percentage_error(actual_points, predicted_points)
print("Mean Percentage Errors for each element:", mean_percentage_errors)
