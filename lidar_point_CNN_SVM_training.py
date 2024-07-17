import struct

from sklearn import logger
import rosbag
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import datetime
import logging
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dropout, UpSampling1D, Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.preprocessing.sequence import pad_sequences
from joblib import dump


def calculate_mean_percentage_error(actual, predicted):
    # Avoid division by zero and handle cases where actual values are zero
    non_zero_mask = actual != 0
    percentage_errors = np.where(non_zero_mask, 100 * (actual - predicted) / actual, 0)
    mean_percentage_errors = np.mean(percentage_errors, axis=0)
    return mean_percentage_errors


def extract_data_from_bag(bag_file):
    bag = rosbag.Bag(bag_file)
    point_cloud_data = []
    lidar_transform_data = []

    point_cloud_times = []
    lidar_times = []

    # Counters and diagnostics
    num_point_clouds = 0
    num_lidar_poses = 0

    for topic, msg, t in bag.read_messages():
        if topic == "/lidar_localizer/aligned_cloud":
            data_array = np.frombuffer(msg.data, dtype=np.uint8)
            point_cloud_data.append(data_array)
            point_cloud_times.append(msg.header.stamp.to_nsec())
            num_point_clouds += 1
        elif topic == "/lidar_localizer/lidar_pose":
            position = msg.pose.pose.position
            orientation = msg.pose.pose.orientation
            lidar_transform_data.append([
                position.x, position.y, position.z,
                orientation.x, orientation.y, orientation.z, orientation.w
            ])
            lidar_times.append(msg.header.stamp.to_nsec())
            num_lidar_poses += 1

    bag.close()

    #Data diagnostics
    logger.info("Number of point cloud entries read: %d", num_point_clouds)
    logger.info("Number of LiDAR pose entries read: %d", num_lidar_poses)
    if point_cloud_data:
        logger.info("Number of input parameters in the first point cloud: %d", len(point_cloud_data[0]))
        logger.info("Data type of point cloud parameters: %s", point_cloud_data[0].dtype)
    if lidar_transform_data:
        logger.info("Number of input parameters in first LiDAR pose: %d", len(lidar_transform_data[0]))
        logger.info("Data type of LiDAR pose parameters: %s", type(lidar_transform_data[0][0]))
        
    # Pad point cloud data to ensure uniform length
    max_length = max(len(x) for x in point_cloud_data)
    padded_point_clouds = pad_sequences(point_cloud_data, maxlen=max_length, dtype='uint8', padding='post')

    logger.info("After padding:")
    logger.info("Shape of padded point cloud data: %s", padded_point_clouds.shape)
    if padded_point_clouds.size > 0:
        logger.info("Sample data from the first padded point cloud entry: %s", padded_point_clouds[0][:10])


    # Synchronize data based on closest timestamps
    synced_point_clouds = []
    synced_poses = []
    for i, pc_time in enumerate(point_cloud_times):
        closest_idx = np.argmin([abs(pc_time - lt) for lt in lidar_times])
        synced_point_clouds.append(padded_point_clouds[i])
        synced_poses.append(lidar_transform_data[closest_idx])

    return np.array(synced_point_clouds), pd.DataFrame(synced_poses, columns=['pos_x', 'pos_y', 'pos_z', 'ori_x', 'ori_y', 'ori_z', 'ori_w'])

def visualize_results(predicted_points, actual_points, folder):
    # Extract x, y, z coordinates
    x_actual = actual_points[:, 0]
    y_actual = actual_points[:, 1]
    z_actual = actual_points[:, 2]
    x_pred = predicted_points[:, 0]
    y_pred = predicted_points[:, 1]
    z_pred = predicted_points[:, 2]

    # Initialize the plotting
    plt.figure(figsize=(18, 10))

    # Plot X Coordinates
    plt.subplot(1, 4, 1)
    plt.scatter(x_actual, x_pred, c='blue')
    plt.title('X Coordinates Comparison')
    plt.xlabel('Actual X')
    plt.ylabel('Predicted X')

    # Plot Y Coordinates
    plt.subplot(1, 4, 2)
    plt.scatter(y_actual, y_pred, c='red')
    plt.title('Y Coordinates Comparison')
    plt.xlabel('Actual Y')
    plt.ylabel('Predicted Y')

    # Plot Z Coordinates
    plt.subplot(1, 4, 3)
    plt.scatter(z_actual, z_pred, c='green')
    plt.title('Z Coordinates Comparison')
    plt.xlabel('Actual Z')
    plt.ylabel('Predicted Z')

    # Plot X-Y Trajectories
    plt.subplot(1, 4, 4)
    plt.scatter(x_actual, y_actual, c='blue', label='Actual Trajectory', alpha=0.6)
    plt.scatter(x_pred, y_pred, c='red', label='Predicted Trajectory', alpha=0.6)
    plt.title('X-Y Trajectories')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()

    plt.tight_layout()
    # Save the full figure
    plot_filename = os.path.join(folder, 'coordinate_comparison.png')
    plt.savefig(plot_filename)
    plt.close()  # Close the plot to free up memory
    logger.info(f"Graphs saved to {plot_filename}")

# visualize_results(predicted_points, actual_points, current_folder)



def create_pointnet_model(input_shape):
    model = Sequential([
        Conv1D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.25),
        Conv1D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', input_shape=input_shape),
        BatchNormalization(),
        #UpSampling1D(2),
        Conv1D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.25),
        Conv1D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal', input_shape=input_shape),
        Flatten(),  # Flatten the output to make it a 1D vector
        Dense(7)  # Final layer with 7 units, one for each target variable
    ])
    return model

"""
# Data preparation logic for pointnet model
def prepare_data(point_cloud_df, num_points):
    # This assumes each point is a separate sample
    if len(point_cloud_df) < num_points:
        print(f"Not enough points, only {len(point_cloud_df)} available.")
        return None, None
    
    # Sample points if more points are available than needed
    sampled_df = point_cloud_df.sample(n=num_points)
    X = np.array(sampled_df)
    y = np.array([sampled_df['x'].mean(), sampled_df['y'].mean(), sampled_df['z'].mean()])  # example target

    return X, y
"""

def train_and_predict(bag_file):
    #point_cloud_df, lidar_df = extract_data_from_bag(bag_file)
    point_clouds, poses = extract_data_from_bag(bag_file)

    #X, y = prepare_data(point_cloud_df, num_points)

    # Check if the arrays are empty. Using len() function for numpy arrays.
    if len(point_clouds) == 0 or len(poses) == 0:
        print("No data available for training. Please check the ROS bag file.")
        return None, None
    
    #y = y.reshape(1, -1)

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(point_clouds, poses, test_size=0.15, random_state=42)

    input_shape = (X_train.shape[1], 1)  # Assuming data is 1D
    model = create_pointnet_model(input_shape)
    optimizer = Adam(learning_rate=0.015)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=8, min_lr=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.15, callbacks=[reduce_lr], verbose = 1)

    model.save('model_12-07-24.h5')  # Save the model as an HDF5 file

    feature_model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-1].output)
    train_features = feature_model.predict(X_train)
    test_features = feature_model.predict(X_test)


    # Ensure y_train is a NumPy array and properly reshaped
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.values  # Convert to NumPy array as it's a DataFrame
 
     # Ensure y_test is a NumPy array and properly reshaped
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.values  # Convert to NumPy array as it's a DataFrame
 

    #train_features = train_features.ravel()
    #test_features = test_features.ravel()

    svm = SVR()
    svm.fit(train_features.reshape(-1, 1), y_train.ravel())
    dump(svm, 'model_09-07-24.joblib')  # Save the SVM model using joblib

    predicted_lidar_points = svm.predict(test_features.reshape(-1, 1))
    predicted_lidar_points = predicted_lidar_points.reshape(-1, 7)  # Reshape predictions into (n_samples, 7) format
    actual_points = y_test # Ensure actual data is also reshaped similarly

    return predicted_lidar_points, actual_points

predicted_points, actual_points = train_and_predict('Issue_ID_4_2024_06_13_07_47_15.bag')

#visualize_results(predicted_points, actual_points)

            #THE TWO TOPICS IN THE BAG FILE ARE:
            # /lidar_localizer/lidar_pose                                                               300 msgs    : geometry_msgs/PoseWithCovarianceStamped               
            #/lidar_localizer/aligned_cloud                                                            300 msgs    : sensor_msgs/PointCloud2                               
