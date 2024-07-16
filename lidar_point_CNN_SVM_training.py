import struct
import rosbag
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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
import os
import datetime
import logging
import sys
from keras.callbacks import Callback

class TrainingLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logger.info(f'Epoch {epoch + 1}: Loss: {logs["loss"]}, Val Loss: {logs["val_loss"]}')


# Current timestamp to create a unique folder
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
current_folder = f"test_results_{current_time}"
os.makedirs(current_folder, exist_ok=True)  # Create the directory if it doesn't exist

# Setup logging
logger = logging.getLogger('LidarTestLogger')
logger.setLevel(logging.INFO)

# Log file
log_filename = os.path.join(current_folder, 'test_log.txt')
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

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

     # Data diagnostics
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

def visualize_results(predicted_points, actual_points):
    logger.info("Predicted LiDAR Points: %s", predicted_points)
    logger.info("Actual LiDAR Points: %s", actual_points)

    # Extract x, y, z coordinates
    x_actual = actual_points[:, 0]
    y_actual = actual_points[:, 1]
    z_actual = actual_points[:, 2]
    x_pred = predicted_points[:, 0]
    y_pred = predicted_points[:, 1]
    z_pred = predicted_points[:, 2]

    # Plotting
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.scatter(x_actual, x_pred, c='blue')
    plt.title('X Coordinates')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    plt.subplot(1, 3, 2)
    plt.scatter(y_actual, y_pred, c='red')
    plt.title('Y Coordinates')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    plt.subplot(1, 3, 3)
    plt.scatter(z_actual, z_pred, c='green')
    plt.title('Z Coordinates')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    plt.tight_layout()
    plt.show()

    mean_percentage_errors = calculate_mean_percentage_error(actual_points, predicted_points)
    logger.info("Mean Percentage Errors for each element: %s", mean_percentage_errors)


def create_slfn_model(input_shape):
    model = Sequential([
        Flatten(input_shape=input_shape),  # Flatten the input if it is not already 1D
        Dense(16, activation='relu'),  # Hidden layer with 128 units and ReLU activation
        Dense(32, activation='relu'),  # Hidden layer with 128 units and ReLU activation
        Dense(64, activation='relu'),  # Hidden layer with 128 units and ReLU activation
        Dense(128, activation='relu'),  # Hidden layer with 128 units and ReLU activation
        Dense(256, activation='relu'),  # Hidden layer with 128 units and ReLU activation
        Dense(512, activation='relu'),  # Hidden layer with 128 units and ReLU activation
        Dense(1024, activation='relu'),  # Hidden layer with 128 units and ReLU activation
        Dense(2048, activation='relu'),  # Hidden layer with 128 units and ReLU activation
        Dense(4096, activation='relu'),  # Hidden layer with 128 units and ReLU activation
        Dense(8192, activation='relu'),  # Hidden layer with 128 units and ReLU activation
        Dense(4096, activation='relu'),  # Hidden layer with 128 units and ReLU activation
        Dense(2048, activation='relu'),  # Hidden layer with 128 units and ReLU activation
        Dense(1024, activation='relu'),  # Hidden layer with 128 units and ReLU activation
        Dense(512, activation='relu'),  # Hidden layer with 128 units and ReLU activation
        Dense(256, activation='relu'),  # Hidden layer with 128 units and ReLU activation
        Dense(128, activation='relu'),  # Hidden layer with 128 units and ReLU activation
        Dense(64, activation='relu'),  # Hidden layer with 128 units and ReLU activation
        Dense(32, activation='relu'),  # Hidden layer with 128 units and ReLU activation
        Dense(16, activation='relu'),  # Hidden layer with 128 units and ReLU activation
        Dense(7, activation='linear')  # Output layer with 7 units (no activation for regression)
    ])

    # Logging each layer's configuration
    for layer in model.layers:
        logger.info(f'Layer {layer.name} - Type: {layer.__class__.__name__}, Output Shape: {layer.output_shape}, Activation: {getattr(layer, "activation", None).__name__ if hasattr(layer, "activation") else "N/A"}')

    return model


def train_and_predict(bag_file):
    point_clouds, poses = extract_data_from_bag(bag_file)
    if len(point_clouds) == 0 or len(poses) == 0:
        print("No data available for training. Please check the ROS bag file.")
        return None, None
    
    X_train, X_test, y_train, y_test = train_test_split(point_clouds, poses, test_size=0.15, random_state=42)

    input_shape = (X_train.shape[1], 1)  # Adjust depending on your actual input shape
    model = create_slfn_model(input_shape)
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')


    # Reduce learning rate when a metric has stopped improving
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.0001, verbose=1)

    # Custom logger for training
    training_logger = TrainingLogger()

    # Train the model
    model.fit(
        X_train, y_train, epochs=150, batch_size=4, validation_split=0.2,
        callbacks=[reduce_lr, training_logger]  # Add the ReduceLROnPlateau and custom logging callback

    )
    # Save model
    model.save(os.path.join(current_folder, 'slfn_model.h5'))

    # Predict
    predicted_points = model.predict(X_test)
    actual_points = y_test.values

    return predicted_points, actual_points

predicted_points, actual_points = train_and_predict('Issue_ID_4_2024_06_13_07_47_15.bag')
#visualize_results(predicted_points, actual_points)




            #THE TWO TOPICS IN THE BAG FILE ARE:
            # /lidar_localizer/lidar_pose                                                               300 msgs    : geometry_msgs/PoseWithCovarianceStamped               
            #/lidar_localizer/aligned_cloud                                                            300 msgs    : sensor_msgs/PointCloud2                               
