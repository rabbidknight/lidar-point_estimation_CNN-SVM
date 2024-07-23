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
from mpl_toolkits.mplot3d import Axes3D  # This is needed for '3d' projection
from scipy.spatial.transform import Rotation as R


class TrainingLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logger.info(f'Epoch {epoch + 1}: Loss: {logs["loss"]}, Val Loss: {logs["val_loss"]}')


# Current timestamp to create a unique folder
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
current_folder = f"slfn_test__{current_time}"
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













def quaternion_to_euler(quaternion):
    """Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw)."""
    r = R.from_quat(quaternion)
    return r.as_euler('xyz', degrees=False)  # Return Euler angles in radians

def create_transformation_matrix(roll, pitch, yaw, tx, ty, tz):
    """Create a 4x4 transformation matrix from Euler angles and translation vector."""
    # Rotation matrices around the x, y, and z axis
    R_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(roll), -np.sin(roll), 0],
        [0, np.sin(roll), np.cos(roll), 0],
        [0, 0, 0, 1]
    ])

    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch), 0],
        [0, 1, 0, 0],
        [-np.sin(pitch), 0, np.cos(pitch), 0],
        [0, 0, 0, 1]
    ])

    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0, 0],
        [np.sin(yaw), np.cos(yaw), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Translation matrix
    T = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])
    # Combined rotation matrix
    R = np.dot(np.dot(R_z, R_y), R_x)

    # Full transformation matrix
    transformation_matrix = np.dot(T, R)
    return transformation_matrix

def extract_and_transform_data(bag_file, seq_offset):
    """Extract data from a ROS bag and apply transformations based on LiDAR poses."""
    bag = rosbag.Bag(bag_file)
    point_cloud_data = {}
    lidar_transform_data = {}

    # Extraction step
    for topic, msg, t in bag.read_messages():
        if topic == "/lidar_localizer/aligned_cloud":
            data_array = np.frombuffer(msg.data, dtype=np.uint8)
            point_cloud_data[msg.header.seq] = data_array
        elif topic == "/lidar_localizer/lidar_pose":
            position = msg.pose.pose.position
            orientation = msg.pose.pose.orientation
            lidar_transform_data[msg.header.seq] = [
                position.x, position.y, position.z,
                orientation.x, orientation.y, orientation.z, orientation.w
            ]

    bag.close()

    # Sync and reshape
    synced_point_clouds = []
    synced_poses = []
    for seq, cloud in point_cloud_data.items():
        corresponding_pose_seq = seq + seq_offset
        if corresponding_pose_seq in lidar_transform_data:
            pose = lidar_transform_data[corresponding_pose_seq]
            synced_point_clouds.append(cloud)
            synced_poses.append(pose)
        else:
            synced_point_clouds.append(cloud)
            synced_poses.append([np.nan]*7)  # Handle missing data with NaNs

    # Transformation
    transformed_point_clouds = []
    transformed_poses = []
    for cloud, pose in zip(synced_point_clouds, synced_poses):
        # Ensure cloud length is divisible by 3
        needed_padding = (-len(cloud) % 3)
        if needed_padding:
            cloud = np.pad(cloud, (0, needed_padding), 'constant', constant_values=0)
        reshaped_cloud = cloud.reshape(-1, 3)    
        # Get rotation matrix and translation vector from the lidar pose
        euler_angles = quaternion_to_euler([pose[6], pose[3], pose[4], pose[5]])   #x y z w
        transformed_poses.append(create_transformation_matrix(*euler_angles, pose[0], pose[1], pose[2]))
        for i in range(len(cloud)):
        # Get rotation matrix and translation vector from the point cloud
            temp1 = (create_transformation_matrix(0, 0, 0, pose[0], pose[1], pose[2]))
            transformed_point_clouds[i].append(np.dot(temp1, transformed_poses))


    return transformed_point_clouds, transformed_poses









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


def plot3d_point_clouds(transformed_point_clouds, current_folder):
    """
    Plot transformed 3D point clouds.

    Args:
    transformed_point_clouds (list of np.array): List of point clouds after transformation.
    current_folder (str): Path to the directory where plot should be saved.
    """
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each point in the transformed point clouds
    for cloud in transformed_point_clouds:
        ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], alpha=0.5, color='b')

    ax.set_title('Transformed Point Clouds X-Y-Z Scatter')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(['Transformed Point Clouds'])

    plt.tight_layout()
    plot_filename = os.path.join(current_folder, '3d_transformed_point_clouds_plot.png')
    plt.savefig(plot_filename)
    plt.close()  # Close the plot to free up memory and save the file
    logger.info(f"3D point cloud plots saved to {plot_filename}")

    # Optionally display the plot
    plt.show()

# Example usage:
# plot3d_point_clouds(transformed_clouds, current_folder)



def plot2d_lidar_positions(actual, predicted):
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




















def create_slfn_model():
    model = Sequential([
        #Flatten(), # Flatten the input if it is not already 1D
        Dense(16, activation='relu'),  # Hidden layer with 16 units and ReLU activation
        BatchNormalization(),  # Batch normalization layer
        Dense(32, activation='relu'),  # Hidden layer with 32 units and ReLU activation
        BatchNormalization(),  # Batch normalization layer
        Dense(64, activation='relu'),  # Hidden layer with 64 units and ReLU activation
        BatchNormalization(),  # Batch normalization layer
        Dense(32, activation='relu'),  # Hidden layer with 32 units and ReLU activation
        BatchNormalization(),  # Batch normalization layer
        Dense(16, activation='relu'),  # Hidden layer with 16 units and ReLU activation
        BatchNormalization(),  # Batch normalization layer
        Dropout(0.2),  # Dropout layer with 20% rate
        Dense(7, activation='linear')  # Output layer with 7 units (no activation for regression)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    #LOGGING
    #for layer in model.layers:
    #   logger.info(f'Layer {layer.name} - Type: {layer.__class__.__name__}, Output Shape: {layer.output_shape}, Activation: {getattr(layer, "activation", None).__name__ if hasattr(layer, "activation") else "N/A"}')
    logger.info("optimizer:", model.optimizer, "loss:", model.loss, "metrics:", model.metrics_names, "learning rate:", model.optimizer.lr)

    return model

def add_channel_dimension(batch):
    # Convert list to a numpy array and add a new axis at the end to act as the channel dimension
    batch_array = np.array([np.array(x) for x in batch])
    return np.expand_dims(batch_array, axis=-1)


def prepare_batches_for_training(point_clouds, batch_size):
    batched_data = []

    for i in range(0, len(point_clouds), batch_size):
        # Extract the batch
        batch = point_clouds[i:i + batch_size]

        # Concatenate all arrays in the batch
        concatenated = np.concatenate(batch)

        # Pad the concatenated array if it's not divisible by 3
        needed_padding = (-len(concatenated)) % 3
        if needed_padding:
            concatenated = np.pad(concatenated, (0, needed_padding), mode='constant')
            logger.info(f"Batch {i // batch_size + 1} was padded with {needed_padding} zeros to make length divisible by 3.")

        # Reshape the array into (x, y, z)
        reshaped = concatenated.reshape(-1, 3)

        batched_data.append(reshaped)

    return batched_data

def manual_split(data, labels, test_ratio=0.15):
    total_samples = len(data)
    split_idx = int(total_samples * (1 - test_ratio))
    X_train, X_test = data[:split_idx], data[split_idx:]
    y_train, y_test = labels[:split_idx], labels[split_idx:]
    return X_train, X_test, y_train, y_test

def train_and_predict(bag_file):
    seq_offset = 25  # Offset to synchronize point clouds and poses
    point_clouds, poses = extract_and_transform_data(bag_file, seq_offset)
    plot3d_point_clouds(point_clouds, current_folder)
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = manual_split(point_clouds, poses)

    # Create and compile the model
    model = create_slfn_model()
"""
    model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1, validation_data=((X_test), y_test), callbacks=[TrainingLogger()])
    # Save model
    model.save(os.path.join(current_folder, 'slfn_model.h5'))
    logger.info("Model saved to %s", os.path.join(current_folder, 'slfn_model.h5'))

    for idx, pc in enumerate(X_test):
        print(f"Shape of point cloud {idx+1}: {pc.shape}")

    # After training, predict on the test set and handle each test point cloud individually
    predictions = []
    for test_point_cloud in (X_test):
        print("Number of point clouds in X_test:", len(X_test))
        test_point_cloud = test_point_cloud[np.newaxis, ..., np.newaxis]  # Add necessary dimensions
        prediction = model.predict(test_point_cloud)
        print(f"Prediction shape: {prediction.shape}")  # Check the shape of each prediction
        predictions.append(prediction)  # Append the single prediction result

    plot2d_lidar_positions(y_test, predictions)  # Call plotting function
        # Evaluate the model at the end of each epoch
        #val_loss = model.evaluate(X_test, y_test, verbose=1)
        #print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")

"""



train_and_predict('Issue_ID_4_2024_06_13_07_47_15.bag')


#visualize_results(predicted_points, actual_points)




            #THE TWO TOPICS IN THE BAG FILE ARE:
            # /lidar_localizer/lidar_pose                                                               300 msgs    : geometry_msgs/PoseWithCovarianceStamped               
            #/lidar_localizer/aligned_cloud                                                            300 msgs    : sensor_msgs/PointCloud2                               

