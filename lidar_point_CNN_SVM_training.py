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


def extract_data_from_bag(bag_file, batch_size):
    bag = rosbag.Bag(bag_file)
    point_cloud_data = []
    lidar_transform_data = []

    point_cloud_seq = []
    lidar_pose_seq = []

    # Counters and diagnostics
    num_point_clouds = 0
    num_lidar_poses = 0
    for topic, msg, t in bag.read_messages():
        if topic == "/lidar_localizer/aligned_cloud":
            data_array = np.frombuffer(msg.data, dtype=np.uint8)
            point_cloud_data.append(data_array)
            point_cloud_seq.append(msg.header.seq)
        elif topic == "/lidar_localizer/lidar_pose":
            position = msg.pose.pose.position
            orientation = msg.pose.pose.orientation
            lidar_transform_data.append([
                position.x, position.y, position.z,
                orientation.x, orientation.y, orientation.z, orientation.w
            ])
            lidar_pose_seq.append(msg.header.seq)

    bag.close()

    # Pad point cloud data to ensure uniform length
    max_length = max(len(x) for x in point_cloud_data)
    padded_point_clouds = pad_sequences(point_cloud_data, maxlen=max_length, dtype='uint8', padding='post')

    # Synchronize data based on sequence numbers
    synced_point_clouds = []
    synced_poses = []
    pose_dict = dict(zip(lidar_pose_seq, lidar_transform_data))

    for seq in point_cloud_seq:
        if seq in pose_dict:
            synced_point_clouds.append(padded_point_clouds[point_cloud_seq.index(seq)])
            synced_poses.append(pose_dict[seq])

    plot3d_point_clouds(synced_point_clouds, synced_poses)

    # Organize point clouds into batches
    padded_point_clouds = []
    max_lengths = []
    logger.info("synced_point_clouds")
    print("synced_poses", synced_poses)
    logger.info("synced_poses")
    '''
    for i in range(0, len(point_cloud_data), batch_size):
        batch = synced_point_clouds[i:i + batch_size]
        max_length = max(len(pc) for pc in batch)
        max_lengths.append(max_length)
        padded_batch = pad_sequences(batch, maxlen=max_length, dtype='uint8', padding='post')
        padded_point_clouds.extend(padded_batch)

    # Log information about the padding and batch processing
    logger.info(f"Processed {len(padded_point_clouds)} point clouds into batches of size {batch_size}")
    for i, ml in enumerate(max_lengths):
        logger.info(f"Batch {i + 1} padded to max length: {ml}")
    #logger.info("After padding:")
    #logger.info("Shape of padded point cloud data: %s", padded_point_clouds.shape)
    #if padded_point_clouds.size > 0:
    #    logger.info("Sample data from the first padded point cloud entry: %s", padded_point_clouds[0][:10])
    '''
    return (synced_point_clouds), pd.DataFrame(synced_poses, columns=['pos_x', 'pos_y', 'pos_z', 'ori_x', 'ori_y', 'ori_z', 'ori_w'])

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

def plot3d_point_clouds(batched_point_clouds, lidar_poses):
    # Set up the plot for 3D scatter
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')


    # Loop through each batch of point clouds and their corresponding LiDAR positions
    for batch_index, (point_clouds, pose) in enumerate(zip(batched_point_clouds, lidar_poses)):
        # Determine how many zeros to pad to make the array divisible by 3
        needed_padding = (-len(point_clouds)) % 3
        if needed_padding > 0:
            # Pad with zeros at the end of the array
            point_clouds = np.pad(point_clouds, (0, needed_padding), mode='constant')
            logger.info(f"Batch {batch_index + 1} was padded with {needed_padding} zeros to make length divisible by 3.")

        # Reshape the array now that it is guaranteed to be divisible by 3
        reshaped_clouds = point_clouds.reshape(-1, 3)

        # Scatter each point adjusted by its corresponding LiDAR position
        for point in reshaped_clouds:
            if batch_index < 1:
                for pose in lidar_poses:
                    x = point[0] + pose[0]
                    y = point[1] + pose[1]
                    z = point[2] + pose[2]
                    ax.scatter(x, y, z, color='b', alpha=0.5)

    ax.set_title('Point Clouds X-Y-Z Scatter')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(['Point Clouds'])

    plt.tight_layout()
    plot_filename = os.path.join(current_folder, '3d_point_clouds_plot.png')
    plt.savefig(plot_filename)
    plt.close()
    logger.info("3D Point cloud plots saved to %s", plot_filename)

    # Optionally display the plot
    plt.show()


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
        Dense(16, activation='relu'),  # Hidden layer with 128 units and ReLU activation
        BatchNormalization(),  # Batch normalization layer
        Dense(32, activation='relu'),  # Hidden layer with 128 units and ReLU activation
        BatchNormalization(),  # Batch normalization layer
        
        Dropout(0.2),  # Dropout layer with 20% rate
        Dense(7, activation='linear')  # Output layer with 7 units (no activation for regression)
    ])
    '''
    # Logging each layer's configuration
    for layer in model.layers:
        logger.info(f'Layer {layer.name} - Type: {layer.__class__.__name__}, Output Shape: {layer.output_shape}, Activation: {getattr(layer, "activation", None).__name__ if hasattr(layer, "activation") else "N/A"}')
    '''
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    #logger.info("optimizer:", model.optimizer, "loss:", model.loss, "metrics:", model.metrics_names, "learning rate:", model.optimizer.lr)

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
    point_clouds, poses = extract_data_from_bag(bag_file, batch_size=1)

    # Convert poses to a proper numpy array if it's not already
    if isinstance(poses, pd.DataFrame):
        poses = poses.values

    #plot3d_point_clouds(point_clouds, poses)  # Call 3D plotting function

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = manual_split(point_clouds, poses)
'''

    # Create and compile the model
    model = create_slfn_model()

    # Train the model
    for epoch in range(1):  # Adjust the number of epochs as necessary
        print(f"Starting epoch {epoch+1}")
        for point_cloud, label in zip(X_train, y_train):
            point_cloud = point_cloud[np.newaxis, ..., np.newaxis]  # Adding necessary dimensions
            label = label.reshape(1,-1)  # Reshape label to match the expected input and  also convert to numpy array
            model.train_on_batch(point_cloud, label)

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
'''



batch_size = 1
train_and_predict('Issue_ID_4_2024_06_13_07_47_15.bag')


#visualize_results(predicted_points, actual_points)




            #THE TWO TOPICS IN THE BAG FILE ARE:
            # /lidar_localizer/lidar_pose                                                               300 msgs    : geometry_msgs/PoseWithCovarianceStamped               
            #/lidar_localizer/aligned_cloud                                                            300 msgs    : sensor_msgs/PointCloud2                               

