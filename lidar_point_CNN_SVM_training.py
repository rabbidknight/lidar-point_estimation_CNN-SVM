import struct
import rosbag
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
#for RNN
from keras.layers import LSTM, Dense, Masking
from keras.regularizers import l2
#for CNN
from keras import Sequential
from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dropout, UpSampling1D, Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
#for SVM
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.preprocessing.sequence import pad_sequences
from joblib import dump
from sklearn.multioutput import MultiOutputRegressor

def calculate_mean_percentage_error(actual, predicted):
    # Avoid division by zero and handle cases where actual values are zero
    non_zero_mask = actual != 0
    percentage_errors = np.where(non_zero_mask, 100 * (actual - predicted) / actual, 0)
    mean_percentage_errors = np.mean(percentage_errors, axis=0)
    return mean_percentage_errors

def preprocess_point_cloud(data):
    """Converts raw byte data to a numpy array of floats shaped for RNN input."""
    data = np.frombuffer(data, dtype=np.uint8)
    data = data.view(np.float32)  # Interpret bytes as floats
    return data.reshape(-1, 3)  # Reshape to [n_points, 3] for x, y, z coordinates

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
    print("Number of point cloud entries read:", num_point_clouds)
    print("Number of LiDAR pose entries read:", num_lidar_poses)
    if point_cloud_data:
        print("Number of input parameters in the first point cloud:", len(point_cloud_data[0]))
        print("Data type of point cloud parameters:", point_cloud_data[0].dtype)
    if lidar_transform_data:
        print("Number of input parameters in first LiDAR pose:", len(lidar_transform_data[0]))
        print("Data type of LiDAR pose parameters:", type(lidar_transform_data[0][0]))

    # Synchronize data based on closest timestamps
    synced_point_clouds = []
    synced_poses = []
    for i, pc_time in enumerate(point_cloud_times):
        closest_idx = np.argmin([abs(pc_time - lt) for lt in lidar_times])
        synced_point_clouds.append(point_cloud_data[i])
        synced_poses.append(lidar_transform_data[closest_idx])

    return synced_point_clouds, pd.DataFrame(synced_poses, columns=['pos_x', 'pos_y', 'pos_z', 'ori_x', 'ori_y', 'ori_z', 'ori_w'])

def data_generator(X_data, y_data, batch_size):
    data_sorted = sorted(zip(X_data, y_data), key=lambda x: len(x[0]))
    X_sorted, y_sorted = zip(*data_sorted)

    for start_idx in range(0, len(X_sorted), batch_size):
        end_idx = min(start_idx + batch_size, len(X_sorted))
        batch_X = list(X_sorted[start_idx:end_idx])
        batch_y = list(y_sorted[start_idx:end_idx])
        batch_X_padded = pad_sequences(batch_X, padding='post', dtype='float32')
        yield batch_X_padded, np.array(batch_y)
        #yield batch_X_padded.reshape((len(batch_X_padded), -1, 3)), np.array(batch_y)

def create_rnn_model():
    model = Sequential([
        # Input LSTM layer with more units and return sequences to stack another LSTM layer
        LSTM(64, return_sequences=True, input_shape=(None, 1), kernel_regularizer=l2(0.01)),
        Dropout(0.2),  # Dropout for regularization
        
        # Additional LSTM Layer
        LSTM(8, return_sequences=False, kernel_regularizer=l2(0.01)),
        Dropout(0.04),  # Dropout for regularization

        Dense(7, activation='relu')  # Assuming 7 features to match with SVM input needs
    ])
    return model


def train_rnn_model(model, X_train, y_train, epochs=10, batch_size=16):
    print("no of epochs",epochs,"batch size",batch_size)
    """Trains the RNN model on provided data with dynamic batching."""
    model.compile(optimizer=Adam(learning_rate=0.015), loss='mean_squared_error')
    print("Model compiled")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs} started.")
        # Create the generator for the current epoch
        train_gen = data_generator(X_train, y_train, batch_size)
        # Initialize batch count
        batch_count = 0
        # Iterate through generated batches
        for X_batch, y_batch in train_gen:
            batch_count += 1
            print(f"Training on batch {batch_count} out of {len(X_train) // batch_size} and epoch {epoch + 1}/{epochs}")
            model.train_on_batch(X_batch, y_batch)
        print(f"Epoch {epoch + 1}/{epochs} completed.")
    # Save the trained RNN model at the end of training
    model.save('model_10-07-24.keras') 
    print("Model saved")


def extract_features(model, X_data, batch_size=32):
    """Extracts features from the RNN model in batches."""
    features = []
    for start_idx in range(0, len(X_data), batch_size):
        end_idx = min(start_idx + batch_size, len(X_data))
        batch_X = list(X_data[start_idx:end_idx])
        batch_X_padded = pad_sequences(batch_X, padding='post', dtype='float32')
        batch_features = model.predict(batch_X_padded)
        features.extend(batch_features)
    return np.array(features)

def train_svm(features, labels):
    svm = SVR()  # This is a simple SVM regressor which normally handles single output
    multi_output_svm = MultiOutputRegressor(svm)  # Wrap it in MultiOutputRegressor to handle multiple outputs
    multi_output_svm.fit(features, labels)  # No need to ravel; fit directly with labels as 2D array
    dump(multi_output_svm, 'model_12-07-24.joblib')
    return multi_output_svm

def predict_with_svm(model, svm, X_test):
    """Predicts using the SVM model on features extracted from the RNN."""
    features = extract_features(model, X_test)
    predictions = svm.predict(features)
    return predictions

point_clouds, poses = extract_data_from_bag('Issue_ID_4_2024_06_13_07_47_15.bag')

# Ensure point_clouds is a list of arrays with shape (n_points, 3)
#point_clouds = [np.array(cloud).reshape(-1, 3) for cloud in point_clouds]

    # Ensure labels are a NumPy array, not a DataFrame
if isinstance(poses, pd.DataFrame):
    poses = poses.values  # Convert DataFrame to NumPy array

print("Received data from bag")
model = create_rnn_model()
print("Created model")
train_rnn_model(model, point_clouds, poses)
print("Trained model")
svm = train_svm(extract_features(model, point_clouds), poses)
print("Trained SVM")
predicted_points = predict_with_svm(model, svm, point_clouds)
print("Predicted points")
#visualize_results(predicted_points, poses.values)

print("model layers are ",model.layers)
            #THE TWO TOPICS IN THE BAG FILE ARE:
            # /lidar_localizer/lidar_pose                                                               300 msgs    : geometry_msgs/PoseWithCovarianceStamped               
            #/lidar_localizer/aligned_cloud                                                            300 msgs    : sensor_msgs/PointCloud2                               
