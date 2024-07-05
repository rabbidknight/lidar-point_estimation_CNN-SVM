import struct
import rosbag
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dropout, UpSampling1D, Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.preprocessing.sequence import pad_sequences

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

    # Pad point cloud data to ensure uniform length
    max_length = max(len(x) for x in point_cloud_data)
    padded_point_clouds = pad_sequences(point_cloud_data, maxlen=max_length, dtype='uint8', padding='post')

    # Diagnostic print statements after padding
    print("After padding:")
    print("Shape of padded point cloud data:", padded_point_clouds.shape)
    if padded_point_clouds.size > 0:
        print("Sample data from the first padded point cloud entry:", padded_point_clouds[0][:10])  # show first 10 elements

    # Synchronize data based on closest timestamps
    synced_point_clouds = []
    synced_poses = []
    for i, pc_time in enumerate(point_cloud_times):
        closest_idx = np.argmin([abs(pc_time - lt) for lt in lidar_times])
        synced_point_clouds.append(padded_point_clouds[i])
        synced_poses.append(lidar_transform_data[closest_idx])

    return np.array(synced_point_clouds), pd.DataFrame(synced_poses, columns=['pos_x', 'pos_y', 'pos_z', 'ori_x', 'ori_y', 'ori_z', 'ori_w'])


def create_pointnet_model(input_shape):
    model = Sequential([
        Conv1D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.25),
        Conv1D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.25),
        Conv1D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', input_shape=input_shape),
        BatchNormalization(),
        Conv1D(1, 3, activation='relu', padding='same', kernel_initializer='he_normal', input_shape=input_shape),
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
    X_train, X_test, y_train, y_test = train_test_split(point_clouds, poses, test_size=0.2, random_state=42)

    input_shape = (X_train.shape[1], 1)  # Assuming data is 1D
    model = create_pointnet_model(input_shape)
    optimizer = Adam(learning_rate=0.0015)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=3, min_lr=0.0001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.15, callbacks=[reduce_lr], verbose = 1)

    feature_model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-1].output)
    train_features = feature_model.predict(X_test)
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
    svm.fit(train_features.reshape(-1, 1), y_test.reshape(-1, 1))

    predicted_lidar_points = svm.predict(test_features.reshape(-1, 1))

    return predicted_lidar_points, y_test.ravel()

predicted_points, actual_points = train_and_predict('Issue_ID_4_2024_06_13_07_47_15.bag')
print("Predicted LiDAR Points:", predicted_points)
print("Actual LiDAR Points:", actual_points)


# Calculate Mean Squared Error
mse = mean_squared_error(actual_points, predicted_points)
print("Mean Squared Error:", mse)

# Calculate Mean Absolute Error
mae = mean_absolute_error(actual_points, predicted_points)
print("Mean Absolute Error:", mae)

            #THE TWO TOPICS IN THE BAG FILE ARE:
            # /lidar_localizer/lidar_pose                                                               300 msgs    : geometry_msgs/PoseWithCovarianceStamped               
            #/lidar_localizer/aligned_cloud                                                            300 msgs    : sensor_msgs/PointCloud2                               
