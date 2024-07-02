import struct
import rosbag
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dropout, UpSampling1D, Dense
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split


def extract_data_from_bag(bag_file):
    bag = rosbag.Bag(bag_file)
    point_cloud_data = []
    lidar_transform_data = []

    for topic, msg, t in bag.read_messages():
        if topic == "/lidar_localizer/aligned_cloud":
            num_points = msg.width
            for i in range(num_points):
                offset = i * msg.point_step
                x, y, z = struct.unpack_from('<fff', msg.data, offset)  # Ensuring correct endianness and format
                point_cloud_data.append([x, y, z])
        elif topic == "/lidar_localizer/lidar_pose":
            position = msg.pose.pose.position
            orientation = msg.pose.pose.orientation
            lidar_transform_data.append([
                position.x, position.y, position.z,
                orientation.x, orientation.y, orientation.z, orientation.w
            ])

    bag.close()
    return pd.DataFrame(point_cloud_data, columns=['x', 'y', 'z']), pd.DataFrame(lidar_transform_data, columns=['pos_x', 'pos_y', 'pos_z', 'ori_x', 'ori_y', 'ori_z', 'ori_w'])


def create_pointnet_model(num_points):
    model = Sequential([
        Conv1D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal', input_shape=(num_points, 3)),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.25),
        Conv1D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.25),
        UpSampling1D(2),
        Conv1D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal'),
        BatchNormalization(),
        UpSampling1D(2),
        Conv1D(1, 3, activation='sigmoid', padding='same')
    ])
    return model

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


def train_and_predict(bag_file, num_points):
    point_cloud_df, lidar_df = extract_data_from_bag(bag_file)

    X, y = prepare_data(point_cloud_df, num_points)

    if point_cloud_df.empty or lidar_df.empty:
        print("No data available for training. Please check the ROS bag file.")
        return None, None

    if len(point_cloud_df) < num_points:
        print(f"Not enough points, only {len(point_cloud_df)} available.")
        return None, None

    y = y.reshape(1, -1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = create_pointnet_model(num_points)
    optimizer = Adam(lr=0.0015)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.15, patience=5, min_lr=0.0001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.15, callbacks=[reduce_lr])

    feature_model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-4].output)
    train_features = feature_model.predict(X_train)
    test_features = feature_model.predict(X_test)

    svm = SVR()
    svm.fit(train_features, y_train.ravel())

    predicted_lidar_points = svm.predict(test_features)

    return predicted_lidar_points, y_test.ravel()

num_points = 1024  # Adjust based on your data
predicted_points, actual_points = train_and_predict('Issue_ID_4_2024_06_13_07_47_15.bag', num_points)
print("Predicted LiDAR Points:", predicted_points)
print("Actual LiDAR Points:", actual_points)

            #THE TWO TOPICS IN THE BAG FILE ARE:
            # /lidar_localizer/lidar_pose                                                               300 msgs    : geometry_msgs/PoseWithCovarianceStamped               
            #/lidar_localizer/aligned_cloud                                                            300 msgs    : sensor_msgs/PointCloud2                               
