import rosbag
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from keras import layers, models, Sequential, Input
#for pointnet
from keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dropout, UpSampling1D, Dense
#for CNN
#from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Sampling2D
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau

def extract_data_from_bag(bag_file):
    bag = rosbag.Bag(bag_file)
    point_cloud_data = []
    lidar_transform_data = []

    for topic, msg, t in bag.read_messages():
        if "/map" in topic:
            if hasattr(msg, "points"):  # Check if 'points' attribute is present
                for point in msg.points:
                    point_cloud_data.append([point.x, point.y, point.z])
        elif "/tf_static" in topic:
            for transform in msg.transforms:
                lidar_transform_data.append([
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w
                ])

    bag.close()
    return pd.DataFrame(point_cloud_data, columns=['x', 'y', 'z']), pd.DataFrame(lidar_transform_data, columns=['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'rot_w'])


# Function to print sample data
def print_sample_data(data, num_samples=5):
    print("Sample Data:")
    print(data.head(num_samples))

"""
# Define the CNN architecture
def build_cnn_model(num_points):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', num_points=num_points),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.25),
        UpSampling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),
        BatchNormalization(),
        UpSampling2D((2, 2)),
        Conv2D(1, (3, 3), activation='sigmoid', padding='same')
    ])
    return model
"""

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
        Conv1D(1, 3, activation='sigmoid', padding='same')  # Output layer; adjust depending on the output shape needed
    ])
    return model

# Training and Prediction Logic
def train_and_predict(bag_file, num_points):
    # Extract data from bag
    point_cloud_df, lidar_df = extract_data_from_bag(bag_file)
    #print_sample_data(point_cloud_df)
    #print_sample_data(lidar_df)

    # Check if there is any data to process
    if point_cloud_df.empty or lidar_df.empty:
        print("No data available for training. Please check the ROS bag file.")
        return None, None

    # Ensure all data fits the model input shape
    if len(point_cloud_df) < num_points:
        print(f"Not enough points, only {len(point_cloud_df)} available.")
        return None, None

    # Prepare data for the model
    X = np.array(point_cloud_df.sample(n=num_points))  # Randomly select 'num_points'
    y = np.array(lidar_df.mean()).reshape(1, -1)  # Average the lidar data or choose another representation

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X.reshape(1, num_points, 3), y, test_size=0.2, random_state=42)
    # Build and train the model
    model = create_pointnet_model(num_points)
    optimizer = Adam(lr=0.0015)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.15, patience=5, min_lr=0.0001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=25, batch_size=32, validation_split=0.15, callbacks=[reduce_lr])

    # Extract features with CNN for SVM training
    feature_model = models.Model(inputs=model.input, outputs=model.layers[-4].output)
    train_features = feature_model.predict(X_train)
    test_features = feature_model.predict(X_test)

    # Train SVM
    svm = SVR()
    svm.fit(train_features, y_train)

    # Make predictions on the test set
    predicted_lidar_points = svm.predict(test_features)

    return predicted_lidar_points, y_test

# Example usage
num_points = 1024 # Adjust based on your actual data shape
predicted_points, actual_points = train_and_predict('Issue_ID_4_2024_06_13_07_47_15.bag', num_points)
print("Predicted LiDAR Points:", predicted_points)
print("Actual LiDAR Points:", actual_points)
