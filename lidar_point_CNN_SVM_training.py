import rosbag
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Function to convert ROS bag to CSV and extract both point cloud and lidar point data
def rosbag_to_csv(bag_file, csv_file, point_cloud_topic, lidar_topic):
    bag = rosbag.Bag(bag_file)
    point_cloud_data = []
    lidar_point_data = []

    for topic, msg, t in bag.read_messages(topics=[point_cloud_topic, lidar_topic]):
        if topic == point_cloud_topic:
            points = np.array([[point.x, point.y, point.z] for point in msg.points])
            point_cloud_data.append(points.flatten())
        elif topic == lidar_topic:
            # Assuming LiDAR points are structured similarly
            lidar_points = np.array([msg.x, msg.y, msg.z])
            lidar_point_data.append(lidar_points.flatten())

    point_cloud_df = pd.DataFrame(point_cloud_data)
    lidar_point_df = pd.DataFrame(lidar_point_data, columns=['lidar_x', 'lidar_y', 'lidar_z'])
    combined_df = pd.concat([point_cloud_df, lidar_point_df], axis=1)
    combined_df.to_csv(csv_file, index=False)
    bag.close()
    return combined_df

# Function to print sample data
def print_sample_data(data, num_samples=5):
    print("Sample Data:")
    print(data.head(num_samples))

# Define the CNN architecture
def build_cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', input_shape=input_shape),
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

# Training and Prediction Logic
def train_and_predict(bag_file, csv_file, point_cloud_topic, lidar_topic, input_shape):
    # Convert rosbag to CSV and load data
    data = rosbag_to_csv(bag_file, csv_file, point_cloud_topic, lidar_topic)
    print_sample_data(data)  # Print sample data for inspection

    features = data.iloc[:, :-3].values  # all columns except lidar points
    labels = data[['lidar_x', 'lidar_y', 'lidar_z']].values  # lidar points

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_cnn_model(input_shape)
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
input_shape = (img_size, img_size, 1)  # Adjust based on your actual data shape
predicted_points, actual_points = train_and_predict('input.bag', 'output.csv', 'point_cloud_topic', 'lidar_topic', input_shape)
print("Predicted LiDAR Points:", predicted_points)
print("Actual LiDAR Points:", actual_points)
