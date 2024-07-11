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
rnn_model = load_model('model_10-07-24.keras')
svm_model = joblib.load('model_10-07-24.joblib')  # Make sure you've saved the SVM model using joblib

# Load test data
X_test, y_test = extract_data_from_bag('Issue_ID_4_2024_06_13_07_47_15.bag')

 
 # Ensure y_test is a NumPy array and properly reshaped
if isinstance(y_test, pd.DataFrame):
     y_test = y_test.values  # Convert to NumPy array as it's a DataFrame

# Use the CNN model to extract features
rnn_features = rnn_model.predict(X_test)
print("Unique features post RNN:", np.unique(rnn_features, axis=0).shape)  # Debugging

# Use the pre-trained SVM to make predictions
predicted_points = svm_model.predict(rnn_features)
predicted_points = predicted_points  # Assuming each prediction consists of 7 elements
actual_points = y_test
# Diagnostic print to check variability in predictions
print("Unique predicted points:", np.unique(predicted_points, axis=0).shape)
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

plt.figure(figsize=(18, 6))
plt.subplot(1, 4, 1)
plt.scatter(x_actual, x_pred, c='blue')
plt.title('X Coordinates')
plt.xlabel('Actual')
plt.ylabel('Predicted')

plt.subplot(1, 4, 2)
plt.scatter(y_actual, y_pred, c='red')
plt.title('Y Coordinates')
plt.xlabel('Actual')
plt.ylabel('Predicted')

plt.subplot(1, 4, 3)
plt.scatter(z_actual, z_pred, c='green')
plt.title('Z Coordinates')
plt.xlabel('Actual')
plt.ylabel('Predicted')

# Plot X-Y Trajectories
plt.subplot(1, 4, 4)
plt.scatter(x_actual, y_actual, c='blue', label='Actual Trajectory', alpha=0.6)
plt.scatter(x_pred, y_pred, c='red', label='Predicted Trajectory', alpha=0.6)
plt.title('X-Y Trajectories')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()


plt.tight_layout()
plt.show()

# Calculate mean percentage errors for each element
mean_percentage_errors = calculate_mean_percentage_error(actual_points, predicted_points)
print("Mean Percentage Errors for each element:", mean_percentage_errors)
