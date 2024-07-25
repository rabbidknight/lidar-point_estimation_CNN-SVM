import os
import datetime
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn import logger
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.callbacks import Callback
from get_data import extract_and_transform_data
from plot_the_final_data import plot3d_point_clouds, plot2d_lidar_positions


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

def train_and_predict(bag_file, current_folder):
    seq_offset = 25  # Offset to synchronize point clouds and poses
    point_clouds, poses = extract_and_transform_data(bag_file, seq_offset)
    plot3d_point_clouds(point_clouds, current_folder)
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = manual_split(point_clouds, poses)

    # Ensure the data is in the correct numpy array format
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array([np.array(y) for y in y_train])
    y_test = np.array([np.array(y) for y in y_test])

    # Create and compile the model
    model = create_slfn_model()
    model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=((X_test), y_test))

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

    plot2d_lidar_positions(y_test, predictions, current_folder)  # Call plotting function
        # Evaluate the model at the end of each epoch
        #val_loss = model.evaluate(X_test, y_test, verbose=1)
        #print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")