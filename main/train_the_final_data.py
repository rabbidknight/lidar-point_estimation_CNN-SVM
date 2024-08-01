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
from keras.callbacks import ReduceLROnPlateau
from keras.models import load_model

def create_slfn_model(input_dim):
    model = Sequential([
        #Flatten(), # Flatten the input if it is not already 1D
        Dense(7, input_dim=input_dim, activation='linear')  # Output layer with 7 units (no activation for regression)
    ])

    model.compile(optimizer=Adam(learning_rate=0.00015), loss='mean_squared_error')

    return model

def add_channel_dimension(batch):
    # Convert list to a numpy array and add a new axis at the end to act as the channel dimension
    batch_array = np.array([np.array(x) for x in batch])
    return np.expand_dims(batch_array, axis=-1)

def predict(current_folder, x, y, model_path):
    # Load the pre-trained model
    model = load_model(model_path)

    predictions = model.predict(x)

    # Optionally visualize predictions
    plot2d_lidar_positions(y, predictions, current_folder)  # Call plotting function

def train_and_predict(bag_file, current_folder, use_pretrained):
    seq_offset = 25  # Offset to synchronize point clouds and poses
    point_clouds, poses_flattened, num_of_points = extract_and_transform_data(bag_file, seq_offset)

    # Prepare data batches correctly
    X, Y = [], []
    index = 0
    for n_points in num_of_points:
        X.append(point_clouds[index:index + n_points])
        Y.append(poses_flattened[index:index + n_points])
        index += n_points

    # Split into training and test sets
    split_index = int(len(X) * 0.8)  # 80% training, 20% testing
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = Y[:split_index], Y[split_index:]
    
    if use_pretrained:
            model = load_model(os.path.join(current_folder, 'slfn_model.h5'))
            # Predict on each test batch
            for x, y in zip(X_test, y_test):
                predict(current_folder, x, y, model)
    else:
        model = create_slfn_model(X_train[0].size)  # Create the model with the correct input shape
        print("Training the model...")
        for x, y in zip(X_train, y_train):
            # Flatten each batch's input and output as needed by the SLFN
            x_flattened = np.concatenate(x).reshape(1, -1)  # Flattening and reshaping to ensure 2D input
            y_flattened = np.concatenate(y).reshape(1, -1)
            model.fit(x_flattened, y_flattened, batch_size=1, epochs=3, verbose=1)

        model.save(os.path.join(current_folder, 'slfn_model.h5'))
        print("Model saved to:", os.path.join(current_folder, 'slfn_model.h5'))

        # Predict on each test batch
        for x, y in zip(X_test, y_test):
            x_flattened = np.concatenate(x).reshape(1, -1)  # Flattening input for prediction
            y_flattened = np.concatenate(y).reshape(1, -1)
            predict(current_folder, x_flattened, y_flattened, os.path.join(current_folder, 'slfn_model.h5'))
