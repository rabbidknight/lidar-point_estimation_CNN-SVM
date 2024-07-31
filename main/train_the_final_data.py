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

def create_slfn_model():
    model = Sequential([
        #Flatten(), # Flatten the input if it is not already 1D
        Dense(7, activation='linear')  # Output layer with 7 units (no activation for regression)
    ])

    model.compile(optimizer=Adam(learning_rate=0.0015), loss='mean_squared_error')

    #LOGGING
    #for layer in model.layers:
    #   logger.info(f'Layer {layer.name} - Type: {layer.__class__.__name__}, Output Shape: {layer.output_shape}, Activation: {getattr(layer, "activation", None).__name__ if hasattr(layer, "activation") else "N/A"}')
    #logger.info("optimizer:", model.optimizer, "loss:", model.loss, "metrics:", model.metrics_names, "learning rate:", model.optimizer.lr)

    return model

def add_channel_dimension(batch):
    # Convert list to a numpy array and add a new axis at the end to act as the channel dimension
    batch_array = np.array([np.array(x) for x in batch])
    return np.expand_dims(batch_array, axis=-1)

def manual_split(data, labels, test_ratio=0.30):
    print("Length of data:", len(data))
    print("Length of labels:", len(labels))
    split_data = int(len(data)* (1 - test_ratio))
    split_labels = int(len(labels) * (1 - test_ratio))
    X_train, X_test = data[:split_data], data[split_data:]
    y_train, y_test = labels[:split_labels], labels[split_labels:]
    return X_train, X_test, y_train, y_test

def predict(current_folder, x, y, model_path):
    # Load the pre-trained model
    model = load_model(model_path)

    # Ensure the input data is correctly shaped
    x = np.array(x).reshape(-1, 3, 1)  # Adjust based on the expected model input
    predictions = model.predict(x)

    # Optionally visualize predictions
    plot2d_lidar_positions(y, predictions, current_folder)  # Call plotting function

def train_and_predict(bag_file, current_folder, use_pretrained):
    seq_offset = 25  # Offset to synchronize point clouds and poses
    point_clouds, poses = extract_and_transform_data(bag_file, seq_offset)

    #plot3d_point_clouds(point_clouds, poses, current_folder)
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = manual_split(point_clouds, poses)

    # Ensure the data is in the correct numpy array format
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array([np.array(y) for y in y_train])
    y_test = np.array([np.array(y) for y in y_test])

    print("Shapes:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    if use_pretrained:
        model_path = os.path.join(current_folder, 'slfn_model.h5')
        predict(current_folder, X_test, y_test, model_path)
    else:
        # Create, compile and train the model
        model = create_slfn_model()
        # Define the ReduceLROnPlateau callback
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=0, min_lr=0.00001, verbose=1)
        
        # Train the model with the callback
        model.fit(X_train, y_train, batch_size=1, epochs=2, validation_data=(X_test, y_test), verbose=1, callbacks=[reduce_lr])

        model.save(os.path.join(current_folder, 'slfn_model.h5'))
        print("Model saved to:", os.path.join(current_folder, 'slfn_model.h5'))

        # After training, predict on the test set
        predict(current_folder, X_test, y_test, os.path.join(current_folder, 'slfn_model.h5'))

