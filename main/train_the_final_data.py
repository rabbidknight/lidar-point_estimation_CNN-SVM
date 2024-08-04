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



class TrainingLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logger.info(f'Epoch {epoch + 1}: Loss: {logs["loss"]}')

def create_slfn_model():
    model = Sequential([
        #Flatten(), # Flatten the input if it is not already 1D
        Dense(3, activation='linear')  # Output layer with 7 units (no activation for regression)
    ])

    model.compile(optimizer=Adam(learning_rate=0.0015), loss='mean_squared_error')

    return model

def add_channel_dimension(batch):
    # Convert list to a numpy array and add a new axis at the end to act as the channel dimension
    batch_array = np.array([np.array(x) for x in batch])
    return np.expand_dims(batch_array, axis=-1)

def predict(current_folder, x, y, model_path):
    # Load the pre-trained model
    model = (model_path)

    predictions = model.predict(x)


    return predictions
    # Optionally visualize predictions
    #plot2d_lidar_positions(y, predictions, current_folder)  # Call plotting function

def train_and_predict(bag_file, current_folder, use_pretrained):
    seq_offset = 25  # Offset to synchronize point clouds and poses
    point_clouds, poses_flattened, num_of_points = extract_and_transform_data(bag_file, seq_offset)

    print("Point cloud len", len(poses_flattened))
    # Split the point clouds and poses into batches based on num_of_points
    X = []
    Y = poses_flattened
    index = 0
    for n in num_of_points:
        if index + n <= len(point_clouds):
            X.append(point_clouds[index:index + n])
            #Y.append(poses_flattened[index:index + n])
        index += n
    print("Number of batches:", len(X))
    print("Number of sequences in each batch:", len(X[0]))
    print("Number of Y sequences in each batch:", len(Y))


    # Finding the minimum length across all batches to truncate
    min_length = min(min(len(x) for x in batch) for batch in X)  # Minimum length of any sequence in all batches

    # Truncate each sequence in every batch to the minimum length
    X = [[seq[:min_length] for seq in batch] for batch in X]

    print("Number of batches:", len(X))
    print("Number of sequences in each batch:", len(X[0]))
    print("Number of Y sequences in each batch:", len(Y))


    # Split into training and test sets
    split_index_x = int(len(X) * 0.1)  # 10% training, 90% testing
    split_index_y = int(len(Y) * 0.1)  # 10% training, 90% testing
    X_train, X_test = X[:split_index_x], X[split_index_x:]
    y_train, y_test = Y[:split_index_y], Y[split_index_y:]
  


    
    if use_pretrained:
            model = load_model(os.path.join(current_folder, 'slfn_model.h5'))
            # Predict on each test batch
            predicted = []
            for x, y in zip(X_test, y_test):
                predicted.append(predict(current_folder, np.array(x), np.array(y), model))
            plot2d_lidar_positions(y_test, predicted, current_folder)
    else:
        print("Creating the model...")
        model = create_slfn_model()  # Create the model
        # Setup ReduceLROnPlateau callback
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=2, min_lr=0.00001, verbose=1)
        print("Done. Training the model now...")
        index = 0
        print("Number of training batches:", (y_train)[0])
        print("Number of sequences in each training batchleeeeeeeeeeeeeeeeeeeennnnnnn:", len(y_train[1]))
        print("Number of Y sequences in each training batch:", (y_train[1][0]))
        for x, y in zip(X_train, y_train):
            # Flatten each batch's input and output as needed by the SLFN
            for xx in x:
                model.fit((xx), np.concatenate(y), batch_size=1, epochs=1, validation_split=0.2, callbacks=[TrainingLogger(), reduce_lr])            
            print("Batch", index, "trained")
            index += 1

        model.save(os.path.join(current_folder, 'slfn_model.h5'))
        print("Done. Trained model saved to:", os.path.join(current_folder, 'slfn_model'))
        print("Predicting on test data...")

        
        
        # Predict on each test batch
        predicted = []
        for x, y in zip(X_test, y_test):
            for xx in x:
                predicted.append(predict(current_folder, xx, np.concatenate(y), model))

        plot2d_lidar_positions(y_test, predicted, current_folder)
