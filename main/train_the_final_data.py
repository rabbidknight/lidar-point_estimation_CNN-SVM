import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from get_data import extract_and_transform_data
from plot_the_final_data import plot2d_lidar_positions

def create_slfn_model():
    print("Defining model...")
    model = Sequential([
        Dense(4, activation='relu'),
        BatchNormalization(),
        Dense(8, activation='relu'),
        BatchNormalization(),
        Dense(4, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        Dense(3, activation='linear')  # Adjust to match your output shape
    ])
    model.compile(optimizer=Adam(learning_rate=0.0015), loss='mean_squared_error')
    print("Model compiled.")
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    print("Training the model...")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    if X_train.size == 0 or y_train.size == 0:
        print("Training data is empty.")
        return

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.00001, verbose=1)
    model.fit(X_train, y_train, batch_size=1, epochs=6, validation_data=(X_test, y_test), verbose=1, callbacks=[reduce_lr])
    print("Training complete.")


def predict(model, current_folder, x, y):
    print("Predicting with model...")
    predictions = model.predict(x)
    plot2d_lidar_positions(y, predictions, current_folder)

def manual_split(data, labels, test_ratio=0.35):
    split_index = int(len(data) * (1 - test_ratio))
    X_train, X_test = data[:split_index], data[split_index:]
    y_train, y_test = labels[:split_index], labels[split_index:]
    return X_train, X_test, y_train, y_test
