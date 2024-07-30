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
from train_the_final_data import train_and_predict


class TrainingLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logger.info(f'Epoch {epoch + 1}: Loss: {logs["loss"]}, Val Loss: {logs["val_loss"]}')

def logger_prep():
    # Current timestamp to create a unique folder
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    current_folder = f"test_logs/slfn_test/without-scipy/slfn_test__{current_time}"
    os.makedirs(current_folder, exist_ok=True)  # Create the directory if it doesn't exist
    #current_folder = os.getcwd()

    # Setup logging
    logger = logging.getLogger('LidarTestLogger')
    logger.setLevel(logging.INFO)

    # Log file
    log_filename = os.path.join(current_folder, 'test_log.txt')
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return current_folder

current_folder = logger_prep()
train_and_predict('Issue_ID_4_2024_06_13_07_47_15.bag', current_folder)
