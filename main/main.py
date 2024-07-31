import os
import glob
import numpy as np
from train_the_final_data import create_slfn_model, train_model, predict, manual_split, extract_and_transform_data
from plot_the_final_data import plot3d_point_clouds
def process_directory(directory, model):
    bag_files = glob.glob(os.path.join(directory, '*.bag'))
    for bag_file in bag_files:
        print(f"Processing file: {bag_file}")
        point_clouds, poses = extract_and_transform_data(bag_file)
        plot3d_point_clouds(point_clouds, poses, directory)
        X_train, X_test, y_train, y_test = manual_split(point_clouds, poses)
        # Ensure the data is in the correct numpy array format
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        y_train = np.array([np.array(y) for y in y_train])
        y_test = np.array([np.array(y) for y in y_test])
        train_model(model, X_train, y_train, X_test, y_test)
        predict(model, directory, X_test, y_test)

if __name__ == "__main__":
    current_directory = os.getcwd()  # or any specific directory containing .bag files
    model = create_slfn_model()  # Create the model only once
    process_directory(current_directory, model)
