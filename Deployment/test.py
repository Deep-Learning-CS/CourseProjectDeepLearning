import os
import h5py
import tensorflow as tf
from tensorflow.keras.models import load_model

# Custom loss function, if used in the model
def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

# Function to check and load a model
def check_and_load_model(model_path):
    if os.path.exists(model_path):
        print(f"'{model_path}' exists!")
        try:
            # Attempt to load the model
            print(f"Attempting to load model from: {model_path}")
            model = load_model(
                model_path,
                custom_objects={"ssim_loss": ssim_loss},
                compile=False
            )
            print(f"Model '{model_path}' loaded successfully!")
            model.summary()
            return model
        except OSError as e:
            print(f"Failed to load model '{model_path}'. OSError: {e}")
        except ValueError as e:
            print(f"Failed to load model '{model_path}'. ValueError: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while loading '{model_path}': {e}")
    else:
        print(f"'{model_path}' does not exist.")
    return None

# Function to check if an HDF5 file is valid
def check_hdf5_file(file_path):
    if os.path.exists(file_path):
        print(f"Checking HDF5 file: {file_path}")
        try:
            with h5py.File(file_path, "r") as f:
                print(f"HDF5 file keys in '{file_path}':", list(f.keys()))
        except OSError as e:
            print(f"Error opening HDF5 file '{file_path}': {e}")
    else:
        print(f"'{file_path}' does not exist.")

# Check and load 'modeldev_luke.h5'
check_and_load_model("modeldev_luke.h5")
check_hdf5_file("modeldev_luke.h5")

# Check and load 'modeldev_luke.keras'
check_and_load_model("modeldev_luke.keras")
check_hdf5_file("modeldev_luke.keras")
