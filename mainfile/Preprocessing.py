#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa.display
import os
import tarfile


# 

# In[5]:


# Path to the dataset tar.gz file
dataset_path = '/content/dev-clean.tar.gz'
# Directory to save extracted files and spectrograms
extracted_dir = '/content/extracted'
spectrogram_dir = '/content/mel_spectrograms'

# Create directories if they don't exist
os.makedirs(extracted_dir, exist_ok=True)
os.makedirs(spectrogram_dir, exist_ok=True)

# Path to the dataset tar.gz file
dataset_path = '/content/dev-clean.tar.gz'
# Directory to extract the files
extracted_dir = '/content/extracted'

# Create the directory if it doesn't exist
os.makedirs(extracted_dir, exist_ok=True)

# Extract the dataset
try:
    with tarfile.open(dataset_path, 'r:gz') as tar:
        tar.extractall(path=extracted_dir)
        print(f"Extraction completed successfully to {extracted_dir}")
except tarfile.ReadError:
    print("Error: Unable to read the tar file. It might be corrupted.")
except EOFError:
    print("Error: The file seems to be incomplete or corrupted.")





# In[18]:


# Function to generate and save Mel spectrogram
def save_mel_spectrogram(audio_path, save_path):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=16000)  # Using 16kHz sample rate
    # Generate Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    # Augment the audio by adding noise
    noise = np.random.randn(len(y))
    augmented_audio = y + noise_factor * noise

    # Save Mel spectrogram as image
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Mel spectrogram at {save_path}")


# 

# In[ ]:


import os
import tarfile
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Path to the dataset tar.gz file
dataset_path = '/content/dev-clean.tar.gz'
# Directory to extract the files
extracted_dir = '/content/extracted'
# Directory to save spectrogram images
spectrogram_dir = '/content/mel_spectrograms'

# Create directories if they don't exist
os.makedirs(extracted_dir, exist_ok=True)
os.makedirs(spectrogram_dir, exist_ok=True)

# Step 1: Extract the dataset
try:
    with tarfile.open(dataset_path, 'r:gz') as tar:
        tar.extractall(path=extracted_dir)
        print(f"Extraction completed successfully to {extracted_dir}")
except tarfile.ReadError:
    print("Error: Unable to read the tar file. It might be corrupted.")
except EOFError:
    print("Error: The file seems to be incomplete or corrupted.")

# Step 2: Function to generate and save Mel spectrogram
def save_mel_spectrogram(audio_path, save_path, noise_factor=0.005):
    """
    Generate and save a Mel spectrogram as an image.
    Optionally augment the audio with noise.

    Parameters:
    audio_path (str): Path to the input audio file.
    save_path (str): Path to save the Mel spectrogram image.
    noise_factor (float): Factor controlling the amount of noise for augmentation.
    """
    try:
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=16000)  # Using 16kHz sample rate

        # Augment the audio by adding noise
        noise = np.random.randn(len(y))
        augmented_audio = y + noise_factor * noise

        # Generate Mel spectrogram
        S = librosa.feature.melspectrogram(y=augmented_audio, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Save Mel spectrogram as image
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel spectrogram')
        plt.tight_layout()

        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"Saved Mel spectrogram at {save_path}")
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")

# Step 3: Process each .flac file in the extracted dataset
def process_dataset(dataset_folder, output_folder, noise_factor=0.005):
    """
    Process each .flac file in a dataset folder, generate Mel spectrograms, and save them as images.

    Parameters:
    dataset_folder (str): Path to the folder containing .flac files.
    output_folder (str): Path to save the spectrogram images.
    noise_factor (float): Factor controlling the amount of noise for augmentation.
    """
    for root, _, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith(".flac"):
                # Input and output paths
                audio_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, dataset_folder)
                spectrogram_save_path = os.path.join(output_folder, relative_path, f"{os.path.splitext(file)[0]}.png")

                # Process and save the spectrogram
                save_mel_spectrogram(audio_path, spectrogram_save_path, noise_factor)

# Example usage
process_dataset(extracted_dir, spectrogram_dir)

