# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 11:55:53 2024

@author: Ayush
"""

import os
import random
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display

# Preprocessing class for handling audio files
class Preprocessing:
    def load_audio(self, file_path: str) -> np.ndarray:
        """
        Load an audio file and return its waveform as a numpy array.

        Parameters:
        file_path (str): Path to the audio file.

        Returns:
        np.ndarray: Audio waveform, and sample rate.
        """
        audio, sr = librosa.load(file_path, sr=None)  # Load audio with original sampling rate
        return audio, sr

    def extract_spectrogram(self, audio: np.ndarray, sr: int = 22050) -> np.ndarray:
        """
        Convert the audio waveform to a Mel spectrogram.

        Parameters:
        audio (np.ndarray): The audio waveform.
        sr (int): The sampling rate of the audio.

        Returns:
        np.ndarray: Spectrogram (magnitude in decibels).
        """
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        return spectrogram_db

    def save_spectrogram(self, spectrogram: np.ndarray, file_path: str):
        """
        Save the spectrogram as a .npy file.

        Parameters:
        spectrogram (np.ndarray): The spectrogram to save.
        file_path (str): Path where to save the spectrogram file.
        """
        np.save(file_path, spectrogram)

    def match_length(self, audio: np.ndarray, target_length: int) -> np.ndarray:
        """
        Match the length of the audio to a target length by trimming or padding.

        Parameters:
        audio (np.ndarray): Input audio waveform.
        target_length (int): Target length for the audio.

        Returns:
        np.ndarray: Length-matched audio waveform.
        """
        if len(audio) < target_length:
            # Pad with zeros if shorter
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            # Trim if longer
            audio = audio[:target_length]
        return audio

# Function to get all noise files from a directory
def get_noise_files(noise_dir: str) -> list:
    """
    Retrieve all noise file paths from the specified directory.

    Parameters:
    noise_dir (str): Path to the directory containing noise files.

    Returns:
    list: List of noise file paths.
    """
    noise_files = []
    for root, _, files in os.walk(noise_dir):
        for file in files:
            if file.endswith(('.wav', '.flac')):  # Process only supported audio files
                noise_files.append(os.path.join(root, file))
    return noise_files

# Function to augment audio with multiple noise samples and save clean spectrograms
def process_audio_with_multiple_noises(input_dir: str, output_dir: str, noise_dir: str, noise_factor: float = 0.02, n_noises: int = 5):
    """
    Process all clean audio files in the input directory, augment them with multiple noise samples from the MS-SNSD folder,
    and save the processed audio and spectrograms, including the clean spectrogram.

    Parameters:
    input_dir (str): Path to the directory containing clean audio files.
    output_dir (str): Path to the directory for saving augmented audio and spectrograms.
    noise_dir (str): Path to the MS-SNSD noise folder.
    noise_factor (float): The factor by which noise is added to clean audio.
    n_noises (int): Number of random noise samples to apply to each clean audio file.
    """
    preprocess = Preprocessing()
    noise_files = get_noise_files(noise_dir)

    if not noise_files:
        print("No noise files found in the specified MS-SNSD directory!")
        return

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.flac'):  # Only process .flac files
                clean_audio_path = os.path.join(root, file)
                clean_audio, sr = preprocess.load_audio(clean_audio_path)

                # Prepare output paths
                output_audio_path = os.path.join(output_dir, os.path.relpath(root, input_dir))
                if not os.path.exists(output_audio_path):
                    os.makedirs(output_audio_path)

                # Save clean spectrogram
                clean_spectrogram = preprocess.extract_spectrogram(clean_audio, sr)
                preprocess.save_spectrogram(clean_spectrogram, os.path.join(output_audio_path, f'clean_spectrogram_{file.replace(".flac", ".npy")}'))

                # Create multiple noisy versions
                for i in range(n_noises):
                    noise_file = random.choice(noise_files)
                    noise_audio, _ = librosa.load(noise_file, sr=None)
                    noise_audio = preprocess.match_length(noise_audio, len(clean_audio))
                    noisy_audio = clean_audio + noise_factor * noise_audio
                    noisy_audio = np.clip(noisy_audio, -1.0, 1.0)

                    # Save noisy audio
                    noisy_audio_file_path = os.path.join(output_audio_path, f'noisy_{i+1}_' + file)
                    sf.write(noisy_audio_file_path, noisy_audio, sr, format='FLAC')

                    # Extract and save noisy spectrogram
                    noisy_spectrogram = preprocess.extract_spectrogram(noisy_audio, sr)
                    preprocess.save_spectrogram(noisy_spectrogram, os.path.join(output_audio_path, f'noisy_spectrogram_{i+1}_' + file.replace('.flac', '.npy')))

                print(f"Processed {n_noises} noisy variants and clean spectrogram for: {clean_audio_path}")

# Call the function with directories
process_audio_with_multiple_noises(
    input_dir="dev-clean",  # Directory containing clean audio
    output_dir="processed-audio",  # Directory to save augmented files
    noise_dir="MS-SNSD/noise_train",  # Directory containing noise files
    noise_factor=0.2,  # Adjust noise factor as needed
    n_noises=1  # Number of noisy variants to generate per audio file
)
