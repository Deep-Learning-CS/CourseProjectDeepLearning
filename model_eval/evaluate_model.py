# extract a zip file
import zipfile
import os
import pathlib
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
import os
from pesq import pesq  # PESQ calculation
from scipy.signal import correlate


with zipfile.ZipFile('noise_test_dataset_mini.zip', 'r') as zip_ref:
    zip_ref.extractall('noise_test_dataset_mini')



def find_file_pairs(base_directory):
    """
    Recursively find pairs of clean and noisy files in a directory and its subdirectories.

    Args:
        base_directory (str): Root directory to start searching from

    Returns:
        list: A list of tuples containing (clean_file_path, noisy_file_path)
    """

    matched_files = []

    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(base_directory):
        print(root, dirs, files)
        # Create a dictionary to match clean and noisy files

        # Collect files, looking for clean and noisy versions
        for testfile in files:

            clean_audio_path = os.path.join(root, testfile)

            print(clean_audio_path)

            filepath = os.path.join(root, testfile)

            # Check if filename starts with 'clean_' or 'noisy_'
            if testfile.startswith('clean_'):
              noisy_file = testfile.replace('clean_', 'noisy_')
              noisy_audio_path = os.path.join(root, noisy_file)

              matched_files.append((noisy_audio_path, filepath))

    return matched_files

# if __name__ == "__main__":
#     # Replace 'your_directory_path' with the path to your root directory
#     base_dir = '/content/noise_test_dataset_mini'

#     # Find file pairs
#     pairs = find_file_pairs(base_dir)
#     print(pairs)
#     # Print out the pairs
#     print(f"Found {len(pairs)} file pairs:")


class Evaluation:
    def __init__(self, model_weights_path: str, test_data_path: str):
        self.model = self.load_model(model_weights_path)
        self.noisy_audio_path = test_data_path

    def load_model(self, model_weights_path: str) -> tf.keras.Model:
        model = tf.keras.models.load_model(model_weights_path)
        return model

    def find_file_pairs(self, base_directory):
        matched_files = []
        for root, dirs, files in os.walk(base_directory):
            for testfile in files:
                clean_audio_path = os.path.join(root, testfile)
                print(clean_audio_path)
                filepath = os.path.join(root, testfile)
                if testfile.startswith('clean_'):
                  noisy_file = testfile.replace('clean_', 'noisy_')
                  noisy_audio_path = os.path.join(root, noisy_file)
                  matched_files.append((noisy_audio_path, filepath))
        return matched_files

    def load_test_data(self, file_path: str) -> np.ndarray:
        # Load audio file and return its waveform
        audio, sr = librosa.load(file_path, sr=16000)
        return audio

    def evaluate_model(self) -> dict:

      psnr_value = 0
      ssim_value = 0
      mse_value = 0
      snr_value = 0
      pesq_value = 0
      mcd_value = 0

      pairs = self.find_file_pairs(self.noisy_audio_path)

      length = len(pairs)

      for pair in pairs:
        self.noisy_audio_path = pair[0]
        self.clean_audio_path = pair[1]

        clean_audio = self.load_test_data(self.clean_audio_path)

        original_length = len(clean_audio) # store the original length of the audio


        # Make predictions
        predicted_audio = denoise_audio(self.noisy_audio_path, downloaded_model)

        predicted_audio = predicted_audio[:original_length] # trim to match original length

        # Compute metrics
        psnr_value += self.compute_PSNR(clean_audio, predicted_audio)
        ssim_value += self.compute_SSIM(clean_audio, predicted_audio)
        mse_value += self.compute_MSE(clean_audio, predicted_audio)
        snr_value += self.compute_SNR(clean_audio, predicted_audio)
        pesq_value += self.compute_PESQ(clean_audio, predicted_audio)
        mcd_value += self.compute_MCD(clean_audio, predicted_audio)

      metrics = {
          'PSNR': psnr_value / length,
          'SSIM': ssim_value / length,
          'MSE': mse_value / length,
          'SNR': snr_value / length,
          'PESQ': pesq_value / length,
          'MCD': mcd_value / length
      }
      return metrics

    def compute_PSNR(self, clean_frames: np.ndarray, noisy_frames: np.ndarray) -> float:
        mse = np.mean((clean_frames - noisy_frames) ** 2)
        if mse == 0:
            return float('inf')  # Infinite PSNR if there is no noise
        max_pixel_value = 1.0  # Normalized range for audio signals
        psnr_value = 20 * np.log10(max_pixel_value / np.sqrt(mse))
        return psnr_value

    def compute_SSIM(self, clean_frames: np.ndarray, noisy_frames: np.ndarray) -> float:
        # SSIM expects 2D images; reshape if necessary
        return ssim(clean_frames, noisy_frames, data_range=clean_frames.max() - clean_frames.min())

    def compute_MSE(self, clean_frames: np.ndarray, noisy_frames: np.ndarray) -> float:
        return np.mean((clean_frames - noisy_frames) ** 2)

    def compute_SNR(self, clean_audio: np.ndarray, predicted_audio: np.ndarray) -> float:
        # Calculate the Signal-to-Noise Ratio (SNR)
        noise = clean_audio - predicted_audio
        signal_power = np.sum(clean_audio ** 2)
        noise_power = np.sum(noise ** 2)
        if noise_power == 0:
            return float('inf')  # Infinite SNR if there's no noise
        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    def compute_PESQ(self, clean_audio: np.ndarray, predicted_audio: np.ndarray) -> float:
        # Use PESQ (requires sampling rate and proper file formats)
        # PESQ expects a clean reference and a degraded version
        return pesq(16000, clean_audio, predicted_audio, 'wb')

    def compute_MCD(self, clean_audio: np.ndarray, predicted_audio: np.ndarray) -> float:
        # Compute Mel Cepstral Distortion (MCD)
        clean_mfcc = librosa.feature.mfcc(y=clean_audio, sr=16000)
        predicted_mfcc = librosa.feature.mfcc(y=predicted_audio, sr=16000)
        mcd = np.mean(np.sqrt(np.sum((clean_mfcc - predicted_mfcc) ** 2, axis=0)))
        return mcd

    def generate_report(self, metrics: dict, file_path: str):
        with open(file_path, 'w') as f:
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")



###########################
# Main Evaluation

if __name__ == "__main__":
    evaluator = Evaluation(
        model_weights_path="modeldev_luke.h5", # enter your model name here
        test_data_path = "noise_test_dataset_mini",
    )

    metrics = evaluator.evaluate_model()
    evaluator.generate_report(metrics, "performance_report.txt")
    print("Evaluation complete. Report generated as 'performance_report.txt'.")