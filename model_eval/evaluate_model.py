import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
import os
from pesq import pesq  # PESQ calculation
from pypesq import pesq as pypesq  # PESQ calculation with pypesq
from scipy.signal import correlate

class Evaluation:
    def __init__(self, model_weights_path: str, noisy_audio_path: str, clean_audio_path: str):
        self.model = self.load_model(model_weights_path)
        self.noisy_audio_path = noisy_audio_path
        self.clean_audio_path = clean_audio_path

    def load_model(self, model_weights_path: str) -> tf.keras.Model:
        model = tf.keras.models.load_model(model_weights_path)
        return model

    def load_test_data(self, file_path: str) -> np.ndarray:
        # Load audio file and return its waveform
        audio, sr = librosa.load(file_path, sr=16000)
        return audio

    def evaluate_model(self) -> dict:
        noisy_audio = self.load_test_data(self.noisy_audio_path)
        clean_audio = self.load_test_data(self.clean_audio_path)

        # Make predictions
        noisy_spectrogram = self.extract_spectrogram(noisy_audio)
        noisy_spectrogram = np.expand_dims(noisy_spectrogram, axis=0)  # Add batch dimension
        predicted_spectrogram = self.model.predict(noisy_spectrogram)

        # Convert back to audio
        predicted_audio = self.inverse_spectrogram(predicted_spectrogram[0])

        # Compute metrics
        psnr_value = self.compute_PSNR(clean_audio, predicted_audio)
        ssim_value = self.compute_SSIM(clean_audio, predicted_audio)
        mse_value = self.compute_MSE(clean_audio, predicted_audio)
        snr_value = self.compute_SNR(clean_audio, predicted_audio)
        pesq_value = self.compute_PESQ(clean_audio, predicted_audio)
        mcd_value = self.compute_MCD(clean_audio, predicted_audio)

        metrics = {
            'PSNR': psnr_value,
            'SSIM': ssim_value,
            'MSE': mse_value,
            'SNR': snr_value,
            'PESQ': pesq_value,
            'MCD': mcd_value
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
        clean_frames = clean_frames.reshape(1, -1)
        noisy_frames = noisy_frames.reshape(1, -1)
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
        clean_mfcc = librosa.feature.mfcc(clean_audio, sr=16000)
        predicted_mfcc = librosa.feature.mfcc(predicted_audio, sr=16000)
        mcd = np.mean(np.sqrt(np.sum((clean_mfcc - predicted_mfcc) ** 2, axis=0)))
        return mcd

    def generate_report(self, metrics: dict, file_path: str):
        with open(file_path, 'w') as f:
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
    
    def extract_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        # Convert the audio waveform to a Mel spectrogram
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=16000, n_fft=2048, hop_length=512)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        return spectrogram_db

    def inverse_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        mel_spectrogram = librosa.db_to_power(spectrogram)
        linear_spectrogram = librosa.feature.inverse.mel_to_stft(mel_spectrogram, sr=16000, n_fft=2048)
        audio_signal = librosa.griffinlim(linear_spectrogram, n_iter=64, hop_length=512, n_fft=2048)
        return audio_signal

###########################
# Main Evaluation

if __name__ == "__main__":
    evaluator = Evaluation(
        model_weights_path="../model_training/trained_model.keras",
        noisy_audio_path="noisy_audio.flac",
        clean_audio_path="clean_audio.flac"
    )
    
    metrics = evaluator.evaluate_model()
    evaluator.generate_report(metrics, "performance_report.txt")
    print("Evaluation complete. Report generated as 'performance_report.txt'.")
