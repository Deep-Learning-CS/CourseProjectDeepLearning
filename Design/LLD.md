# Low-Level Design (LLD) for Noise Reduction Using Deep Learning

## Overview

This document outlines the low-level design of the **Noise Reduction Using Deep Learning** project. The design is focused on leveraging a pretrained model, freezing its layers, adding custom layers, and proceeding with training and deployment. The project is divided into five main components.

## Components

### 1. Preprocessing

**Objective:** Prepare noise data for training.

- **Input:** `noisy_audio.wav`, `clean_audio.wav`
- **Output:** `noisy_spectrogram.npy`, `clean_spectrogram.npy`

**Functions:**
- `load_audio(file_path: str) -> np.ndarray`
    - Loads the noisy and clean audio files.
- `extract_spectrogram(audio_data: np.ndarray) -> np.ndarray`
    - Converts the audio data into spectrograms for training.
- `augment_data(spectrogram: np.ndarray) -> np.ndarray`
    - Augments the spectrogram with techniques like noise addition and time-stretching.
- `save_spectrogram(spectrogram: np.ndarray, file_path: str)`
    - Saves the spectrogram as `.npy` files for later use.

---

### 2. Model Development

**Objective:** Build the model by utilizing a pretrained model, freezing layers, and adding custom layers.

- **Input:** `noisy_spectrogram.npy`, `clean_spectrogram.npy`
- **Output:** `extended_noise_reduction_model.h5`

**Functions:**
- `load_pretrained_model(model_name: str) -> keras.Model`
    - Loads a pretrained model (e.g., U-Net, VGG).
- `freeze_layers(model: keras.Model) -> keras.Model`
    - Freezes layers of the pretrained model to retain learned features.
- `add_custom_layers(model: keras.Model) -> keras.Model`
    - Adds custom layers (e.g., LSTM, Dense layers) to improve noise reduction.
- `compile_model(model: keras.Model, optimizer: str, loss_function: str)`
    - Compiles the updated model using appropriate loss functions and optimizers.
- `save_model(model: keras.Model, file_path: str)`
    - Saves the extended model to the specified file path.

---

### 3. Training

**Objective:** Train the extended model on the prepared data.

- **Input:** `extended_noise_reduction_model.h5`, `noisy_spectrogram.npy`, `clean_spectrogram.npy`
- **Output:** `best_model_weights.h5`, `training_logs.csv`

**Functions:**
- `train_model(model: keras.Model, training_data: np.ndarray, labels: np.ndarray, epochs: int, batch_size: int)`
    - Trains the model with noisy and clean spectrograms.
- `save_weights(model: keras.Model, file_path: str)`
    - Saves the best-performing model weights during training.
- `optimize_training_schedule(model: keras.Model) -> keras.Model`
    - Optimizes training efficiency using learning rate schedules.

---

### 4. Evaluation

**Objective:** Measure the performance of the trained model.

- **Input:** `best_model_weights.h5`, `noisy_audio_test.wav`, `clean_audio_test.wav`
- **Output:** `performance_report.txt (SNR, PESQ, MCD)`, `evaluate_extended_model.py`

**Functions:**
- `load_test_audio(file_path: str) -> np.ndarray`
    - Loads test audio data for evaluation.
- `compute_spectrogram(audio_data: np.ndarray) -> np.ndarray`
    - Converts audio data into spectrograms.
- `evaluate_model(model: keras.Model, test_spectrogram: np.ndarray) -> dict`
    - Evaluates the model’s performance on test data and returns metrics.
- `compute_SNR(clean_audio: np.ndarray, noisy_audio: np.ndarray) -> float`
    - Computes the Signal-to-Noise Ratio (SNR) between clean and noisy audio.
- `compute_PESQ(clean_audio: np.ndarray, noisy_audio: np.ndarray) -> float`
    - Computes the Perceptual Evaluation of Speech Quality (PESQ) between clean and noisy audio.
- `compute_MCD(clean_spectrogram: np.ndarray, noisy_spectrogram: np.ndarray) -> float`
    - Computes the Mel-Cepstral Distortion (MCD) between the spectrograms.
- `generate_report(metrics: dict, file_path: str)`
    - Generates a performance report and saves it.

---

### 5. Deployment

**Objective:** Deploy the trained model and create APIs for real-time audio denoising.

- **Input:** `best_model_weights.h5`, `performance_report.txt`
- **Output:** API endpoint, deployed model on AWS/GCP

**Functions:**
- `create_API(model: keras.Model)`
    - Creates a REST API to expose the noise reduction functionality.
- `deploy_model_to_cloud(model: keras.Model, cloud_platform: str)`
    - Deploys the model to a cloud platform for scalability.
- `process_real_time_audio(audio_stream: str) -> np.ndarray`
    - Accepts live audio streams, processes them, and returns denoised audio.

---

## Relationships

- **Preprocessing** provides training spectrograms to **Model Development**.
- **Model Development** provides the extended model to **Training**.
- **Training** sends the trained model to **Evaluation**.
- **Evaluation** provides evaluation results to **Deployment**.
- **Training** shares the final model weights with **Deployment**.

---

## Conclusion

This LLD outlines the detailed functionality and relationships of each component in the noise reduction project using deep learning techniques. The design is intended to facilitate a smooth workflow for developers involved in the project.
