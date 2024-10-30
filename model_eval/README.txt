This script will evaluate the model using PSNR, SSIM, and MSE metrics based on the given input files.

Class Definition: The Evaluation class includes methods for loading the model, loading test data, evaluating the model, and calculating PSNR, SSIM, and MSE.

Model Loading: The model weights are loaded from the specified path.

Data Loading: The audio data is loaded using librosa.

Evaluation: The evaluate_model method extracts the spectrogram from noisy audio, predicts using the model, and computes metrics.

Metrics Calculation: PSNR, SSIM, and MSE are calculated based on the clean and predicted audio signals.

Report Generation: The results are written to a text file.