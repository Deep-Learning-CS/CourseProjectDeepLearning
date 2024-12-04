This script will evaluate the model using PSNR, SSIM, and MSE metrics based on the given input files.

Class Definition: The Evaluation class includes methods for loading the model, loading test data, evaluating the model, and calculating PSNR, SSIM, and MSE.

Model Loading: The model weights are loaded from the specified path.

Data Loading: The audio data is loaded using librosa.

Evaluation: The evaluate_model method extracts the spectrogram from noisy audio, predicts using the model, and computes metrics.

Metrics Calculation: PSNR, SSIM, and MSE are calculated based on the clean and predicted audio signals.

Report Generation: The results are written to a text file.

# update 2:

Added SNR, PESQ and MCD metrics as mentioned by professor.

bugs: as we are currently working with multiple models with various functions, it wont be possible to create an all-in-one metrics script for all of them . However you should be able to import the functions/class in your existing model to compute metrics with miniumum changes.