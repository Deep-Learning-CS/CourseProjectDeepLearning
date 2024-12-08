import os

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import logging
from io import BytesIO
from typing import Dict, Literal, Optional

import librosa
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torchaudio
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from scipy import signal
from segan import PretrainedSeganNoiseReducer
from starlette.responses import StreamingResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Audio Noise Reduction API",
    description="Audio noise reduction API supporting multiple denoising models",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Function to perform SSIM loss -> needed for Luke's model
@tf.keras.utils.register_keras_serializable()
def ssim_loss(y_true, y_pred):
  """
  Compute the SSIM loss between the true and predicted spectrograms.
  Parameters:
  y_true (tf.Tensor): True spectrogram.
  y_pred (tf.Tensor): Predicted spectrogram.
  """
  return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

class LukeNoiseReducer:
    def __init__(self):
        try:
            logger.info("Loading Luke model...")
            self.model = tf.keras.models.load_model('modeldev_luke.keras', compile=False)
            logger.info("Luke model loaded successfully")
            logger.info(f"Model input shape: {self.model.input_shape}")
            logger.info(f"Model output shape: {self.model.output_shape}")
            # Store STFT parameters as class variables for consistency
            self.nperseg = 256
            self.noverlap = self.nperseg // 2
            self.window = signal.windows.hann(self.nperseg)
        except Exception as e:
            logger.error(f"Error loading Luke model: {str(e)}")
            raise

    def prepare_spectrogram(self, audio, sr=16000):
        """Convert audio to spectrogram format"""
        try:
            # Ensure proper shape and type
            if isinstance(audio, torch.Tensor):
                audio = audio.numpy()
            
            if len(audio.shape) > 1:
                if audio.shape[0] > 1:  # If stereo, convert to mono
                    audio = np.mean(audio, axis=0)
                else:
                    audio = audio.squeeze()
                    
            # constants
            N_FFT = 2048
            HOP_LENGTH = 512
            WINDOW_DURATION = 1.02
            MIN_VAL = -80
            MAX_VAL = 0

            # Get phase information for better post-processing
            stft_result = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
            phase = np.angle(stft_result)
            magnitude = np.abs(stft_result)

            window_samples = int(sr * WINDOW_DURATION)
            spectrograms = []

            # divide audio into chunks of constant size and extract spectrograms
            # from audio [shape: (128, 32)]
            for start in range(0, len(audio), window_samples):
                end = start + window_samples

                window = audio[start:end]

                if len(window) < window_samples:
                    # pad with silence until the window is filled
                    padding = window_samples - len(window)
                    window = np.pad(window, (0, padding), mode='constant')

                spec = librosa.feature.melspectrogram(y=window, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
                spec_db = librosa.power_to_db(spec, ref=np.max)
                spectrograms.append(spec_db)

            # Add single channel for grey-scale RGB
            rgb_spectrograms = np.expand_dims(spectrograms, axis=-1)
            # Normalize the spectrograms
            normalized_spectrograms = (rgb_spectrograms - MIN_VAL) / (MAX_VAL - MIN_VAL)
            
            return normalized_spectrograms, phase, magnitude
            
        except Exception as e:
            logger.error(f"Error in prepare_spectrogram: {str(e)}")
            raise

    def process_audio(self, audio, sample_rate):
        try:
            # constants
            TARGET_SAMPLE_RATE = 16000
            N_FFT = 2048
            HOP_LENGTH = 512
            MIN_VAL = -80
            MAX_VAL = 0

            # Resample the audio so that it has the right sample rate
            resampled_audio = audio
            if sample_rate != TARGET_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
                resampled_audio = resampler(audio)

            # process and get the spectrograms
            normalized_spectrograms, phase, magnitude = self.prepare_spectrogram(resampled_audio, TARGET_SAMPLE_RATE)

            # Get predictions from model
            predictions = self.model.predict(normalized_spectrograms, verbose=0)

            # Unnormalize, undo single-channel RGB, combine the spectrograms together
            unscaled_predictions = predictions * (MAX_VAL - MIN_VAL) + MIN_VAL
            predicted_spectrograms = np.squeeze(unscaled_predictions, axis=-1)
            combined_spectrogram = np.hstack(predicted_spectrograms)
            
            # Inverse the spectrograms back into audio
            mel_spectrogram = librosa.db_to_power(combined_spectrogram)
            linear_spectrogram = librosa.feature.inverse.mel_to_stft(mel_spectrogram, sr=TARGET_SAMPLE_RATE, n_fft=N_FFT)
            # Ensure linear spectrogram matches phase dimensions
            min_time_frames = min(linear_spectrogram.shape[1], phase.shape[1])
            linear_spectrogram = linear_spectrogram[:, :min_time_frames]
            phase = phase[:, :min_time_frames]
            # reconstruct stft using phase information
            reconstructed_stft = linear_spectrogram * np.exp(1j * phase)

            # Get the audio from the STFT
            audio_signal = librosa.istft(reconstructed_stft, hop_length=HOP_LENGTH, n_fft=N_FFT)
            audio_signal *= 2
            
            # Return the audio signal and new sample rate
            return torch.from_numpy(audio_signal), TARGET_SAMPLE_RATE
            
        except Exception as e:
            logger.error(f"Error in process_audio: {str(e)}")
            raise

class NoiseReducer(nn.Module):
    def __init__(self):
        super(NoiseReducer, self).__init__()
        try:
            logger.info("Loading DNS model...")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = torch.hub.load('facebookresearch/denoiser', 'dns48', pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            logger.info("DNS model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading DNS model: {str(e)}")
            raise

    @torch.no_grad()
    def forward(self, audio, sample_rate):
        try:
            # Ensure audio is in the correct format (batch, channels, samples)
            if len(audio.shape) == 1:  # If 1D tensor
                audio = audio.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            elif len(audio.shape) == 2:  # If 2D tensor (channels, samples)
                audio = audio.unsqueeze(0)  # Add batch dimension
            
            # Resample if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, 
                    new_freq=16000
                ).to(self.device)
                audio = resampler(audio)
            
            # Convert stereo to mono if needed
            if audio.shape[1] > 1:
                audio = torch.mean(audio, dim=1, keepdim=True)
            
            # Normalize
            audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
            
            # Process in chunks if audio is too long
            chunk_size = 16000 * 30  # 30 seconds chunks
            if audio.shape[2] > chunk_size:
                chunks = torch.split(audio, chunk_size, dim=2)
                processed_chunks = []
                for chunk in chunks:
                    processed_chunk = self.model(chunk.to(self.device))
                    processed_chunks.append(processed_chunk.cpu())
                processed_audio = torch.cat(processed_chunks, dim=2)
            else:
                processed_audio = self.model(audio.to(self.device)).cpu()
            
            # Resample back to original rate if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=16000,
                    new_freq=sample_rate
                )
                processed_audio = resampler(processed_audio)
            
            # Return as 2D tensor (channels, samples)
            return processed_audio.squeeze(0)
            
        except Exception as e:
            logger.error(f"Error in DNS forward pass: {str(e)}")
            raise

# Initialize models individually for better error handling
dns_reducer = None
luke_reducer = None
segan_reducer = None

try:
    logger.info("Starting DNS model initialization...")
    dns_reducer = NoiseReducer()
    logger.info("DNS model initialized successfully")
except Exception as e:
    logger.error(f"Error initializing DNS model: {str(e)}")

try:
    luke_reducer = LukeNoiseReducer()
    logger.info("Luke model initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Luke model: {str(e)}")

try:
    segan_reducer = PretrainedSeganNoiseReducer()
    logger.info("SEGAN model initialized successfully")
except Exception as e:
    logger.error(f"Error initializing SEGAN model: {str(e)}")

# Determine overall status after all initializations
models_status = "operational" if all([dns_reducer, luke_reducer, segan_reducer]) else \
                "partial" if any([dns_reducer, luke_reducer, segan_reducer]) else \
                "failed"

logger.info(f"Models initialization completed. Status: {models_status}")

@app.put(
    "/process-audio/",
    response_class=StreamingResponse,
    summary="Process audio file to reduce noise"
)
async def process_audio(
    file: UploadFile = File(..., description="Audio file to process (WAV or FLAC format)"),
    model: Literal["dns", "luke", "segan"] = Query(
        default="dns",
        description="Select the model to use for processing (dns, luke, or segan)"
    )
) -> StreamingResponse:
    if model == "dns" and dns_reducer is None:
        raise HTTPException(
            status_code=503,
            detail="DNS model not available"
        )
    if model == "luke" and luke_reducer is None:
        raise HTTPException(
            status_code=503,
            detail="Luke model not available"
        )
    if model == "segan" and segan_reducer is None:
        raise HTTPException(
            status_code=503,
            detail="SEGAN model not available"
        )

    if not (file.filename.endswith('.wav') or file.filename.endswith('.flac')):
        raise HTTPException(
            status_code=400,
            detail="Only WAV and FLAC files are supported"
        )
    
    try:
        # Read audio file
        audio_tensor, sample_rate = torchaudio.load(BytesIO(await file.read()))
        
        # Process audio with selected model
        if model == "luke":
            enhanced_audio, adjusted_sample_rate = luke_reducer.process_audio(audio_tensor, sample_rate)

            # Ensure output is 2D (channels, samples) for saving
            if len(enhanced_audio.shape) == 1:
                enhanced_audio = enhanced_audio.unsqueeze(0)

            # Save to buffer
            buffer = BytesIO()
            torchaudio.save(buffer, enhanced_audio, adjusted_sample_rate, format="wav")
            buffer.seek(0)
            
            # Send file as response
            return StreamingResponse(
                buffer, 
                media_type="audio/wav",
                headers={"Content-Disposition": f"attachment; filename=processed_{file.filename.rsplit('.', 1)[0]}.wav"}
            )
        elif model == "segan":
            enhanced_audio = segan_reducer.process_audio(audio_tensor, sample_rate)
        else:  # dns
            enhanced_audio = dns_reducer(audio_tensor, sample_rate)
        
        # Ensure output is 2D (channels, samples) for saving
        if len(enhanced_audio.shape) == 1:
            enhanced_audio = enhanced_audio.unsqueeze(0)
        
        # Normalize output
        enhanced_audio = enhanced_audio * 0.95 / (torch.max(torch.abs(enhanced_audio)) + 1e-8)
        
        # Save to buffer
        buffer = BytesIO()
        torchaudio.save(buffer, enhanced_audio, sample_rate, format="wav")
        buffer.seek(0)
        
        return StreamingResponse(
            buffer, 
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename=processed_{file.filename.rsplit('.', 1)[0]}.wav"}
        )

    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio: {str(e)}"
        )

@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Audio Noise Reduction API is running!"}

@app.get("/status")
async def status() -> Dict[str, str]:
    available_models = []
    if dns_reducer is not None:
        available_models.append("DNS48")
    if luke_reducer is not None:
        available_models.append("Luke")
    if segan_reducer is not None:
        available_models.append("SEGAN")
    
    device = str(next(dns_reducer.parameters()).device) if dns_reducer is not None else "none"
    
    return {
        "status": models_status,
        "available_models": available_models,
        "device": device
    }