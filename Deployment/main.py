from io import BytesIO
import torch
import torchaudio
import torch.nn as nn
import tensorflow as tf
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse
from typing import Dict, Optional, Literal
import numpy as np
import logging
from scipy import signal

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

class LukeNoiseReducer:
    def __init__(self):
        try:
            logger.info("Loading Luke model...")
            self.model = tf.keras.models.load_model('modeldev_luke.h5', compile=False)
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
                    
            # Normalize audio
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            # Compute STFT with explicit parameters
            frequencies, times, Zxx = signal.stft(
                audio,
                fs=sr,
                window=self.window,
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                boundary='zeros'  # Use zeros padding for boundary
            )
            
            # Get magnitude spectrogram
            spectrogram = np.abs(Zxx)
            
            # Store original shapes for later use
            self.original_freq_len = spectrogram.shape[0]
            self.original_time_len = spectrogram.shape[1]
            
            # Trim or pad to 128 frequency bins
            if spectrogram.shape[0] > 128:
                spectrogram = spectrogram[:128, :]
                Zxx = Zxx[:128, :]
            else:
                pad_size = 128 - spectrogram.shape[0]
                spectrogram = np.pad(spectrogram, ((0, pad_size), (0, 0)))
                Zxx = np.pad(Zxx, ((0, pad_size), (0, 0)))
            
            return spectrogram, Zxx, frequencies, times
            
        except Exception as e:
            logger.error(f"Error in prepare_spectrogram: {str(e)}")
            raise

    def process_audio(self, audio, sample_rate):
        try:
            # Handle input audio format
            if isinstance(audio, torch.Tensor):
                if len(audio.shape) == 3:
                    audio = audio.squeeze(0)
                if audio.shape[0] > 1:
                    audio = torch.mean(audio, dim=0)
                audio = audio.numpy()
            
            # Resample if necessary
            if sample_rate != 16000:
                num_samples = int(len(audio) * 16000 / sample_rate)
                audio = signal.resample(audio, num_samples)
                sample_rate = 16000
            
            # Prepare spectrogram
            spectrogram, Zxx, frequencies, times = self.prepare_spectrogram(audio, sample_rate)
            
            # Process in chunks
            num_frames = spectrogram.shape[1]
            processed_frames = []
            
            for i in range(0, num_frames, 32):
                chunk = spectrogram[:, i:i+32]
                if chunk.shape[1] < 32:
                    pad_size = 32 - chunk.shape[1]
                    chunk = np.pad(chunk, ((0, 0), (0, pad_size)))
                
                model_input = chunk.reshape(1, 128, 32, 1)
                batch_output = self.model.predict(model_input, verbose=0)
                
                valid_frames = min(32, num_frames - i)
                if valid_frames < 32:
                    batch_output = batch_output[:, :, :valid_frames, :]
                
                processed_frames.append(batch_output.squeeze())
            
            # Combine processed frames
            processed_spectrogram = np.concatenate(processed_frames, axis=1)
            
            # Match dimensions with original
            processed_spectrogram = processed_spectrogram[:, :self.original_time_len]
            
            # Reconstruct with original phase
            phase = np.angle(Zxx[:128, :])
            reconstructed_complex = processed_spectrogram * np.exp(1j * phase)
            
            # Add back original frequency dimensions if needed
            if self.original_freq_len > 128:
                pad_size = self.original_freq_len - 128
                reconstructed_complex = np.pad(reconstructed_complex, ((0, pad_size), (0, 0)))
            
            # Inverse STFT with same parameters
            _, reconstructed_audio = signal.istft(
                reconstructed_complex,
                fs=sample_rate,
                window=self.window,
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                boundary='zeros'
            )
            
            # Convert back to original sample rate if needed
            if sample_rate != 16000:
                num_samples = int(len(reconstructed_audio) * sample_rate / 16000)
                reconstructed_audio = signal.resample(reconstructed_audio, num_samples)
            
            return torch.from_numpy(reconstructed_audio).float()
            
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

# Initialize models
try:
    dns_reducer = NoiseReducer()
    luke_reducer = LukeNoiseReducer()
    models_status = "operational"
    logger.info("Both models initialized successfully")
except Exception as e:
    logger.error(f"Error during model initialization: {str(e)}")
    models_status = "partial" if (dns_reducer is not None or luke_reducer is not None) else "failed"
    dns_reducer = None
    luke_reducer = None

@app.put(
    "/process-audio/",
    response_class=StreamingResponse,
    summary="Process audio file to reduce noise"
)
async def process_audio(
    file: UploadFile = File(..., description="Audio file to process (WAV or FLAC format)"),
    model: Literal["dns", "luke"] = Query(
        default="dns",
        description="Select the model to use for processing"
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
            enhanced_audio = luke_reducer.process_audio(audio_tensor, sample_rate)
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
    
    device = str(next(dns_reducer.parameters()).device) if dns_reducer is not None else "none"
    
    return {
        "status": models_status,
        "available_models": available_models,
        "device": device
    }