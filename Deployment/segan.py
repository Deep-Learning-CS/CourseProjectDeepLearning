import logging
import os
from typing import Tuple

import numpy as np
import requests
import torch
import torch.nn as nn
import torchaudio

# Set up logging
logger = logging.getLogger(__name__)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # Encoder layers with batch normalization
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 16, 4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(16, 32, 4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2)
        )
        self.enc4 = nn.Sequential(
            nn.Conv1d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2)
        )
        
        # Decoder layers with batch normalization
        self.dec4 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(128, 32, 4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(64, 16, 4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(32, 1, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Decoder with skip connections
        d4 = self.dec4(e4)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        
        return d1

class PretrainedSeganNoiseReducer:
    def __init__(self):
        try:
            logger.info("Initializing SEGAN model...")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.generator = Generator().to(self.device)
            
            # Initialize model weights
            self.generator.apply(self._init_weights)
            
            logger.info("SEGAN model initialized")
            logger.info(f"Using device: {self.device}")
            
            # Parameters
            self.segment_length = 16384  # 1 second at 16kHz
            self.sample_rate = 16000
            
        except Exception as e:
            logger.error(f"Error initializing SEGAN model: {str(e)}")
            raise

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def process_audio(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Process audio using SEGAN model
        
        Args:
            audio (torch.Tensor): Input audio tensor
            sample_rate (int): Sample rate of input audio
            
        Returns:
            torch.Tensor: Enhanced audio tensor
        """
        try:
            with torch.no_grad():
                # Convert to tensor if numpy array
                if isinstance(audio, np.ndarray):
                    audio = torch.from_numpy(audio).float()
                
                # Handle audio format
                if len(audio.shape) == 3:  # (batch, channels, samples)
                    audio = audio.squeeze(0)
                
                # Convert stereo to mono if needed
                if len(audio.shape) == 2 and audio.shape[0] > 1:  # (channels, samples)
                    audio = torch.mean(audio, dim=0)
                
                # Ensure shape is (1, 1, samples)
                if len(audio.shape) == 1:  # (samples,)
                    audio = audio.unsqueeze(0).unsqueeze(0)
                elif len(audio.shape) == 2:  # (1, samples)
                    audio = audio.unsqueeze(0)
                
                # Resample if necessary
                if sample_rate != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sample_rate,
                        new_freq=self.sample_rate
                    ).to(self.device)
                    audio = resampler(audio)
                
                # Normalize
                audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
                
                # Store original length
                original_length = audio.shape[-1]
                
                # Process in chunks
                chunk_size = self.segment_length
                enhanced_chunks = []
                
                for start in range(0, original_length, chunk_size):
                    end = min(start + chunk_size, original_length)
                    chunk = audio[:, :, start:end]
                    
                    # Pad if necessary
                    if chunk.shape[-1] < chunk_size:
                        padding_size = chunk_size - chunk.shape[-1]
                        chunk = torch.nn.functional.pad(chunk, (0, padding_size), mode='reflect')
                    
                    # Process chunk
                    enhanced_chunk = self.generator(chunk.to(self.device))
                    
                    # Remove padding if any
                    if end == original_length and padding_size:
                        enhanced_chunk = enhanced_chunk[:, :, :-padding_size]
                    
                    enhanced_chunks.append(enhanced_chunk.cpu())
                
                # Combine all chunks
                enhanced_audio = torch.cat(enhanced_chunks, dim=2)
                
                # Ensure we match the original length
                enhanced_audio = enhanced_audio[:, :, :original_length]
                
                # Resample back if necessary
                if sample_rate != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=self.sample_rate,
                        new_freq=sample_rate
                    )
                    enhanced_audio = resampler(enhanced_audio)
                
                # Return as (channels, samples)
                return enhanced_audio.squeeze(0)
                
        except Exception as e:
            logger.error(f"Error in process_audio: {str(e)}")
            raise