from io import BytesIO
import torch
import torchaudio
import torch.nn as nn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.responses import StreamingResponse
from typing import Dict

app = FastAPI(
    title="Audio Noise Reduction API",
    description="Audio noise reduction API using Facebook's Denoiser model",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NoiseReducer(nn.Module):
    def __init__(self):
        super(NoiseReducer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load('facebookresearch/denoiser', 'dns48', pretrained=True)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def forward(self, audio, sample_rate):
        # Resample if needed (model expects 16kHz)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=16000
            ).to(self.device)
            audio = resampler(audio)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        # Ensure correct dimensions (batch, channel, time)
        if len(audio.shape) == 2:
            audio = audio.unsqueeze(0)  # Add batch dimension
        
        # Normalize audio
        audio = audio / torch.max(torch.abs(audio))
        
        # Process audio in chunks to avoid memory issues
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
        
        # Remove batch dimension
        processed_audio = processed_audio.squeeze(0)
        
        # Resample back to original sample rate if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=16000,
                new_freq=sample_rate
            )
            processed_audio = resampler(processed_audio)
        
        return processed_audio

# Initialize the noise reducer
try:
    noise_reducer = NoiseReducer()
except Exception as e:
    print(f"Error initializing model: {str(e)}")
    noise_reducer = None

@app.put(
    "/process-audio/",
    response_class=StreamingResponse,
    summary="Process audio file to reduce noise"
)
async def process_audio(
    file: UploadFile = File(..., description="Audio file to process (WAV format)")
) -> StreamingResponse:
    if noise_reducer is None:
        raise HTTPException(
            status_code=500,
            detail="Model not initialized properly"
        )

    if not file.filename.endswith('.wav'):
        raise HTTPException(
            status_code=400,
            detail="Only WAV files are supported"
        )
    
    try:
        # Read audio file
        audio_tensor, sample_rate = torchaudio.load(BytesIO(await file.read()))
        
        # Process audio
        enhanced_audio = noise_reducer(audio_tensor, sample_rate)
        
        # Normalize output
        enhanced_audio = enhanced_audio * 0.95 / torch.max(torch.abs(enhanced_audio))
        
        # Save to buffer
        buffer = BytesIO()
        torchaudio.save(buffer, enhanced_audio, sample_rate, format="wav")
        buffer.seek(0)
        
        return StreamingResponse(
            buffer, 
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename=processed_{file.filename}"}
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio: {str(e)}"
        )

@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Audio Noise Reduction API is running!"}

@app.get("/status")
async def status() -> Dict[str, str]:
    if noise_reducer is None:
        status_msg = "model initialization failed"
        device = "none"
    else:
        status_msg = "operational"
        device = str(next(noise_reducer.parameters()).device)
    
    return {
        "status": status_msg,
        "model": "Facebook Denoiser DNS48",
        "device": device
    }