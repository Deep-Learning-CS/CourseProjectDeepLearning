from io import BytesIO

import torch
import torchaudio
from df import enhance, init_df
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model, df_state, _ = init_df()

@app.put("/process-audio/", response_class=StreamingResponse)
async def process_audio(file: UploadFile):
    try:
        audio_tensor, sample_rate = torchaudio.load(BytesIO(await file.read()))

        enhanced_audio_tensor = enhance(model, df_state, audio_tensor)

        buffer = BytesIO()
        torchaudio.save(buffer, enhanced_audio_tensor, sample_rate, format="wav")
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="audio/wav", headers={
            "Content-Disposition": "attachment; filename=processed_audio.wav"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Hello, Professor!"}
