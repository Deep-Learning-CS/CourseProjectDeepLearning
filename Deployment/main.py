from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

app = FastAPI()

def noise_Reduction():
    return "noise reduced"

@app.get("/")
async def root():
    return {"message": "Hello, Professor!"}

@app.post("/noise_reduction")
async def upload_audio(file: UploadFile = File(...)):
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid audio file.")
    
    audio_data = await file.read()
    audio_stream = BytesIO(audio_data)
    
    return StreamingResponse(audio_stream, media_type=file.content_type)
    