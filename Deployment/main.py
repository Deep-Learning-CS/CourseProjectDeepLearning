from io import BytesIO

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
from starlette.responses import StreamingResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.put("/process-audio/", response_class=StreamingResponse)
async def process_audio(file: UploadFile):
    try:
        audio = AudioSegment.from_file(file.file)
        
        processed_audio = audio + 5
        
        buffer = BytesIO()
        processed_audio.export(buffer, format="wav")
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="audio/wav", headers={"Content-Disposition": "attachment; filename=processed_audio.wav"})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Hello, Professor!"}

