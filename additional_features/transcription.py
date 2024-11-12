from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
asr_model.config.forced_decoder_ids = None

def transcribe(filename):
    audio_data, sampling_rate = librosa.load(filename, sr=None) # sr=None to keep original sampling rate
    sample = {"array": audio_data, "sampling_rate": sampling_rate}

    input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 
    predicted_ids = asr_model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    return transcription

if __name__ == "__main__":
    transcription = transcribe('clean_audio.flac')
    print(transcription)
