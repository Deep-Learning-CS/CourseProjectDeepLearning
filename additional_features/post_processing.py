## Apply equalizer to processed audio

import librosa
import soundfile as sf
import numpy as np

# Load the denoised audio (assuming it's a flac file)
audio_file = "clean_audio.flac"
y, sr = librosa.load(audio_file, sr=None)  # y is the audio signal, sr is the sample rate

from scipy.signal import butter, lfilter

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_bass_boost(audio, cutoff=200, fs=44100):
    b, a = butter_lowpass(cutoff, fs, order=5)
    return lfilter(b, a, audio)

def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def apply_treble_boost(audio, cutoff=5000, fs=44100):
    b, a = butter_highpass(cutoff, fs, order=5)
    return lfilter(b, a, audio)

def apply_equalizer(audio, bass_cutoff=150, treble_cutoff=5000, fs=44100):
    y_bass_boosted = apply_bass_boost(audio, cutoff=bass_cutoff, fs=fs)
    y_treble_boosted = apply_treble_boost(y_bass_boosted, cutoff=treble_cutoff, fs=fs)
    return y_treble_boosted

# Apply the custom equalizer to the denoised audio
y_equalized = apply_equalizer(y, bass_cutoff=150, treble_cutoff=5000, fs=sr)

y_equalized = y_equalized / np.max(np.abs(y_equalized))

# Save the processed (equalized) audio to a new file
output_file = "processed_audio.wav"
sf.write(output_file, y_equalized, sr)