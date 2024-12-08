import axios from 'axios';
import React, { useEffect, useRef, useState } from 'react';
import './AudioUpload.css';

const AudioUpload = () => {
  // State management
  const [file, setFile] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [processedAudio, setProcessedAudio] = useState(null);
  const [selectedModel, setSelectedModel] = useState('dns');
  const [status, setStatus] = useState('');
  const [error, setError] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [activeTab, setActiveTab] = useState('upload');

  // Refs
  const mediaRecorderRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const chunksRef = useRef([]);
  const timerIntervalRef = useRef(null);

  const BACKEND_URL = "http://localhost:8081";

  // Cleanup effect
  useEffect(() => {
    return () => {
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach(track => track.stop());
      }
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
      }
    };
  }, []);

  // Format recording time
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Handle file upload
  const handleFileChange = (event) => {
    const uploadedFile = event.target.files[0];
    if (uploadedFile) {
      if (!uploadedFile.name.toLowerCase().endsWith('.wav') && 
          !uploadedFile.name.toLowerCase().endsWith('.flac')) {
        setError('Please upload a WAV or FLAC file');
        setFile(null);
        event.target.value = null;
        return;
      }
      setFile(uploadedFile);
      setError('');
      setStatus('File selected');
    }
  };

  // Start recording
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false
        }
      });

      mediaStreamRef.current = stream;
      chunksRef.current = [];

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus',
        audioBitsPerSecond: 128000
      });

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      mediaRecorder.onstop = async () => {
        try {
          const webmBlob = new Blob(chunksRef.current, { type: 'audio/webm' });
          const wavBlob = await convertToWav(webmBlob);
          setRecordedBlob(wavBlob);
          setStatus('Recording saved');
        } catch (error) {
          console.error('Error converting recording:', error);
          setError('Error saving recording');
        }
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start(1000); // Collect data every second
      setIsRecording(true);
      setStatus('Recording...');
      setError('');
      
      setRecordingTime(0);
      timerIntervalRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);

    } catch (err) {
      console.error('Recording error:', err);
      setError('Error accessing microphone. Please check permissions.');
    }
  };

  // Stop recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach(track => track.stop());
      }
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
      }
      setIsRecording(false);
    }
  };

  // Convert audio to WAV format
  const convertToWav = async (blob) => {
    try {
      const audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000
      });
      
      const arrayBuffer = await blob.arrayBuffer();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      
      // Get audio data and handle resampling if needed
      let audioData = audioBuffer.getChannelData(0);
      if (audioBuffer.sampleRate !== 16000) {
        audioData = resampleAudio(audioData, audioBuffer.sampleRate, 16000);
      }
      
      const wavBuffer = createWavBuffer(audioData, 16000);
      return new Blob([wavBuffer], { type: 'audio/wav' });
    } catch (err) {
      console.error('Error converting audio:', err);
      throw err;
    }
  };

  // Resample audio to target sample rate
  const resampleAudio = (audioData, originalSampleRate, targetSampleRate) => {
    const ratio = originalSampleRate / targetSampleRate;
    const newLength = Math.round(audioData.length / ratio);
    const result = new Float32Array(newLength);
    
    for (let i = 0; i < newLength; i++) {
      const position = i * ratio;
      const index = Math.floor(position);
      result[i] = audioData[index];
    }
    
    return result;
  };

  // Create WAV buffer with correct duration
  const createWavBuffer = (audioData, sampleRate) => {
    const numChannels = 1;
    const bitsPerSample = 16;
    const bytesPerSample = bitsPerSample / 8;
    const blockAlign = numChannels * bytesPerSample;
    
    const arrayBuffer = new ArrayBuffer(44 + audioData.length * bytesPerSample);
    const view = new DataView(arrayBuffer);
    
    // Write WAV header
    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };
    
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + audioData.length * bytesPerSample, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM format
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitsPerSample, true);
    writeString(36, 'data');
    view.setUint32(40, audioData.length * bytesPerSample, true);
    
    // Write audio data
    const volume = 0.8;
    let offset = 44;
    for (let i = 0; i < audioData.length; i++) {
      const sample = Math.max(-1, Math.min(1, audioData[i]));
      view.setInt16(offset, sample * 0x7FFF * volume, true);
      offset += 2;
    }
    
    return arrayBuffer;
  };

  // Process audio
  const processAudio = async () => {
    const audioToProcess = file || recordedBlob;
    if (!audioToProcess) {
      setError('No audio to process');
      return;
    }

    const formData = new FormData();
    formData.append('file', audioToProcess, 'audio.wav');

    setIsProcessing(true);
    setError('');
    setStatus('Processing audio...');

    try {
      const response = await axios.put(
        `${BACKEND_URL}/process-audio/?model=${selectedModel}`,
        formData,
        {
          responseType: 'blob',
          headers: {
            'Accept': 'audio/wav',
            'Content-Type': 'multipart/form-data',
          },
          timeout: 30000,
        }
      );

      const audioUrl = URL.createObjectURL(new Blob([response.data], { type: 'audio/wav' }));
      setProcessedAudio(audioUrl);
      setStatus('Audio processed successfully!');
    } catch (err) {
      console.error('Processing error:', err);
      if (err.code === 'ECONNREFUSED') {
        setError('Cannot connect to server. Please ensure the backend is running.');
      } else if (err.response) {
        setError(`Error: ${err.response.data.detail || 'Server error'}`);
      } else {
        setError('Error processing audio. Please try again.');
      }
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="upload-container">
      <h1 className="upload-title">Audio Noise Reduction</h1>
      
      <div className="tab-container">
        <div className="tab-buttons">
          <button 
            className={`tab-button ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => setActiveTab('upload')}
          >
            Upload Audio
          </button>
          <button 
            className={`tab-button ${activeTab === 'record' ? 'active' : ''}`}
            onClick={() => setActiveTab('record')}
          >
            Record Audio
          </button>
        </div>
      </div>

      <div className="model-selection">
        <label htmlFor="model-select">Select Model:</label>
        <select
          id="model-select"
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          className="model-select"
          disabled={isProcessing}
        >
          <option value="dns">DNS Model</option>
          <option value="luke">Luke Model</option>
          <option value="segan">SEGAN Model</option>
        </select>
      </div>

      {activeTab === 'upload' ? (
        <div className="upload-section">
          <input
            type="file"
            accept=".wav,.flac"
            onChange={handleFileChange}
            className="file-input"
            disabled={isProcessing}
          />
          <button 
            onClick={processAudio}
            className={`upload-button ${isProcessing ? 'processing' : ''}`}
            disabled={!file || isProcessing}
          >
            {isProcessing ? 'Processing...' : 'Upload and Process'}
          </button>
        </div>
      ) : (
        <div className="recording-section">
          <div className="recording-controls">
            {!isRecording ? (
              <button 
                onClick={startRecording}
                className="record-button stopped"
                disabled={isProcessing}
              >
                ⏺
              </button>
            ) : (
              <button 
                onClick={stopRecording}
                className="record-button recording"
              >
                ⏹
              </button>
            )}
            <div className="timer">{formatTime(recordingTime)}</div>
          </div>
          
          <div className="recording-status">
            {isRecording ? 'Recording...' : recordedBlob ? 'Recording saved' : 'Ready to record'}
          </div>
          
          {recordedBlob && (
            <button 
              onClick={processAudio}
              className={`upload-button ${isProcessing ? 'processing' : ''}`}
              disabled={isProcessing}
            >
              {isProcessing ? 'Processing...' : 'Process Recording'}
            </button>
          )}
        </div>
      )}

      {(file || recordedBlob) && (
        <div className="audio-section">
          <h3>Original Audio:</h3>
          <audio 
            controls 
            src={file ? URL.createObjectURL(file) : URL.createObjectURL(recordedBlob)}
            className="audio-player"
          />
        </div>
      )}

      {processedAudio && (
        <div className="audio-section">
          <h3>Processed Audio:</h3>
          <audio 
            controls 
            src={processedAudio}
            className="audio-player"
          />
          <a
            href={processedAudio}
            download="processed_audio.wav"
            className="download-button"
          >
            Download Processed Audio
          </a>
        </div>
      )}

      {error && <div className="message error">{error}</div>}
      {status && !error && <div className="message success">{status}</div>}
    </div>
  );
};

export default AudioUpload;