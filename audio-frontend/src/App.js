// AudioUpload.js
import axios from "axios";
import React, { useState } from "react";
import "./AudioUpload.css";

const AudioUpload = () => {
  const [file, setFile] = useState(null);
  const [uploadedAudioUrl, setUploadedAudioUrl] = useState("");
  const [processedAudioUrl, setProcessedAudioUrl] = useState("");
  const [message, setMessage] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedModel, setSelectedModel] = useState("dns"); // Add model selection state

  const BACKEND_URL = "http://localhost:8000";

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile && !selectedFile.name.toLowerCase().endsWith('.wav')) {
      setMessage("Please select a WAV file.");
      setFile(null);
      setUploadedAudioUrl("");
      return;
    }
    
    setFile(selectedFile);
    setMessage("");

    if (selectedFile) {
      const url = URL.createObjectURL(selectedFile);
      setUploadedAudioUrl(url);
    }
  };

  const handleModelChange = (event) => {
    setSelectedModel(event.target.value);
  };

  const handleUpload = async () => {
    if (!file) {
      setMessage("Please select an audio file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    setIsProcessing(true);
    setMessage("Processing audio...");

    try {
      const response = await axios.put(
        `${BACKEND_URL}/process-audio/?model=${selectedModel}`,
        formData,
        {
          responseType: "blob",
          headers: {
            'Accept': 'audio/wav',
            'Content-Type': 'multipart/form-data',
          },
          timeout: 30000,
        }
      );

      const processedUrl = window.URL.createObjectURL(new Blob([response.data], { type: 'audio/wav' }));
      setProcessedAudioUrl(processedUrl);
      setMessage("Audio processed successfully!");
    } catch (error) {
      console.error("Error details:", error);
      if (error.code === 'ECONNREFUSED') {
        setMessage("Cannot connect to server. Please ensure the backend is running.");
      } else if (error.response) {
        setMessage(`Error: ${error.response.data.detail || 'Server error'}`);
      } else if (error.request) {
        setMessage("No response from server. Please try again.");
      } else {
        setMessage("Error processing audio. Please try again.");
      }
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="upload-container">
      <h2 className="upload-title">Audio Noise Reduction</h2>
      <div className="upload-section">
        <div className="model-selection">
          <label htmlFor="model-select">Select Model:</label>
          <select
            id="model-select"
            value={selectedModel}
            onChange={handleModelChange}
            className="model-select"
            disabled={isProcessing}
          >
            <option value="dns">DNS Model</option>
            <option value="luke">Luke Model</option>
          </select>
        </div>
        <input
          type="file"
          accept=".wav"
          onChange={handleFileChange}
          className="file-input"
          disabled={isProcessing}
        />
        <button 
          onClick={handleUpload} 
          className={`upload-button ${isProcessing ? 'processing' : ''}`}
          disabled={!file || isProcessing}
        >
          {isProcessing ? 'Processing...' : 'Upload and Process'}
        </button>
      </div>

      {uploadedAudioUrl && (
        <div className="audio-section">
          <h3>Original Audio:</h3>
          <audio controls src={uploadedAudioUrl} className="audio-player"></audio>
        </div>
      )}

      {processedAudioUrl && (
        <div className="audio-section">
          <h3>Processed Audio:</h3>
          <audio controls src={processedAudioUrl} className="audio-player"></audio>
          <a 
            href={processedAudioUrl} 
            download="processed_audio.wav" 
            className="download-button"
          >
            Download Processed Audio
          </a>
        </div>
      )}

      {message && <p className={`message ${message.includes('Error') ? 'error' : 'success'}`}>
        {message}
      </p>}
    </div>
  );
};

export default AudioUpload;