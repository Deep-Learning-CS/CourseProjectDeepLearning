import axios from "axios";
import React, { useState } from "react";
import "./AudioUpload.css";

const AudioUpload = () => {
  const [file, setFile] = useState(null);
  const [uploadedAudioUrl, setUploadedAudioUrl] = useState("");
  const [processedAudioUrl, setProcessedAudioUrl] = useState("");
  const [message, setMessage] = useState("");

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);

    // Create a local URL for the uploaded audio file to play it
    if (selectedFile) {
      const url = URL.createObjectURL(selectedFile);
      setUploadedAudioUrl(url);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setMessage("Please select an audio file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.put("http://127.0.0.1:8081/process-audio/", formData, {
        responseType: "blob",
      });

      const processedUrl = window.URL.createObjectURL(new Blob([response.data]));
      setProcessedAudioUrl(processedUrl);

      setMessage("Audio processed successfully!");
    } catch (error) {
      setMessage("Error processing audio. Please try again.");
    }
  };

  return (
    <div className="upload-container">
      <h2 className="upload-title">Upload Audio File</h2>
      <input
        type="file"
        accept="audio/*"
        onChange={handleFileChange}
        className="file-input"
      />
      {uploadedAudioUrl && (
        <div>
          <h3>Uploaded Audio:</h3>
          <audio controls src={uploadedAudioUrl}></audio>
        </div>
      )}
      <button onClick={handleUpload} className="upload-button">
        Upload and Process
      </button>
      {processedAudioUrl && (
        <div>
          <h3>Processed Audio:</h3>
          <audio controls src={processedAudioUrl}></audio>
          <a href={processedAudioUrl} download="processed_audio.wav" className="download-link">
            Download Processed Audio
          </a>
        </div>
      )}
      {message && <p className="upload-message">{message}</p>}
    </div>
  );
};

export default AudioUpload;
