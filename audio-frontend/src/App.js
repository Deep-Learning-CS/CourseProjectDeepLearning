// src/AudioUpload.js

import axios from "axios";
import React, { useState } from "react";
import "./AudioUpload.css"; // Import the CSS file

const AudioUpload = () => {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
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

      // Create a link to download the processed audio
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const a = document.createElement("a");
      a.href = url;
      a.download = "processed_audio.wav";
      document.body.appendChild(a);
      a.click();
      a.remove();

      setMessage("Audio processed successfully!");
    } catch (error) {
      setMessage("Error processing audio. Please try again.");
    }
  };

  return (
    <div className="upload-container">
      <h2 className="upload-title">Upload Audio File</h2>
      <input type="file" accept="audio/*" onChange={handleFileChange} className="file-input" />
      <button onClick={handleUpload} className="upload-button">Upload</button>
      {message && <p className="upload-message">{message}</p>}
    </div>
  );
};

export default AudioUpload;
