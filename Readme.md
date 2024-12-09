# Noise Reduction using Deep Learning

## Overview
This project aims to reduce noise in audio signals using deep learning techniques. We leverage state-of-the-art models to enhance the quality of audio by minimizing unwanted noise while preserving important sound details. The project demonstrates the power of deep learning in real-world audio signal processing.

## Team Members

### 1. **Ayush** - *Preprocessing*
- Implemented audio data preprocessing pipeline
- Developed signal processing utilities
- Handled format conversions and standardization

### 2. **Luke** - *Model Development & Training*
- Designed and implemented the Luke model
- Conducted model training and optimization
- Specialized in spectrogram-based processing

### 3. **Drashti** - *Model Development & Training*
- Led SEGAN model implementation
- Managed training infrastructure
- Conducted performance optimization

### 4. **Anayat** - *Model Evaluation*
- Designed evaluation metrics
- Conducted comparative analysis
- Documented model performance

### 5. **Sanidhaya** - *Model Deployment*
- Implemented deployment infrastructure
- Managed cloud resources
- Handled CI/CD pipeline

## Live Demo
The project is deployed and accessible at: [Audio Noise Reducer](https://course-project-deep-learning-git-main-sandys-projects-9dc0cf01.vercel.app/)

⚠️ **Known Issues:**
- The Luke model is currently not functioning in the deployment environment (works locally)
- The SEGAN model's performance is not meeting expected quality standards

## Features

### Audio Input Methods
1. **File Upload**
   - Supports WAV and FLAC formats
   - Input validation for file types
   - Real-time feedback on upload status

2. **Direct Recording**
   - Browser-based audio recording
   - Real-time recording duration display
   - Automatic format conversion to WAV

### Noise Reduction Models
1. **DNS Model (DNS48)**
   - Facebook's Denoiser model
   - Best performing model in deployment
   - Optimized for real-time processing

2. **Luke Model**
   - Custom implementation using TensorFlow
   - Currently functioning in localhost only
   - Uses spectrogram-based processing

3. **SEGAN Model**
   - GAN-based noise reduction
   - Currently showing suboptimal performance
   - Implemented with PyTorch

## Technical Architecture

### Frontend (React)
- **Key Components:**
  - Audio recording handler
  - File upload interface
  - Model selection
  - Audio playback
  - Download functionality

### Backend (FastAPI)
- **Core Features:**
  - Multi-model support
  - Audio processing pipeline
  - Error handling
  - Format conversion
  - Sample rate adjustment

### Processing Pipeline
1. Audio input validation
2. Format standardization
3. Model-specific preprocessing
4. Noise reduction processing
5. Output format conversion

## Development Setup

### Prerequisites
```
Python 3.8+
Node.js 14+
```

### Backend Dependencies
```
fastapi==0.103.2
numpy==1.26.0
torch==2.0.1
torchaudio==2.0.2
tensorflow-cpu==2.14.0
librosa==0.10.1
scipy==1.11.3
```

### Frontend Dependencies
```
react
axios
web-audio-api
```

## Local Development

1. **Backend Setup**
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

2. **Frontend Setup**
```bash
npm install
npm start
```

## Usage Instructions

1. **File Upload Method**
   - Click "Upload Audio"
   - Select WAV or FLAC file
   - Choose noise reduction model
   - Click "Upload and Process"

2. **Recording Method**
   - Click "Record Audio"
   - Press record button (⏺)
   - Stop recording when finished (⏹)
   - Choose model and process

3. **Download Results**
   - Preview processed audio
   - Use download button for processed file
   - Compare original and processed audio

## Current Limitations

1. **Luke Model Deployment Issues**
   - Works in localhost environment
   - Memory allocation issues in deployment
   - Currently disabled in production

2. **SEGAN Model Performance**
   - Suboptimal noise reduction quality
   - Higher processing latency
   - Requires optimization

3. **General Limitations**
   - Maximum file size restrictions
   - Processing time varies by file size
   - Browser compatibility considerations

## Future Improvements

1. **Model Enhancements**
   - Fix Luke model deployment issues
   - Optimize SEGAN performance
   - Implement model versioning

2. **Feature Additions**
   - Real-time processing option
   - Additional audio formats
   - Batch processing capability

3. **UI/UX Improvements**
   - Progress indicators
   - Advanced audio visualization
   - Custom processing parameters

## Contributing
Contributions are welcome! Please submit issues and pull requests to the project repository.

## License
This project is licensed under the MIT License.