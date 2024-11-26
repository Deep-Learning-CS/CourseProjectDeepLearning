import React, { useState, useEffect, useRef } from 'react';
import { Button } from '@/components/ui/button';

const NoiseReductionComponent = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [audioContext, setAudioContext] = useState(null);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const audioInputRef = useRef(null);
  const processedAudioRef = useRef(null);

  // Load RNNoise WebAssembly module
  useEffect(() => {
    const loadRNNoise = async () => {
      try {
        // Note: You'll need to host the rnnoise.wasm file
        const rnnoise = await import('@/lib/rnnoise-module');
        await rnnoise.default();
        console.log('RNNoise WebAssembly module loaded');
      } catch (error) {
        console.error('Failed to load RNNoise module:', error);
      }
    };

    loadRNNoise();
  }, []);

  const startRecording = async () => {
    try {
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false
        } 
      });

      // Create audio context and source
      const context = new AudioContext();
      const source = context.createMediaStreamSource(stream);

      // Create RNNoise noise reduction node (hypothetical)
      const noiseReductionNode = context.createScriptProcessor(4096, 1, 1);
      noiseReductionNode.onaudioprocess = (event) => {
        // Apply RNNoise noise reduction 
        // This is a placeholder - actual implementation depends on RNNoise.js integration
        const inputBuffer = event.inputBuffer.getChannelData(0);
        const outputBuffer = event.outputBuffer.getChannelData(0);
        
        // Simulated noise reduction
        inputBuffer.forEach((sample, index) => {
          // Basic noise reduction simulation
          outputBuffer[index] = sample * 0.8;
        });
      };

      // Connect audio graph
      source.connect(noiseReductionNode);
      noiseReductionNode.connect(context.destination);

      // Setup media recorder for processed audio
      const recorder = new MediaRecorder(stream);
      const audioChunks = [];

      recorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
      };

      recorder.onstop = () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);
        
        if (processedAudioRef.current) {
          processedAudioRef.current.src = audioUrl;
        }
      };

      // Start recording
      recorder.start();
      setIsRecording(true);
      setAudioContext(context);
      setMediaRecorder(recorder);

    } catch (error) {
      console.error('Error starting recording:', error);
    }
  };

  const stopRecording = () => {
    if (mediaRecorder) {
      mediaRecorder.stop();
      
      // Stop all tracks
      audioContext.destination.mediaStream.getTracks().forEach(track => track.stop());
      
      // Close audio context
      audioContext.close();
      
      setIsRecording(false);
      setAudioContext(null);
      setMediaRecorder(null);
    }
  };

  return (
    <div className="w-full max-w-md p-4 space-y-4">
      <h2 className="text-xl font-bold">RNNoise Noise Reduction Demo</h2>
      
      <div className="flex space-x-4">
        <Button 
          onClick={startRecording} 
          disabled={isRecording}
          className="w-full"
        >
          Start Recording
        </Button>
        <Button 
          onClick={stopRecording} 
          disabled={!isRecording}
          variant="destructive"
          className="w-full"
        >
          Stop Recording
        </Button>
      </div>

      {/* Processed Audio Playback */}
      <audio 
        ref={processedAudioRef} 
        controls 
        className="w-full mt-4"
      />
    </div>
  );
};

export default NoiseReductionComponent;