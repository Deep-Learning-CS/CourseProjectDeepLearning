document.getElementById('start').addEventListener('click', startAudio);

let audioContext;
let mediaStream;
let audioSource;
let outputAudioElement = document.getElementById('outputAudio');

async function startAudio() {
  // Request access to the user's microphone
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    audioSource = audioContext.createMediaStreamSource(mediaStream);
    
    // Create an audio destination for output
    let destination = audioContext.createMediaStreamDestination();
    outputAudioElement.srcObject = destination.stream;
    
    // Load the RNNoise WebAssembly module
    rnnoise.onRuntimeInitialized = () => {
      const noiseSuppressor = new rnnoise.RnNoise();
      processAudioStream(audioSource, destination, noiseSuppressor);
    };
  } catch (err) {
    console.error("Error accessing microphone: ", err);
  }
}

function processAudioStream(audioSource, destination, noiseSuppressor) {
  const bufferSize = 4096;
  const inputNode = audioContext.createScriptProcessor(bufferSize, 1, 1);
  
  // Connect the input (microphone) to the processor and the output to the destination
  audioSource.connect(inputNode);
  inputNode.connect(destination);

  inputNode.onaudioprocess = function(event) {
    const inputData = event.inputBuffer.getChannelData(0);
    const outputData = event.outputBuffer.getChannelData(0);
    
    // Use the RNNoise module to suppress noise
    noiseSuppressor.process(inputData, outputData);
  };
}
