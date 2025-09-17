import React, { useState, useEffect, useRef } from 'react';
import { Text, View, StyleSheet, TouchableOpacity, Alert } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { Buffer } from 'buffer';
import { GoogleGenerativeAI } from '@google/generative-ai';
// Import TensorFlow.js and COCO-SSD
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { decodeJpeg } from '@tensorflow/tfjs-react-native';

const GEMINI_API_KEY = 'add your Gemini API key here';

export default function App() {
  const cameraRef = useRef(null);
  const [facing, setFacing] = useState('back');
  const [permission, requestPermission] = useCameraPermissions();
  const [cameraReady, setCameraReady] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [lastPrediction, setLastPrediction] = useState(null);
  const [geminiReady, setGeminiReady] = useState(false);
  const [autoDetection, setAutoDetection] = useState(false);
  const [tensorflowReady, setTensorflowReady] = useState(false);
  
  const genAI = useRef(null);
  const model = useRef(null);
  const cocoModel = useRef(null);
  const intervalRef = useRef(null);
  const lastCardDetected = useRef(null);
  const lastObjectState = useRef(null);

  // Initialize TensorFlow and models
  useEffect(() => {
    const initializeModels = async () => {
      try {
        // Initialize TensorFlow.js platform for React Native
        await tf.ready();
        console.log('TensorFlow.js is ready!');
        console.log('Platform: ', tf.getBackend());

        // Load COCO-SSD model
        cocoModel.current = await cocoSsd.load();
        console.log('COCO-SSD model loaded!');
        setTensorflowReady(true);

        // Initialize Gemini
        if (!GEMINI_API_KEY || GEMINI_API_KEY === 'YOUR_GEMINI_API_KEY') {
          throw new Error('Gemini API key is not set. Please add your key.');
        }
        genAI.current = new GoogleGenerativeAI(GEMINI_API_KEY);
        model.current = genAI.current.getGenerativeModel({ model: 'gemini-2.5-flash-lite' });
        setGeminiReady(true);
        console.log('All models initialized!');

      } catch (error) {
        console.error('Error initializing models:', error);
        Alert.alert('Initialization Error', 'Failed to load detection models. Check console for details.');
      }
    };

    initializeModels();
  }, []);

  // Detect card-like objects using COCO-SSD
  const detectObjects = async (base64String) => {
    if (!cocoModel.current) return null;

    try {
      // Convert base64 to Uint8Array
      const imageBuffer = Buffer.from(base64String, 'base64');
      const uint8Array = new Uint8Array(imageBuffer);
      
      // Decode JPEG to tensor using TensorFlow.js React Native
      const imageTensor = decodeJpeg(uint8Array);
      
      const predictions = await cocoModel.current.detect(imageTensor);
      
      // Clean up tensor
      imageTensor.dispose();
      
      // Filter for card-like objects (books, paper, etc.)
      const cardLikeObjects = predictions.filter(prediction => {
        const className = prediction.class.toLowerCase();
        return (
          className.includes('book') ||
          className.includes('paper') ||
          className.includes('magazine') ||
          // Add more relevant classes that might detect cards
          (prediction.score > 0.3 && // Reasonable confidence
           prediction.bbox[2] > 50 && // Width > 50px
           prediction.bbox[3] > 70 && // Height > 70px
           prediction.bbox[2] / prediction.bbox[3] > 0.5 && // Not too thin
           prediction.bbox[2] / prediction.bbox[3] < 2) // Not too wide
        );
      });

      return cardLikeObjects;
    } catch (error) {
      console.error('COCO-SSD detection error:', error);
      return null;
    }
  };

  // Check if object state has changed significantly
  const hasObjectStateChanged = (currentObjects) => {
    const currentState = {
      count: currentObjects ? currentObjects.length : 0,
      positions: currentObjects ? currentObjects.map(obj => ({
        x: Math.round(obj.bbox[0] / 30) * 30, // Quantize position to reduce noise
        y: Math.round(obj.bbox[1] / 30) * 30,
        w: Math.round(obj.bbox[2] / 30) * 30,
        h: Math.round(obj.bbox[3] / 30) * 30
      })) : []
    };

    if (!lastObjectState.current) {
      lastObjectState.current = currentState;
      return currentState.count > 0; // Changed if we now detect objects
    }

    // Check if count changed
    if (currentState.count !== lastObjectState.current.count) {
      lastObjectState.current = currentState;
      return true;
    }

    // Check if positions changed significantly (for same count)
    if (currentState.count > 0) {
      const positionsChanged = !currentState.positions.every((pos, index) => {
        const lastPos = lastObjectState.current.positions[index];
        if (!lastPos) return false;
        const distance = Math.sqrt(
          Math.pow(pos.x - lastPos.x, 2) + Math.pow(pos.y - lastPos.y, 2)
        );
        return distance < 60; // Allow some movement tolerance
      });

      if (positionsChanged) {
        lastObjectState.current = currentState;
        return true;
      }
    }

    return false;
  };

  const fileToGenerativePart = (base64String) => {
    return {
      inlineData: {
        data: base64String,
        mimeType: 'image/jpeg',
      },
    };
  };

  const predictCard = async (imageBase64) => {
    if (!geminiReady || !tensorflowReady || isProcessing) return;

    setIsProcessing(true);

    try {
      // First, check for card-like objects using COCO-SSD
      const detectedObjects = await detectObjects(imageBase64);
      
      // Only proceed with Gemini if object state changed
      if (!hasObjectStateChanged(detectedObjects)) {
        console.log('No significant object changes detected, skipping API call');
        setIsProcessing(false);
        return;
      }

      console.log(`Detected ${detectedObjects?.length || 0} card-like objects, calling Gemini...`);

      const imagePart = fileToGenerativePart(imageBase64);
      const prompt = "What playing card is visible in this image? Respond only with the card's name, like 'Ace of Spades' or 'Jack of Hearts'. If no playing card is clearly visible, respond with 'Not a card'.";
      const result = await model.current.generateContent([prompt, imagePart]);
      const response = await result.response;
      const text = response.text().trim();

      // Skip if it's the same card as last time
      if (text === lastCardDetected.current) {
        console.log('Same card detected, skipping notification');
        setIsProcessing(false);
        return;
      }

      lastCardDetected.current = text;

      const resultObject = {
        card: text,
        confidence: text === 'Not a card' ? 'N/A' : 'High',
        objectsDetected: detectedObjects?.length || 0
      };

      setLastPrediction(resultObject);
      
      // Only show alert for actual cards
      if (text !== 'Not a card') {
        triggerCardDetection(resultObject);
      }

    } catch (error) {
      console.error('Error predicting card:', error);
      if (!autoDetection) {
        Alert.alert('Prediction Error', 'Failed to analyze image.');
      }
    } finally {
      setIsProcessing(false);
    }
  };

  const triggerCardDetection = (prediction) => {
    Alert.alert(
      '🎴 New Card Detected!',
      `${prediction.card}\nConfidence: ${prediction.confidence}\nObjects detected: ${prediction.objectsDetected}`,
      [{ text: 'OK', style: 'default' }]
    );
  };

  const handleCameraReady = () => {
    setCameraReady(true);
  };

  const captureAndAnalyze = async (isAutoCapture = false) => {
    if (!cameraReady || !geminiReady || !tensorflowReady || (isProcessing && !isAutoCapture)) {
      return;
    }

    try {
      if (!cameraRef.current) {
        if (!isAutoCapture) {
          Alert.alert('Camera Error', 'Camera not ready to capture a photo.');
        }
        return;
      }

      const photo = await cameraRef.current.takePictureAsync({
        base64: true,
        quality: 0.8,
      });

      if (!photo?.base64) {
        if (!isAutoCapture) {
          Alert.alert('Capture Error', 'Failed to obtain image data from camera.');
        }
        return;
      }

      await predictCard(photo.base64);

    } catch (error) {
      console.error('Error capturing image:', error);
      if (!isAutoCapture) {
        Alert.alert('Error', 'Failed to capture image');
      }
    }
  };

  const toggleAutoDetection = () => {
    if (autoDetection) {
      // Stop auto detection
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      setAutoDetection(false);
      lastObjectState.current = null;
      lastCardDetected.current = null;
    } else {
      // Start auto detection
      if (!tensorflowReady || !geminiReady) {
        Alert.alert('Not Ready', 'Please wait for all models to load before starting auto-detection.');
        return;
      }
      
      setAutoDetection(true);
      intervalRef.current = setInterval(() => {
        if (cameraReady && geminiReady && tensorflowReady && !isProcessing) {
          captureAndAnalyze(true);
        }
      }, 1500); // Check every 1.5 seconds
    }
  };

  // Cleanup interval on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  function toggleCameraFacing() {
    setFacing(current => (current === 'back' ? 'front' : 'back'));
    // Reset detection state when switching cameras
    lastObjectState.current = null;
    lastCardDetected.current = null;
  }

  if (!permission) {
    return (
      <View style={styles.container}>
        <Text style={styles.message}>Loading camera...</Text>
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.message}>We need your permission to show the camera</Text>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Smart Card Detection Camera</Text>

      <View style={styles.statusContainer}>
        <Text style={styles.statusText}>
          TensorFlow: {tensorflowReady ? '✅' : '⏳'} |
          Gemini: {geminiReady ? '✅' : '⏳'} |
          Camera: {cameraReady ? '✅' : '⏳'} |
          Auto: {autoDetection ? '🔄' : '⏸️'}
        </Text>
      </View>

      <CameraView
        ref={cameraRef}
        style={styles.camera}
        facing={facing}
        onCameraReady={handleCameraReady}
      />

      <View style={styles.overlay}>
        <View style={styles.targetFrame}>
          <Text style={styles.overlayText}>
            {!cameraReady ? 'Starting camera...' :
             !tensorflowReady ? 'Loading TensorFlow...' :
             !geminiReady ? 'Connecting to API...' :
             isProcessing ? 'Processing image...' :
             autoDetection ? 'Smart detection active' :
             'Ready for detection'}
          </Text>

          {lastPrediction && (
            <Text style={styles.predictionText}>
              Last: {lastPrediction.card} 
              {lastPrediction.objectsDetected > 0 && ` (${lastPrediction.objectsDetected} objects)`}
            </Text>
          )}
        </View>

        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.flipButton} onPress={toggleCameraFacing}>
            <Text style={styles.buttonText}>Flip Camera</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.autoButton, autoDetection && styles.activeButton]}
            onPress={toggleAutoDetection}
            disabled={!cameraReady || !geminiReady || !tensorflowReady}
          >
            <Text style={styles.buttonText}>
              {autoDetection ? 'Stop Auto' : 'Start Auto'}
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.captureButton, isProcessing && styles.disabledButton]}
            onPress={captureAndAnalyze}
            disabled={!cameraReady || !geminiReady || !tensorflowReady || isProcessing}
          >
            <Text style={styles.buttonText}>
              {isProcessing ? 'Processing...' : 'Detect Now'}
            </Text>
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
    justifyContent: 'center',
  },
  title: {
    color: 'white',
    fontSize: 20,
    fontWeight: 'bold',
    textAlign: 'center',
    paddingTop: 50,
    paddingBottom: 10,
    backgroundColor: '#000',
  },
  statusContainer: {
    backgroundColor: 'rgba(0,0,0,0.8)',
    paddingVertical: 5,
    paddingHorizontal: 10,
  },
  statusText: {
    color: '#00ff00',
    fontSize: 12,
    textAlign: 'center',
    fontFamily: 'monospace',
  },
  message: {
    textAlign: 'center',
    paddingBottom: 10,
    color: 'white',
    fontSize: 16,
  },
  camera: {
    flex: 1,
  },
  overlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'transparent',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 50,
  },
  targetFrame: {
    width: 300,
    height: 600,
    borderWidth: 3,
    borderColor: '#00ff00',
    borderRadius: 15,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 255, 0, 0.1)',
    padding: 10,
    shadowColor: '#00ff00',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 10,
  },
  overlayText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
    textAlign: 'center',
    textShadowColor: 'rgba(0, 0, 0, 0.8)',
    textShadowOffset: { width: 2, height: 2 },
    textShadowRadius: 3,
    marginBottom: 10,
  },
  predictionText: {
    color: '#00ff00',
    fontSize: 14,
    fontWeight: 'bold',
    textAlign: 'center',
    textShadowColor: 'rgba(0, 0, 0, 0.8)',
    textShadowOffset: { width: 1, height: 1 },
    textShadowRadius: 2,
  },
  buttonContainer: {
    flexDirection: 'row',
    backgroundColor: 'transparent',
    margin: 20,
    gap: 10,
    flexWrap: 'wrap',
    justifyContent: 'center',
  },
  button: {
    backgroundColor: '#2196F3',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 25,
    marginHorizontal: 10,
  },
  flipButton: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    paddingHorizontal: 15,
    paddingVertical: 10,
    borderRadius: 25,
    borderWidth: 1,
    borderColor: 'white',
  },
  autoButton: {
    backgroundColor: 'rgba(255, 165, 0, 0.3)',
    paddingHorizontal: 15,
    paddingVertical: 10,
    borderRadius: 25,
    borderWidth: 2,
    borderColor: '#FFA500',
  },
  activeButton: {
    backgroundColor: 'rgba(255, 165, 0, 0.6)',
  },
  captureButton: {
    backgroundColor: 'rgba(0, 255, 0, 0.3)',
    paddingHorizontal: 15,
    paddingVertical: 10,
    borderRadius: 25,
    borderWidth: 2,
    borderColor: '#00ff00',
  },
  disabledButton: {
    backgroundColor: 'rgba(128, 128, 128, 0.3)',
    borderColor: 'gray',
  },
  buttonText: {
    fontSize: 14,
    fontWeight: 'bold',
    color: 'white',
    textAlign: 'center',
  },
});