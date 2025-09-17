import React, { useState, useEffect, useRef } from 'react';
import { Text, View, StyleSheet, TouchableOpacity, Alert } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { Buffer } from 'buffer';
import { GoogleGenerativeAI } from '@google/generative-ai';

const GEMINI_API_KEY = 'add your Gemini API key here';

export default function App() {
  const cameraRef = useRef(null);
  const [facing, setFacing] = useState('back');
  const [permission, requestPermission] = useCameraPermissions();
  const [cameraReady, setCameraReady] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [lastPrediction, setLastPrediction] = useState(null);
  const [geminiReady, setGeminiReady] = useState(false);
  const genAI = useRef(null);
  const model = useRef(null);

  useEffect(() => {
    try {
      if (!GEMINI_API_KEY || GEMINI_API_KEY === 'YOUR_GEMINI_API_KEY') {
        throw new Error('Gemini API key is not set. Please add your key.');
      }
      genAI.current = new GoogleGenerativeAI(GEMINI_API_KEY);
      model.current = genAI.current.getGenerativeModel({ model: 'gemini-2.5-flash-lite' });
      setGeminiReady(true);
    } catch (error) {
      console.error('Error initializing Gemini API:', error);
      Alert.alert('API Error', 'Failed to initialize Gemini API. Check your API key.');
    }
  }, []);

  const fileToGenerativePart = (base64String) => {
    return {
      inlineData: {
        data: base64String,
        mimeType: 'image/jpeg',
      },
    };
  };

  const predictCard = async (imageBase64) => {
    if (!geminiReady || isProcessing) return;

    setIsProcessing(true);

    try {
      const imagePart = fileToGenerativePart(imageBase64);
      const prompt = "What playing card is visible in this image? Respond only with the card's name, like 'Ace of Spades' or 'Jack of Hearts'. If no playing card is clearly visible, respond with 'Not a card'.";
      const result = await model.current.generateContent([prompt, imagePart]);
      const response = await result.response;
      const text = response.text().trim();

      const resultObject = {
        card: text,
        confidence: text === 'Not a card' ? 'N/A' : 'High',
      };

      setLastPrediction(resultObject);
      triggerCardDetection(resultObject);

    } catch (error) {
      console.error('Error predicting card with Gemini:', error);
      Alert.alert('Prediction Error', 'Failed to analyze image with Gemini.');
    } finally {
      setIsProcessing(false);
    }
  };

  const triggerCardDetection = (prediction) => {
    Alert.alert(
      '🎴 Card Detected!',
      `${prediction.card}\nConfidence: ${prediction.confidence}`,
      [{ text: 'OK', style: 'default' }]
    );
  };

  const handleCameraReady = () => {
    setCameraReady(true);
  };

  const captureAndAnalyze = async () => {
    if (!cameraReady || !geminiReady || isProcessing) {
      return;
    }

    try {
      if (!cameraRef.current) {
        Alert.alert('Camera Error', 'Camera not ready to capture a photo.');
        return;
      }

      const photo = await cameraRef.current.takePictureAsync({
        base64: true,
        quality: 1.0,
      });

      if (!photo?.base64) {
        Alert.alert('Capture Error', 'Failed to obtain image data from camera.');
        return;
      }
      await predictCard(photo.base64);

    } catch (error) {
      console.error('Error capturing image:', error);
      Alert.alert('Error', 'Failed to capture image');
    }
  };

  function toggleCameraFacing() {
    setFacing(current => (current === 'back' ? 'front' : 'back'));
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
      <Text style={styles.title}>Card Detection Camera</Text>

      <View style={styles.statusContainer}>
        <Text style={styles.statusText}>
          API: {geminiReady ? '✅' : '⏳'} |
          Camera: {cameraReady ? '✅' : '⏳'}
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
             !geminiReady ? 'Connecting to API...' :
             isProcessing ? 'Processing image...' :
             'Point cards in green frame'}
          </Text>

          {lastPrediction && (
            <Text style={styles.predictionText}>
              Last: {lastPrediction.card} ({lastPrediction.confidence})
            </Text>
          )}
        </View>

        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.flipButton} onPress={toggleCameraFacing}>
            <Text style={styles.buttonText}>Flip Camera</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.captureButton, isProcessing && styles.disabledButton]}
            onPress={captureAndAnalyze}
            disabled={!cameraReady || !geminiReady || isProcessing}
          >
            <Text style={styles.buttonText}>
              {isProcessing ? 'Processing...' : 'Detect Card'}
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
    gap: 15,
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
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 25,
    borderWidth: 1,
    borderColor: 'white',
  },
  captureButton: {
    backgroundColor: 'rgba(0, 255, 0, 0.3)',
    paddingHorizontal: 20,
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
    fontSize: 16,
    fontWeight: 'bold',
    color: 'white',
    textAlign: 'center',
  },
});