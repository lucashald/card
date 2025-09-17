import React, { useState, useEffect, useRef } from 'react';
import { Text, View, StyleSheet, TouchableOpacity, Alert } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import { bundleResourceIO, decodeJpeg } from '@tensorflow/tfjs-react-native';
import { Buffer } from 'buffer';
import { getInfoAsync, documentDirectory } from 'expo-file-system/legacy';
import { Asset } from 'expo-asset';

export default function App() {
  const cameraRef = useRef(null);
  const [facing, setFacing] = useState('back');
  const [permission, requestPermission] = useCameraPermissions();
  const [cameraReady, setCameraReady] = useState(false);
  const [model, setModel] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [lastPrediction, setLastPrediction] = useState(null);
  const [tfReady, setTfReady] = useState(false);

  // Initialize TensorFlow
  useEffect(() => {
    const initTensorFlow = async () => {
      try {
        console.log('Initializing TensorFlow.js...');
        await tf.ready();
        console.log('TensorFlow.js is ready!');
        console.log('Backend:', tf.getBackend());
        setTfReady(true);
        
        // Load the model
        await loadModel();
      } catch (error) {
        console.error('Error initializing TensorFlow:', error);
        Alert.alert('Error', 'Failed to initialize TensorFlow');
      }
    };
    
    initTensorFlow();
  }, []);

  // Load the playing cards model
  const loadModel = async () => {
    try {
      console.log('Loading playing cards model...');
      
      // Try loading the converted TensorFlow.js model first
      try {
        console.log('Attempting to load TensorFlow.js model...');
        
        // Load the converted TensorFlow.js model
        // Prefer multi-shard artifacts produced by the official converter
        const weightsOptions = [
          [
            require('./assets/cards_model/group1-shard1of2.bin'),
            require('./assets/cards_model/group1-shard2of2.bin'),
          ],
          // Fallback to a single-shard artifact if present
          [require('./assets/cards_model/group1-shard1of1.bin')],
        ];

        let loadedModel = null;
        let lastErr = null;
        // Try loading as a LayersModel first (format: "layers-model"), then GraphModel
        for (const weights of weightsOptions) {
          const ioHandler = bundleResourceIO(
            require('./assets/cards_model/model.json'),
            weights
          );
          // Try Layers
          try {
            loadedModel = await tf.loadLayersModel(ioHandler);
            console.log('Loaded TFJS LayersModel.');
            break;
          } catch (layersErr) {
            lastErr = layersErr;
            console.warn('LayersModel load failed, will try GraphModel:', layersErr?.message);
          }
          // Try Graph
          try {
            loadedModel = await tf.loadGraphModel(ioHandler);
            console.log('Loaded TFJS GraphModel.');
            break;
          } catch (graphErr) {
            lastErr = graphErr;
            console.warn('GraphModel load failed with this weights set:', graphErr?.message);
          }
        }
        if (!loadedModel) {
          throw lastErr || new Error('Unable to load TFJS model with provided weights');
        }
        
        console.log('✅ TensorFlow.js model loaded successfully!');
        // Determine model type and input shape
        let modelType = 'tfjs';
        let inputShape = undefined;
        try {
          // LayersModel exposes model.inputShape; GraphModel has inputs array
          if (loadedModel && 'predict' in loadedModel && 'layers' in loadedModel) {
            modelType = 'tfjs-layers';
            inputShape = loadedModel.inputs?.[0]?.shape || loadedModel.inputShape;
            console.log('Detected LayersModel with input shape:', inputShape);
          } else {
            modelType = 'tfjs-graph';
            if (!loadedModel.inputs?.length || !loadedModel.outputs?.length) {
              throw new Error('Loaded TFJS GraphModel has no inputs/outputs. The artifacts may be invalid.');
            }
            console.log('Model inputs:', loadedModel.inputs.map(input => ({
              name: input.name,
              shape: input.shape,
              dtype: input.dtype
            })));
            console.log('Model outputs:', loadedModel.outputs.map(output => ({
              name: output.name, 
              shape: output.shape,
              dtype: output.dtype
            })));
            inputShape = loadedModel.inputs[0].shape;
          }
        } catch (logErr) {
          console.warn('Could not log model IO metadata:', logErr?.message);
        }

        setModel({ 
          model: loadedModel,
          loaded: true, 
          type: modelType,
          inputShape: inputShape
        });
        return;
        
      } catch (tfjsError) {
        console.error('TensorFlow.js model loading failed:', tfjsError.message);
        // Note: Expo Go cannot run native TFLite; skip attempting TFLite here to avoid noisy errors.
        throw new Error(
          'TensorFlow.js model failed to load. Ensure model.json is a valid TFJS model (Layers or Graph) converted with the official converter. TFLite is not available in Expo Go.'
        );
      }
      
    } catch (error) {
      console.error('All model loading attempts failed:', error.message);
      Alert.alert('Model Error', `Failed to load model: ${error.message}`);
      // Do not enable simulated mode; keep model unavailable
      setModel(null);
    }
  };

  // Card class names (based on typical playing card datasets)
  const cardClasses = [
    'ace of clubs', 'ace of diamonds', 'ace of hearts', 'ace of spades',
    'two of clubs', 'two of diamonds', 'two of hearts', 'two of spades',
    'three of clubs', 'three of diamonds', 'three of hearts', 'three of spades',
    'four of clubs', 'four of diamonds', 'four of hearts', 'four of spades',
    'five of clubs', 'five of diamonds', 'five of hearts', 'five of spades',
    'six of clubs', 'six of diamonds', 'six of hearts', 'six of spades',
    'seven of clubs', 'seven of diamonds', 'seven of hearts', 'seven of spades',
    'eight of clubs', 'eight of diamonds', 'eight of hearts', 'eight of spades',
    'nine of clubs', 'nine of diamonds', 'nine of hearts', 'nine of spades',
    'ten of clubs', 'ten of diamonds', 'ten of hearts', 'ten of spades',
    'jack of clubs', 'jack of diamonds', 'jack of hearts', 'jack of spades',
    'queen of clubs', 'queen of diamonds', 'queen of hearts', 'queen of spades',
    'king of clubs', 'king of diamonds', 'king of hearts', 'king of spades'
  ];

  // Model expects 70x70 grayscale images, outputs 52 classes
  const MODEL_INPUT_SHAPE = [1, 70, 70, 1];
  const MODEL_OUTPUT_CLASSES = 52;

  // Preprocess a base64 image into [1, H, W, 1] float32 tensor in [0,1]
  const preprocessBase64Image = (base64, targetH, targetW) => {
    try {
      const bytes = Uint8Array.from(Buffer.from(base64, 'base64'));
      const img = decodeJpeg(bytes, 3); // [H,W,3] uint8
      const result = tf.tidy(() => {
        const rgb = img.toFloat();
        const [r, g, b] = tf.split(rgb, 3, 2);
        const y = r.mul(0.299).add(g.mul(0.587)).add(b.mul(0.114)); // [H,W,1]
        const resized = tf.image.resizeBilinear(y, [targetH, targetW], true);
        const norm = resized.div(255.0); // [0,1]
        return norm.expandDims(0); // [1,H,W,1]
      });
      img.dispose();
      return result;
    } catch (e) {
      console.warn('Failed to preprocess image, falling back to dummy input:', e?.message);
      return null;
    }
  };

  // Predict card from image
  const predictCard = async (imageBase64) => {
    if (!model || isProcessing) return;
    
    setIsProcessing(true);
    
    try {
  // Real TensorFlow.js model prediction (supports layers and graph)
      if (model.type?.startsWith('tfjs') && model.model) {
        console.log('Running TensorFlow.js model prediction...');
        
        // Create input tensor using model-declared shape if available
        const declared = model.inputShape || MODEL_INPUT_SHAPE;
        const inputShape = declared.map((d) => (d == null || d <= 0 ? 1 : d));
        console.log('Expected input shape:', inputShape);
        
        // Build input tensor from camera (base64) if provided, else dummy tensor
        let inputData = null;
        if (!imageBase64) {
          throw new Error('No image provided for prediction');
        }
        inputData = preprocessBase64Image(imageBase64, inputShape[1], inputShape[2]);
        if (!inputData) {
          throw new Error('Image preprocessing failed');
        }
        
        // Run prediction
        let prediction = await model.model.predict(inputData);
        if (Array.isArray(prediction)) {
          prediction = prediction[0];
        }
        // Ensure we have a Tensor
        if (!prediction || typeof prediction.data !== 'function') {
          throw new Error('Model prediction did not return a tensor');
        }
        const probabilities = await prediction.data();
        
        // Get the top prediction
        const maxIndex = Array.from(probabilities).indexOf(Math.max(...probabilities));
        const confidence = (probabilities[maxIndex] * 100).toFixed(2);
        
        const result = {
          card: cardClasses[maxIndex] || `Card ${maxIndex}`,
          confidence: confidence
        };
        
        console.log('TensorFlow.js model prediction:', result);
        console.log('Top 3 predictions:');
        const sortedIndices = Array.from(probabilities)
          .map((prob, index) => ({ prob, index }))
          .sort((a, b) => b.prob - a.prob)
          .slice(0, 3);
        
        sortedIndices.forEach(({ prob, index }, i) => {
          console.log(`  ${i + 1}. ${cardClasses[index]}: ${(prob * 100).toFixed(1)}%`);
        });
        
        setLastPrediction(result);
        triggerCardDetection(result);
        
        // Clean up tensors
        inputData.dispose();
        prediction.dispose();
        
      } else if (model.type === 'tflite' && model.interpreter) {
        console.log('Running TFLite model prediction...');
        
        // TFLite prediction code (if we ever get it working)
        const inputData = tf.randomNormal(MODEL_INPUT_SHAPE);
        model.interpreter.setInput(0, inputData);
        await model.interpreter.invoke();
        const outputTensor = model.interpreter.getOutput(0);
        const probabilities = await outputTensor.data();
        
        const maxIndex = Array.from(probabilities).indexOf(Math.max(...probabilities));
        const confidence = (probabilities[maxIndex] * 100).toFixed(2);
        
        const result = {
          card: cardClasses[maxIndex] || `Card ${maxIndex}`,
          confidence: confidence
        };
        
        console.log('TFLite model prediction:', result);
        setLastPrediction(result);
        triggerCardDetection(result);
        
        inputData.dispose();
        outputTensor.dispose();
        
      } else {
        console.log('Model type not supported:', model.type);
        throw new Error('Unsupported model type');
      }
      
    } catch (error) {
      console.error('Error predicting card:', error);
      Alert.alert('Prediction Error', `Failed to analyze image: ${error.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  // Simulate push notification for magic trick
  const triggerCardDetection = (prediction) => {
    console.log(`🎴 CARD DETECTED: ${prediction.card} (${prediction.confidence}% confidence)`);
    
    // In a real app, this would trigger a push notification
    Alert.alert(
      '🎴 Card Detected!', 
      `${prediction.card}\nConfidence: ${prediction.confidence}%`,
      [{ text: 'OK', style: 'default' }]
    );
  };

  // Handle camera ready
  const handleCameraReady = () => {
    setCameraReady(true);
    console.log('Camera is ready for card detection!');
  };

  // Capture and analyze frame
  const captureAndAnalyze = async () => {
    if (!cameraReady || !model || isProcessing) {
      console.log('Not ready for capture:', { cameraReady, model: !!model, isProcessing });
      return;
    }

    try {
      if (!cameraRef.current || !cameraRef.current.takePictureAsync) {
        console.warn('Camera ref not ready');
        Alert.alert('Camera Error', 'Camera not ready to capture a photo.');
        return;
      }

      const photo = await cameraRef.current.takePictureAsync({
        base64: true,
        quality: 0.5,
        skipProcessing: true,
      });

      if (!photo?.base64) {
        console.warn('No base64 from camera');
        Alert.alert('Capture Error', 'Failed to obtain image data from camera.');
        return;
      }
      await predictCard(photo.base64);
      
    } catch (error) {
      console.error('Error capturing image:', error);
      Alert.alert('Error', 'Failed to capture image');
    }
  };

  // Toggle camera facing
  function toggleCameraFacing() {
    setFacing(current => (current === 'back' ? 'front' : 'back'));
  }

  // Permission handling
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
      
      {/* Status indicators */}
      <View style={styles.statusContainer}>
        <Text style={styles.statusText}>
          TF: {tfReady ? '✅' : '⏳'} | 
          Model: {model ? '✅' : '⏳'} | 
          Camera: {cameraReady ? '✅' : '⏳'}
        </Text>
      </View>
      
      <CameraView 
        ref={cameraRef}
        style={styles.camera} 
        facing={facing}
        onCameraReady={handleCameraReady}
      />
      
      {/* Overlay positioned absolutely on top of camera */}
      <View style={styles.overlay}>
        {/* Target frame for card detection */}
        <View style={styles.targetFrame}>
          <Text style={styles.overlayText}>
            {!cameraReady ? 'Starting camera...' :
             !model ? 'Loading model...' :
             isProcessing ? 'Processing...' :
             'Point at playing cards'}
          </Text>
          
          {lastPrediction && (
            <Text style={styles.predictionText}>
              Last: {lastPrediction.card} ({lastPrediction.confidence}%)
            </Text>
          )}
        </View>
        
        {/* Control buttons */}
        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.flipButton} onPress={toggleCameraFacing}>
            <Text style={styles.buttonText}>Flip Camera</Text>
          </TouchableOpacity>
          
          <TouchableOpacity 
            style={[styles.captureButton, isProcessing && styles.disabledButton]} 
            onPress={captureAndAnalyze}
            disabled={!cameraReady || !model || isProcessing}
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
    height: 200,
    borderWidth: 3,
    borderColor: '#00ff00',
    borderRadius: 15,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
    padding: 10,
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