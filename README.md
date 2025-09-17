# Card App — Model Loading Notes

This app loads a TensorFlow.js GraphModel from bundled assets. It also detects cards using the device camera.

## Key points

- Expo Go supports TensorFlow.js via `@tensorflow/tfjs-react-native` (WebGL backend) but does NOT support native TFLite.
- The included `assets/cards_model` must contain a valid TFJS GraphModel converted with the official converter.
- The previously generated `model.json` was a placeholder and will fail at runtime (no valid inputs/outputs).

## Provide a valid TFJS model

Use the official converter to convert your original TF/keras model to TFJS graph model artifacts.

Example (from a saved model):

```
# Install converter
pip install tensorflowjs

# Convert SavedModel to TFJS GraphModel
tensorflowjs_converter \
  --input_format=tf_saved_model \
  --output_format=tfjs_graph_model \
  --signature_name=serving_default \
  --saved_model_tags=serve \
  path/to/saved_model \
  ./assets/cards_model
```

Or from a Keras `.h5` model:

```
tensorflowjs_converter \
  --input_format=keras \
  path/to/model.h5 \
  ./assets/cards_model
```

Ensure the produced files are:
- `assets/cards_model/model.json`
- `assets/cards_model/group1-shard1of1.bin` (or multiple shards)

The `model.json` should include proper `signature` or at least valid `modelTopology` with nodes and the weights manifest.

## Loading in-app

The app uses:

- `bundleResourceIO(require('./assets/cards_model/model.json'), require('./assets/cards_model/group1-shard1of1.bin'))`
- `tf.loadGraphModel(modelUrl)`

If loading succeeds, you’ll see logs for inputs/outputs. If it fails, the app enters a Simulation mode for development.

## About TFLite

- TFLite is not available in Expo Go.
- If you need TFLite, create a custom development client or EAS build with a TFLite-native module and wire it up. Until then, the app skips TFLite to reduce noise.

## Common pitfalls

- Placeholder `model.json` without real topology/weights → results in `Cannot read property 'outputs' of undefined`.
- Wrong path or missing asset bundling → ensure `app.json` includes `assetBundlePatterns: ["assets/**/*"]` (already present).
- Backend fallback → On iOS, ensure `expo-gl` is installed (already in package.json). The logs should show `Backend: rn-webgl`.

## Quick smoke test

After replacing the model artifacts, run the app and tap "Detect Card". The app currently uses a dummy tensor for inference while camera preprocessing is being implemented.