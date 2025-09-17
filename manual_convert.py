import tensorflow as tf
import numpy as np
import json
import os

def create_tfjs_model():
    print("Creating TensorFlow.js model structure...")
    
    # Create output directory
    os.makedirs('./assets/cards_model', exist_ok=True)
    
    # Load TFLite model to get weights (as much as we can)
    interpreter = tf.lite.Interpreter(model_path='./assets/64x3-cards.tflite')
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Model info:")
    print(f"  Input: {input_details[0]['shape']} ({input_details[0]['dtype']})")
    print(f"  Output: {output_details[0]['shape']} ({output_details[0]['dtype']})")
    
    # Create a simplified TensorFlow.js model structure
    model_json = {
        "format": "graph-model",
        "generatedBy": "manual-converter",
        "convertedBy": "TensorFlow.js Converter 4.0.0",
        "signature": {
            "inputs": {
                "input": {
                    "name": "input:0",
                    "dtype": "DT_FLOAT",
                    "tensorShape": {
                        "dim": [
                            {"size": "1"},
                            {"size": "70"},
                            {"size": "70"}, 
                            {"size": "1"}
                        ]
                    }
                }
            },
            "outputs": {
                "output": {
                    "name": "output:0",
                    "dtype": "DT_FLOAT",
                    "tensorShape": {
                        "dim": [
                            {"size": "1"},
                            {"size": "52"}
                        ]
                    }
                }
            }
        },
        "userDefinedMetadata": {
            "inputShape": "[1,70,70,1]",
            "outputShape": "[1,52]",
            "classes": "52"
        },
        "modelTopology": {
            "node": [
                {
                    "name": "input",
                    "op": "Placeholder",
                    "attr": {
                        "dtype": {"type": "DT_FLOAT"},
                        "shape": {"shape": {"dim": [{"size": "1"}, {"size": "70"}, {"size": "70"}, {"size": "1"}]}}
                    }
                },
                {
                    "name": "output",
                    "op": "Identity",
                    "input": ["dense/BiasAdd"],
                    "attr": {
                        "T": {"type": "DT_FLOAT"}
                    }
                }
            ],
            "library": {},
            "versions": {"producer": 1}
        },
        "weightsManifest": [{
            "paths": ["group1-shard1of1.bin"],
            "weights": [
                {
                    "name": "conv2d/kernel",
                    "shape": [3, 3, 1, 32],
                    "dtype": "float32"
                },
                {
                    "name": "conv2d/bias", 
                    "shape": [32],
                    "dtype": "float32"
                },
                {
                    "name": "dense/kernel",
                    "shape": [1152, 52],
                    "dtype": "float32"
                },
                {
                    "name": "dense/bias",
                    "shape": [52], 
                    "dtype": "float32"
                }
            ]
        }]
    }
    
    # Save model.json
    with open('./assets/cards_model/model.json', 'w') as f:
        json.dump(model_json, f, indent=2)
    
    # Create dummy weights (random weights for testing)
    # In a real conversion, we'd extract actual weights from TFLite
    conv_kernel = np.random.randn(3, 3, 1, 32).astype(np.float32) * 0.1
    conv_bias = np.random.randn(32).astype(np.float32) * 0.1
    dense_kernel = np.random.randn(1152, 52).astype(np.float32) * 0.1
    dense_bias = np.random.randn(52).astype(np.float32) * 0.1
    
    # Concatenate all weights into one binary file
    all_weights = np.concatenate([
        conv_kernel.flatten(),
        conv_bias.flatten(),
        dense_kernel.flatten(),
        dense_bias.flatten()
    ])
    
    # Save weights
    all_weights.astype(np.float32).tofile('./assets/cards_model/group1-shard1of1.bin')
    
    print(f"✅ Created TensorFlow.js model structure!")
    print(f"📁 Files created:")
    for file in os.listdir('./assets/cards_model/'):
        size = os.path.getsize(f'./assets/cards_model/{file}')
        print(f"  - {file} ({size} bytes)")
    
    print(f"\n⚠️  Note: This uses random weights for testing!")
    print(f"   Real predictions will be random until we extract actual weights.")

if __name__ == '__main__':
    create_tfjs_model()
