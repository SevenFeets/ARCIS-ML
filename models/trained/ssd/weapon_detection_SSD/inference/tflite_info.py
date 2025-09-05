#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
import argparse
from pathlib import Path

def load_tflite_model(tflite_path):
    """Load TFLite model and print its details"""
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("\n===== TFLite Model Information =====")
    print(f"Model path: {tflite_path}")
    print(f"Model size: {os.path.getsize(tflite_path) / (1024 * 1024):.2f} MB")
    
    print("\n----- Input Details -----")
    for i, input_detail in enumerate(input_details):
        print(f"Input {i}:")
        print(f"  Name: {input_detail['name']}")
        print(f"  Shape: {input_detail['shape']}")
        print(f"  Type: {input_detail['dtype']}")
        print(f"  Quantization: {input_detail.get('quantization', 'None')}")
    
    print("\n----- Output Details -----")
    for i, output_detail in enumerate(output_details):
        print(f"Output {i}:")
        print(f"  Name: {output_detail['name']}")
        print(f"  Shape: {output_detail['shape']}")
        print(f"  Type: {output_detail['dtype']}")
        print(f"  Quantization: {output_detail.get('quantization', 'None')}")
    
    # Get all tensor details
    tensor_details = interpreter.get_tensor_details()
    
    print(f"\nTotal number of tensors: {len(tensor_details)}")
    print("Top 5 tensors:")
    for i, tensor in enumerate(tensor_details[:5]):
        print(f"  Tensor {i}: {tensor['name']} (shape: {tensor['shape']}, type: {tensor['dtype']})")
    
    # Test the model with a dummy input
    print("\n----- Model Test -----")
    try:
        # Get input shape
        input_shape = input_details[0]['shape']
        
        # Create a dummy input tensor
        dummy_input = np.zeros(input_shape, dtype=np.float32)
        
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        
        # Run inference
        interpreter.invoke()
        
        # Get the output tensors
        outputs = []
        for output_detail in output_details:
            output = interpreter.get_tensor(output_detail['index'])
            outputs.append(output)
            
        print("Model successfully ran inference on dummy input")
        for i, output in enumerate(outputs):
            print(f"  Output {i} shape: {output.shape}, dtype: {output.dtype}")
    
    except Exception as e:
        print(f"Error testing model: {e}")
    
    return interpreter

def main():
    parser = argparse.ArgumentParser(description='TFLite Model Information')
    parser.add_argument('--model', default='runs/ssd_small/models/model.tflite', help='Path to the TFLite model file')
    parser.add_argument('--quantized', action='store_true', help='Use quantized model')
    args = parser.parse_args()
    
    # Use quantized model if specified
    if args.quantized:
        model_path = Path(args.model).parent / "model_quantized.tflite"
    else:
        model_path = args.model
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    # Load and analyze the model
    interpreter = load_tflite_model(model_path)

if __name__ == '__main__':
    main() 