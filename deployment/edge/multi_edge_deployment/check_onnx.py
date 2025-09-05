#!/usr/bin/env python3
import onnx
import os
import sys

def check_onnx_model(model_path):
    """
    Check if an ONNX model is valid
    
    Args:
        model_path: Path to the ONNX model file
    """
    try:
        print(f"Loading ONNX model from {model_path}...")
        model = onnx.load(model_path)
        
        print("Checking model structure...")
        onnx.checker.check_model(model)
        
        print("\n✅ ONNX model is valid!")
        print(f"Model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        
        # Print model metadata
        print("\nModel Metadata:")
        print(f"Producer: {model.producer_name}")
        print(f"Producer Version: {model.producer_version}")
        print(f"IR Version: {model.ir_version}")
        print(f"Domain: {model.domain}")
        
        # Print input and output information
        print("\nInputs:")
        for input in model.graph.input:
            print(f"  - {input.name}")
        
        print("\nOutputs:")
        for output in model.graph.output:
            print(f"  - {output.name}")
        
        return True
    except Exception as e:
        print(f"\n❌ Error checking ONNX model: {e}")
        return False

if __name__ == "__main__":
    # Path to your ONNX model
    model_path = "runs/detect/train/weights/best.onnx"
    
    # Check if the file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        sys.exit(1)
    
    # Check the model
    check_onnx_model(model_path) 