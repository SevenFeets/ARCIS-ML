#!/usr/bin/env python3
from pathlib import Path
import sys
import os

# Set the HSA_OVERRIDE_GFX_VERSION environment variable
# This helps with AMD GPU compatibility
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

# Add additional environment variables that might help with ROCm
os.environ["GPU_MAX_HEAP_SIZE"] = "100"
os.environ["GPU_MAX_ALLOC_PERCENT"] = "100"
os.environ["GPU_SINGLE_ALLOC_PERCENT"] = "100"

# Add the Utilities directory to the path so we can import train_yolo
utilities_path = Path(__file__).parent / "Utilities"
sys.path.append(str(utilities_path))

from Utilities.train_yolo import train_yolo  # type: ignore

if __name__ == "__main__":
    # Path to the data.yaml file
    data_yaml = str(Path(__file__).parent / "database" / "data.yaml")
    
    # Check if the data.yaml file exists
    if not os.path.exists(data_yaml):
        print(f"Error: Data file {data_yaml} not found!")
        sys.exit(1)
    
    print(f"Using data configuration from: {data_yaml}")
    
    # Configure training parameters - smaller values for better compatibility
    config = {
        "data_yaml_path": data_yaml,
        "epochs": 50,               # Reduced epochs
        "imgsz": 320,               # Further reduced image size for better compatibility
        "batch_size": 4,            # Smaller batch size for better compatibility
        "model_type": "yolov8n.pt"  # Using nano model for better performance
    }
    
    print("\nTraining configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nStarting training with ROCm on AMD GPU...")
    print("Using HSA_OVERRIDE_GFX_VERSION=10.3.0 for compatibility")
    print("Added additional ROCm environment variables for optimization")
    
    # Import torch here to ensure environment variables are set before torch is loaded
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    
    train_yolo(**config) 