#!/usr/bin/env python3
from pathlib import Path
import sys
import os

# Set the HSA_OVERRIDE_GFX_VERSION environment variable
# This helps with AMD GPU compatibility
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

# Add the Utilities directory to the path so we can import train_yolo
utilities_path = Path(__file__).parent / "Utilities"
sys.path.append(str(utilities_path))

# Import the train_yolo function from the Utilities directory
from Utilities.train_yolo import train_yolo

if __name__ == "__main__":
    # Path to the data.yaml file
    data_yaml = str(Path(__file__).parent / "database" / "data.yaml")
    
    # Check if the data.yaml file exists
    if not os.path.exists(data_yaml):
        print(f"Error: Data file {data_yaml} not found!")
        sys.exit(1)
    
    print(f"Using data configuration from: {data_yaml}")
    
    # Configure training parameters - optimized for 12GB VRAM
    config = {
        "data_yaml_path": data_yaml,
        "epochs": 100,              # Total training epochs
        "imgsz": 640,               # Full resolution images for better accuracy
        "batch_size": 32,           # Increased batch size for 12GB VRAM
        "model_type": "yolov8n.pt"  # Using nano model for better performance
    }
    
    print("\nTraining configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nStarting training with ROCm on AMD GPU...")
    print("Using HSA_OVERRIDE_GFX_VERSION=10.3.0 for compatibility")
    print("Using full dataset and optimized settings for 12GB VRAM")
    
    train_yolo(**config) 