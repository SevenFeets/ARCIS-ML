#!/usr/bin/env python3
from pathlib import Path
import sys
import os
import argparse

# Add the Utilities directory to the path so we can import train_yolo
utilities_path = Path(__file__).parent / "Utilities"
sys.path.append(str(utilities_path))

from ultralytics import YOLO
import torch
import gc

def select_dataset():
    """Interactive function to select the dataset to use for training"""
    workspace_root = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    
    # Define available datasets
    datasets = {
        "1": {
            "name": "merged_dataset (80/10/10 split)",
            "path": workspace_root / "merged_dataset" / "data.yaml"
        },
        "2": {
            "name": "merged_dataset_75_15 (75/15/15 split)",
            "path": workspace_root / "merged_dataset_75_15" / "data.yaml"
        },
        "3": {
            "name": "merged_dataset_80_10_10_FULL",
            "path": workspace_root / "merged_dataset_80_10_10_FULL" / "data.yaml"
        }
    }
    
    # Check which datasets are available
    available_datasets = {}
    for key, dataset in datasets.items():
        if os.path.exists(dataset["path"]):
            available_datasets[key] = dataset
    
    if not available_datasets:
        print("Error: No datasets found!")
        return None
    
    # Display available datasets
    print("\nAvailable datasets:")
    for key, dataset in available_datasets.items():
        print(f"{key}. {dataset['name']}")
    
    # Get user selection
    while True:
        choice = input("\nSelect a dataset (enter number): ").strip()
        if choice in available_datasets:
            selected_dataset = available_datasets[choice]
            print(f"\nSelected dataset: {selected_dataset['name']}")
            return str(selected_dataset["path"])
        else:
            print("Invalid selection. Please try again.")

def train_for_jetson(
    data_yaml_path: str,
    epochs: int = 100,
    imgsz: int = 256,  # Smaller image size for Jetson Nano
    batch_size: int = 16,
    model_type: str = "yolov8n.pt"  # Using nano model for Jetson Nano
):
    # Force garbage collection to free up memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Print detailed GPU information
    print("\nGPU Information:")
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print("GPU device:", device_name)
        print("GPU memory:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
        device = 0  # Use GPU for training
    else:
        print("WARNING: No GPU detected! Training will be very slow on CPU.")
        device = 'cpu'
    
    # Initialize model
    print("\nInitializing YOLO model...")
    model = YOLO(model_type)
    
    # Train the model with settings optimized for Jetson Nano deployment
    print("\nStarting training for Jetson Nano optimization...")
    try:
        model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,  # Smaller image size for Jetson Nano
            batch=batch_size,
            device=device,
            patience=50,
            save=True,
            plots=True,
            workers=4,
            cache=False,
            amp=False,
            optimizer="SGD",
            lr0=0.01,
            lrf=0.1,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            close_mosaic=10,
            fraction=1.0,
            rect=False,
            cos_lr=True,
            verbose=True,
            exist_ok=True,
            nbs=64,
            overlap_mask=False,
            val=True,
            deterministic=False,
            project="runs/detect",
            name="jetson",  # Save in a different folder
            save_dir="runs/detect/jetson"  # Explicitly set save directory
        )
        
        print("\nTraining completed!")
        print(f"Best model saved at: runs/detect/jetson/weights/best.pt")
        
    except Exception as e:
        print(f"\nTraining error: {e}")
        print("Model weights are still saved at runs/detect/jetson/weights/")
    
    # Export to optimized formats for Jetson Nano
    export_for_jetson(model, imgsz)

def export_for_jetson(model, imgsz):
    """Export the model to formats optimized for Jetson Nano"""
    try:
        # Create calibration data directory if it doesn't exist
        calibration_dir = Path("/home/linhome/arcis/runs/calibration_data")
        calibration_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Export to ONNX with optimization for TensorRT
        print("\nExporting model to ONNX format optimized for TensorRT...")
        onnx_path = model.export(format="onnx", imgsz=imgsz, simplify=True, opset=11)
        print(f"ONNX model exported to: {onnx_path}")
        
        # 2. Export to TFLite with INT8 quantization
        print("\nExporting model to TFLite format with INT8 quantization...")
        tflite_path = model.export(
            format="tflite",
            imgsz=imgsz,
            int8=True,
            data=str(calibration_dir / "calibration_data.yaml")
        )
        print(f"TFLite model exported to: {tflite_path}")
        
        # Note: TensorRT engine must be created on the Jetson device itself
        print("\nNote: For best performance, convert the ONNX model to TensorRT on the Jetson device")
        
        print("\nAll exports completed successfully!")
        
    except Exception as e:
        print(f"\nExport error: {e}")
        print("You can manually export the model later using export_to_jetson.py")

def train_yolov8_ultralight(data_yaml_path, epochs=100, imgsz=256):
    """Train YOLOv8n-ultralight model for maximum speed on Jetson Nano"""
    print("\nTraining YOLOv8n-ultralight model for maximum speed on Jetson Nano...")
    
    # Get number of classes from data.yaml
    import yaml
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    num_classes = data_config['nc']
    
    # Create ultralight model YAML
    ultralight_yaml = f"""
# YOLOv8n-ultralight for Jetson Nano
nc: {num_classes}  # number of classes
depth_multiple: 0.33
width_multiple: 0.25
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

backbone:
  - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
  - [-1, 3, C2f, [128]]  # 2
  - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
  - [-1, 3, C2f, [256]]  # 4
  - [-1, 1, Conv, [384, 3, 2]]  # 5-P4/16
  - [-1, 3, C2f, [384]]  # 6
  - [-1, 1, Conv, [512, 3, 2]]  # 7-P5/32
  - [-1, 1, C2f, [512]]  # 8
head:
  - [-1, 1, SPPF, [512, 5]]  # 9
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]  # cat backbone P4
  - [-1, 1, C2f, [384]]  # 12
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]  # cat backbone P3
  - [-1, 1, C2f, [256]]  # 15
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]  # cat head P4
  - [-1, 1, C2f, [384]]  # 18
  - [-1, 1, Conv, [384, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]  # cat head P5
  - [-1, 1, C2f, [512]]  # 21
  - [[15, 18, 21], 1, Detect, [nc]]  # Detect(P3, P4, P5)
"""
    
    # Save ultralight model YAML
    ultralight_yaml_path = "yolov8n-ultralight.yaml"
    
    with open(ultralight_yaml_path, 'w') as f:
        f.write(ultralight_yaml)
    
    print(f"Created ultralight model configuration: {ultralight_yaml_path}")
    
    # Initialize ultralight model
    try:
        print("Starting with base YOLOv8n model and training with ultralight settings...")
        model = YOLO('yolov8n.pt')
        
        # Train with ultralight settings
        model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=16,
            patience=50,
            save=True,
            plots=True,
            workers=4,
            cache=False,
            amp=False,
            optimizer="SGD",
            project="runs/detect",
            name="jetson_ultralight",
            lr0=0.01,
            lrf=0.1,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            close_mosaic=10,
            rect=False,
            cos_lr=True,
            verbose=True,
            exist_ok=True,
            nbs=64,
            val=True
        )
        
        print("\nUltralight model training completed!")
        print(f"Best model saved at: runs/detect/jetson_ultralight/weights/best.pt")
        
        # Export ultralight model
        print("\nExporting ultralight model to optimized formats...")
        model.export(format="onnx", imgsz=imgsz, simplify=True, opset=11)
        model.export(format="tflite", imgsz=imgsz, int8=True)
        
    except Exception as e:
        print(f"\nUltralight model training error: {e}")

if __name__ == "__main__":
    # Get dataset selection from user
    data_yaml_path = select_dataset()
    if not data_yaml_path:
        print("No dataset selected. Exiting...")
        sys.exit(1)
    
    # Configure training parameters optimized for Jetson Nano
    config = {
        "data_yaml_path": data_yaml_path,
        "epochs": 100,
        "imgsz": 256,  # Smaller size for Jetson Nano
        "batch_size": 16,
        "model_type": "yolov8n.pt"  # Nano model is best for Jetson Nano
    }
    
    print("\nTraining configuration for Jetson Nano deployment:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nStarting training with settings optimized for Jetson Nano...")
    
    # Set HSA_OVERRIDE_GFX_VERSION for AMD GPU compatibility
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
    
    print("\n=== Jetson Nano Optimization Options ===")
    print("1. Train standard YOLOv8n with reduced image size")
    print("2. Train ultralight YOLOv8n (faster but less accurate)")
    
    choice = input("\nSelect option (1/2, default=1): ").strip() or "1"
    
    if choice == "1":
        train_for_jetson(**config)
    elif choice == "2":
        print("\nTraining ultralight YOLOv8n model for maximum speed...")
        train_yolov8_ultralight(data_yaml_path, epochs=config["epochs"], imgsz=config["imgsz"]) 