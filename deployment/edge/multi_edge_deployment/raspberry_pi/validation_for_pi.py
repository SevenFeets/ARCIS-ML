#!/usr/bin/env python3
from pathlib import Path
import sys
import os

# Set AMD GPU compatibility environment variable at the very beginning
# This is crucial for AMD GPUs like the RX 6700 XT
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

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

def select_device():
    """Allow the user to select the device for validation"""
    print("\nSelect device for validation:")
    print("1. CPU (slower but more stable)")
    print("2. GPU - AMD ROCm (faster but may require tuning)")
    
    while True:
        choice = input("\nSelect device (enter number, default=2): ").strip() or "2"
        if choice == "1":
            return "cpu", False  # device, half_precision
        elif choice == "2":
            # For AMD GPUs, check if we can detect it
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                print(f"\nDetected GPU: {device_name}")
                if "AMD" in device_name:
                    print("AMD GPU detected with ROCm support.")
                    print("Using recommended settings from AMD_GPU_TRAINING.md")
                    # HSA_OVERRIDE_GFX_VERSION is already set at the top of the file
            return 0, False  # Use GPU but disable half precision for AMD GPUs
        else:
            print("Invalid selection. Please try again.")

def validate_model_performance(model_path: str, data_yaml_path: str, imgsz: int = 320):
    """Validate the performance of a trained YOLOv8 model."""
    print(f"\nLoading model from: {model_path}")
    
    # Force garbage collection to free up memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Let user select device
    device_for_val, half_precision_for_val = select_device()
    
    if device_for_val == "cpu":
        print(f"\nValidating on CPU with FP32...")
        batch_size = 4  # Smaller batch size for CPU
        workers = 0     # No workers for CPU
    else:
        print(f"\nValidating on GPU with FP32 (amp disabled for AMD compatibility)...")
        batch_size = 8  # Recommended in documentation for AMD GPUs
        workers = 2     # Recommended in documentation for AMD GPUs
    
    # Load model
    model = YOLO(model_path)
    
    print(f"\nStarting validation using dataset: {data_yaml_path}")
    try:
        metrics = model.val(
            data=data_yaml_path,
            imgsz=imgsz,
            split='val',           # Explicitly validate on the validation set
            save_json=True,        # Save metrics to a JSON file
            save_hybrid=False,     # Disable saving predictions in YOLO format for stability
            conf=0.001,            # Object confidence threshold for plotting
            iou=0.6,               # IoU threshold for NMS
            half=half_precision_for_val,  # Use FP32 for AMD stability
            plots=True,            # Save plots
            workers=workers,       # Adjusted workers based on device
            project="runs/detect",
            name=f"validation_{Path(model_path).parent.parent.name}", # Save in a validation specific folder
            exist_ok=True,         # Allow existing folder
            device=device_for_val, # Use selected device
            batch=batch_size,      # Adjusted batch size per documentation
            verbose=False          # Reduce console output for stability
        )
        print("\nValidation completed!")
        print(f"Metrics: {metrics.results_dict}")
    except Exception as e:
        print(f"\nValidation error: {e}")
        print("\nIf you encounter GPU-related errors, try these troubleshooting steps:")
        print("1. Run with CPU instead")
        print("2. Reduce batch size further (--batch 4)")
        print("3. Check your ROCm version compatibility")
        print("4. See Documentation/AMD_GPU_TRAINING.md for more troubleshooting tips")

if __name__ == "__main__":
    print("\n=== YOLOv8 Model Validation for Raspberry Pi System ===")
    print("Using AMD GPU compatibility settings from documentation")

    # Get model path from user
    model_to_validate = input("\nEnter the path to the model you want to validate (e.g., runs/detect/your_run_name/weights/best.pt): ").strip()
    if not os.path.exists(model_to_validate):
        print(f"Error: Model file not found at {model_to_validate}")
        sys.exit(1)

    # Get dataset for validation
    data_yaml_path = select_dataset()
    if not data_yaml_path:
        print("No dataset selected for validation. Exiting...")
        sys.exit(1)

    # Use the image size from edge training documentation
    validation_imgsz = 320  # Recommended size for Raspberry Pi models

    validate_model_performance(model_to_validate, data_yaml_path, imgsz=validation_imgsz)

    print("\nValidation script finished.") 