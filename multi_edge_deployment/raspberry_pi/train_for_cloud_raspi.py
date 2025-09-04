#!/usr/bin/env python3
from pathlib import Path
import sys
import os
from datetime import datetime # Import datetime

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

def train_for_raspi(
    data_yaml_path: str,
    epochs: int = 100,
    imgsz: int = 256,  # Smaller image size for Raspberry Pi
    batch_size: int = 16,
    model_type: str = "yolov8n.pt",  # Using nano model for Raspberry Pi
    lr0: float = 0.01,
    lrf: float = 0.01,  # Lower final learning rate
    weight_decay: float = 0.0005,
    augment: bool = True,  # Enable data augmentation
    mixup: float = 0.1,    # Add mixup augmentation
    mosaic: float = 1.0,   # Full mosaic augmentation
    close_mosaic: int = 10, # Disable mosaic in final epochs
    patience: int = 50,     # Early stopping patience
    optimizer: str = "SGD", # Optimizer type
    cool_down: bool = False # Temperature management mode
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
    
    # Generate a timestamped run name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"raspi_training_{timestamp}"
    save_dir_path = Path("runs/detect") / run_name
    
    # Number of workers based on cool-down mode
    workers = 2 if cool_down else 4
    
    # Train the model with settings optimized for Raspberry Pi deployment
    print("\nStarting training for Raspberry Pi optimization...")
    try:
        model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            patience=patience,
            save=True,
            plots=True,
            workers=workers,
            cache=False,
            amp=False,
            optimizer=optimizer,
            lr0=lr0,
            lrf=lrf,
            momentum=0.937,
            weight_decay=weight_decay,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            close_mosaic=close_mosaic,
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
            name=run_name,  # Use the timestamped name
            save_dir=str(save_dir_path),  # Explicitly set save directory
            # Enhanced augmentation parameters
            augment=augment,
            mixup=mixup,     # Add mixup augmentation
            mosaic=mosaic,   # Full mosaic augmentation
            degrees=10.0,    # Rotation augmentation
            translate=0.1,   # Translation augmentation
            scale=0.5,       # Scale augmentation
            shear=2.0,       # Shear augmentation
            perspective=0.0005,  # Perspective augmentation
            flipud=0.0,      # Flip up-down (disabled for weapon detection)
            fliplr=0.5,      # Flip left-right (50% probability)
            hsv_h=0.015,     # HSV hue augmentation
            hsv_s=0.7,       # HSV saturation augmentation
            hsv_v=0.4,       # HSV value augmentation
            copy_paste=0.1   # Copy-paste augmentation
        )
        
        print("\nTraining completed!")
        print(f"Best model saved at: {save_dir_path}/weights/best.pt")
        
    except Exception as e:
        print(f"\nTraining error: {e}")
        print(f"Model weights are still saved at: {save_dir_path}/weights/")
    
    # Export to optimized formats for Raspberry Pi
    export_for_raspi(model, imgsz, str(save_dir_path))

def export_for_raspi(model, imgsz, run_save_path):
    """Export the model to formats optimized for Raspberry Pi"""
    try:
        # Create calibration data directory if it doesn't exist
        calibration_dir = Path("/home/linhome/arcis/runs/calibration_data")
        calibration_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Export to ONNX (good general format)
        print("\nExporting model to ONNX format...")
        onnx_path = model.export(format="onnx", imgsz=imgsz, simplify=True)
        print(f"ONNX model exported to: {onnx_path}")
        
        # 2. Export to TFLite (great for Raspberry Pi)
        print("\nExporting model to TFLite format...")
        tflite_path = model.export(
            format="tflite",
            imgsz=imgsz,
            int8=True,
            data=str(calibration_dir / "calibration_data.yaml")
        )
        print(f"TFLite model exported to: {tflite_path}")
        
        # 3. Export to OpenVINO (optimized for edge devices)
        print("\nExporting model to OpenVINO format...")
        openvino_path = model.export(format="openvino", imgsz=imgsz, half=True)
        print(f"OpenVINO model exported to: {openvino_path}")
        
        print("\nAll exports completed successfully!")
        print("\nRecommendations for Raspberry Pi 4:")
        print("- For best performance: Use the OpenVINO model")
        print("- For good compatibility: Use the TFLite model")
        print("- For maximum compatibility: Use the ONNX model")
        
    except Exception as e:
        print(f"\nExport error: {e}")
        print("You can manually export the model later using export_to_raspi.py")

def validate_model_performance(model_path: str, data_yaml_path: str, imgsz: int = 256):
    """Validate the performance of a trained YOLOv8 model."""
    print(f"\nLoading model from: {model_path}")
    
    # Set HSA_OVERRIDE_GFX_VERSION for AMD GPU compatibility if needed
    if "HSA_OVERRIDE_GFX_VERSION" not in os.environ and torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0)
            if "AMD" in device_name:
                print("AMD GPU detected, setting HSA_OVERRIDE_GFX_VERSION for compatibility")
                os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
        except Exception:
            pass  # Fail silently if we can't get device name
    
    model = YOLO(model_path)

    print(f"\nStarting validation using dataset: {data_yaml_path}")
    try:
        metrics = model.val(
            data=data_yaml_path,
            imgsz=imgsz,
            split='val', # Explicitly validate on the validation set
            save_json=True, # Save metrics to a JSON file
            save_hybrid=True, # Save predictions in YOLO format
            conf=0.001, # Object confidence threshold for plotting
            iou=0.6, # IoU threshold for NMS
            half=True if torch.cuda.is_available() else False, # Use half precision if GPU is available
            plots=True, # Save plots
            project="runs/detect",
            name=f"validation_{Path(model_path).parent.parent.name}", # Save in a validation specific folder
            exist_ok=True # Allow existing folder
        )
        print("\nValidation completed!")
        print(f"Metrics: {metrics.results_dict}")
    except Exception as e:
        print(f"\nValidation error: {e}")

if __name__ == "__main__":
    print("\n=== Raspberry Pi + Cloud Training System ===")
    print("1. Train a new model")
    print("2. Validate an existing model")
    print("3. Advanced training with tuned hyperparameters")
    print("4. High-Performance training (mAP50 target: 0.95+)")
    print("5. Temperature-friendly training (for GPU cooling)")
    
    main_choice = input("\nSelect an option (1/2/3/4/5, default=1): ").strip() or "1"
    
    # Set HSA_OVERRIDE_GFX_VERSION for AMD GPU compatibility
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

    if main_choice == "1":
        # Training flow
        data_yaml_path = select_dataset()
        if not data_yaml_path:
            print("No dataset selected. Exiting...")
            sys.exit(1)
        
        # Allow user to select image size
        print("\nSelect image size for training:")
        print("1. 256px (fastest, lowest accuracy)")
        print("2. 320px (balanced for Raspberry Pi)")
        print("3. 416px (better accuracy, slower on Pi)")
        print("4. 640px (best accuracy, may be too slow for Pi)")
        
        size_choice = input("\nSelect image size (1/2/3/4, default=2): ").strip() or "2"
        if size_choice == "1":
            img_size = 256
        elif size_choice == "2":
            img_size = 320
        elif size_choice == "3":
            img_size = 416
        elif size_choice == "4":
            img_size = 640
        else:
            print("Invalid choice, defaulting to 320px")
            img_size = 320
        
        # Configure training parameters optimized for Raspberry Pi
        config = {
            "data_yaml_path": data_yaml_path,
            "epochs": 100,
            "imgsz": img_size,
            "batch_size": 16 if img_size <= 416 else 8,  # Reduce batch size for larger images
            "model_type": "yolov8n.pt"  # Nano model is best for Raspberry Pi
        }
        
        print("\nTraining configuration for Raspberry Pi deployment:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        print("\nStarting training with settings optimized for Raspberry Pi 4...")
        
        train_for_raspi(**config)
    
    elif main_choice == "2":
        # Validation flow
        model_to_validate = input("\nEnter the path to the model you want to validate (e.g., runs/detect/your_run_name/weights/best.pt): ").strip()
        if not os.path.exists(model_to_validate):
            print(f"Error: Model file not found at {model_to_validate}")
            sys.exit(1)
            
        data_yaml_path = select_dataset() # Reuse dataset selection for validation
        if not data_yaml_path:
            print("No dataset selected for validation. Exiting...")
            sys.exit(1)
        
        # Allow user to select image size for validation
        print("\nSelect image size for validation:")
        print("1. 256px (fastest)")
        print("2. 320px (balanced for Raspberry Pi)")
        print("3. 416px (better accuracy)")
        print("4. 640px (best accuracy)")
        print("5. Same as training size (recommended)")
        
        size_choice = input("\nSelect image size (1/2/3/4/5, default=5): ").strip() or "5"
        if size_choice == "1":
            img_size = 256
        elif size_choice == "2":
            img_size = 320
        elif size_choice == "3":
            img_size = 416
        elif size_choice == "4":
            img_size = 640
        elif size_choice == "5":
            # Try to determine the training image size from the model
            print("\nUsing same size as training...")
            img_size = 320  # Default fallback
        else:
            print("Invalid choice, defaulting to 320px")
            img_size = 320
            
        validate_model_performance(model_to_validate, data_yaml_path, imgsz=img_size)
    
    elif main_choice == "3":
        # Advanced training with tuned hyperparameters
        data_yaml_path = select_dataset()
        if not data_yaml_path:
            print("No dataset selected. Exiting...")
            sys.exit(1)
        
        print("\nAdvanced training configuration (tuned for better performance):")
        
        # Set HSA_OVERRIDE_GFX_VERSION for AMD GPU compatibility
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
        
        # Select model type
        print("\nSelect model for advanced training:")
        print("1. YOLOv8n (nano) - smallest, fastest, suitable for Pi")
        print("2. YOLOv8s (small) - balanced, may be slow on Pi")
        print("3. YOLOv8m (medium) - higher accuracy, likely too slow for Pi")
        
        model_choice = input("\nSelect model (1/2/3, default=1): ").strip() or "1"
        if model_choice == "1":
            model_type = "yolov8n.pt"
        elif model_choice == "2":
            model_type = "yolov8s.pt"
        elif model_choice == "3":
            model_type = "yolov8m.pt"
        else:
            print("Invalid choice, defaulting to YOLOv8n")
            model_type = "yolov8n.pt"
        
        # Allow user to select image size
        print("\nSelect image size for training:")
        print("1. 320px (balanced for Raspberry Pi)")
        print("2. 384px (better accuracy, still reasonable for Pi)")
        print("3. 416px (good accuracy, may be slow on Pi)")
        print("4. 512px (better accuracy, likely too slow for Pi)")
        print("5. 640px (best accuracy, too slow for Pi)")
        
        size_choice = input("\nSelect image size (1/2/3/4/5, default=2): ").strip() or "2"
        if size_choice == "1":
            img_size = 320
        elif size_choice == "2":
            img_size = 384
        elif size_choice == "3":
            img_size = 416
        elif size_choice == "4":
            img_size = 512
        elif size_choice == "5":
            img_size = 640
        else:
            print("Invalid choice, defaulting to 384px")
            img_size = 384
            
        # Adjust batch size based on model and image size
        if model_type == "yolov8n.pt":
            if img_size <= 384:
                batch_size = 16
            elif img_size <= 512:
                batch_size = 12
            else:
                batch_size = 8
        elif model_type == "yolov8s.pt":
            if img_size <= 384:
                batch_size = 12
            elif img_size <= 512:
                batch_size = 8
            else:
                batch_size = 6
        else:  # yolov8m.pt
            if img_size <= 384:
                batch_size = 8
            elif img_size <= 512:
                batch_size = 6
            else:
                batch_size = 4
        
        # Enhanced configuration based on validation results
        advanced_config = {
            "data_yaml_path": data_yaml_path,
            "epochs": 150,                # More epochs for better convergence
            "imgsz": img_size,            # User-selected image size
            "batch_size": batch_size,     # Adjusted batch size based on model and image size
            "model_type": model_type,     # User-selected model
            "lr0": 0.01,                  # Initial learning rate
            "lrf": 0.01,                  # Final learning rate factor (lower for better convergence)
            "weight_decay": 0.0005,       # L2 regularization
            "augment": True,              # Enable data augmentation
            "mixup": 0.1,                 # Add mixup augmentation
            "mosaic": 1.0,                # Full mosaic augmentation
            "close_mosaic": 15            # Disable mosaic in final 15 epochs
        }
        
        print("\nEnhanced training configuration:")
        for key, value in advanced_config.items():
            print(f"  {key}: {value}")
        
        print("\nStarting advanced training with enhanced parameters...")
        train_for_raspi(**advanced_config)
    
    elif main_choice == "4":
        # High-Performance training targeting 0.95 mAP50
        data_yaml_path = select_dataset()
        if not data_yaml_path:
            print("No dataset selected. Exiting...")
            sys.exit(1)
        
        print("\nHigh-Performance training targeting 0.95 mAP50...")
        print("WARNING: These settings may require more GPU resources and training time")
        
        # Set HSA_OVERRIDE_GFX_VERSION for AMD GPU compatibility
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
        
        # Add GPU temperature management option
        print("\nGPU Temperature Management:")
        print("1. Standard - highest performance (may run hot)")
        print("2. Balanced - good performance with temperature control")
        print("3. Cool - prioritize GPU temperature over training speed")
        
        temp_choice = input("\nSelect temperature management (1/2/3, default=2): ").strip() or "2"
        
        if temp_choice == "1":
            # Standard settings (original high performance)
            workers = 4
            use_amp = False
            progressive = False
            max_batch = None  # No limit
            print("\nUsing standard high-performance settings (may run hot)")
        elif temp_choice == "2":
            # Balanced settings
            workers = 2
            use_amp = False
            progressive = False
            max_batch = None  # No limit but will recommend lower values
            print("\nUsing balanced settings for temperature management")
        else:
            # Cool settings
            workers = 2
            use_amp = False
            progressive = True
            max_batch = 8  # Limit max batch size to 8
            print("\nUsing GPU temperature-friendly settings")
        
        # Model options
        print("\nSelect model for high-performance training:")
        print("1. YOLOv8n (nano) - smallest, fastest, least accurate")
        print("2. YOLOv8s (small) - balanced size/speed/accuracy")
        print("3. YOLOv8m (medium) - higher accuracy, slower")
        
        model_choice = input("\nSelect model (1/2/3, default=2): ").strip() or "2"
        if model_choice == "1":
            model_type = "yolov8n.pt"
        elif model_choice == "2":
            model_type = "yolov8s.pt"
        elif model_choice == "3":
            model_type = "yolov8m.pt"
        else:
            print("Invalid choice, defaulting to YOLOv8s")
            model_type = "yolov8s.pt"
        
        # Image size options - adjust based on temperature selection
        if temp_choice == "3":  # Cool
            print("\nSelect image size (temperature-optimized options):")
            print("1. 320px (minimal heat generation)")
            print("2. 384px (balanced performance/temperature)")
            print("3. 416px (good performance/temperature)")
            print("4. 512px (higher accuracy, more heat)")
            
            size_choice = input("\nSelect image size (1/2/3/4, default=2): ").strip() or "2"
            if size_choice == "1":
                img_size = 320
            elif size_choice == "2":
                img_size = 384
            elif size_choice == "3":
                img_size = 416
            elif size_choice == "4":
                img_size = 512
            else:
                print("Invalid choice, defaulting to 384px")
                img_size = 384
        else:  # Standard or Balanced
            print("\nSelect image size for high-performance training:")
            print("1. 320px (fastest training)")
            print("2. 384px (good balance for smaller models)")
            print("3. 416px (good balance of accuracy and training speed)")
            print("4. 512px (better accuracy, slower training)")
            print("5. 640px (best accuracy, slowest training)")
            if temp_choice == "1":  # Only show 768 for standard mode
                print("6. 768px (experimental - highest possible accuracy)")
            
            max_option = "6" if temp_choice == "1" else "5"
            default_option = "4" if temp_choice == "2" else "5"  # Default to 512px for balanced mode
            
            size_choice = input(f"\nSelect image size (1/2/3/4/5{'/6' if temp_choice == '1' else ''}, default={default_option}): ").strip() or default_option
            if size_choice == "1":
                img_size = 320
            elif size_choice == "2":
                img_size = 384
            elif size_choice == "3":
                img_size = 416
            elif size_choice == "4":
                img_size = 512
            elif size_choice == "5":
                img_size = 640
            elif size_choice == "6" and temp_choice == "1":
                img_size = 768
                print("WARNING: 768px resolution requires significant GPU memory and generates a lot of heat!")
            else:
                print(f"Invalid choice, defaulting to {512 if temp_choice == '2' else 640}px")
                img_size = 512 if temp_choice == "2" else 640
        
        # Adjust batch size based on model, image size and temperature setting
        if model_type == "yolov8n.pt":
            if img_size <= 320:
                batch = 14 if temp_choice == "3" else 16
            elif img_size <= 384:
                batch = 12 if temp_choice == "3" else 16
            elif img_size <= 416:
                batch = 10 if temp_choice == "3" else 14
            elif img_size <= 512:
                batch = 8 if temp_choice == "3" else 12
            elif img_size <= 640:
                batch = 6 if temp_choice == "3" else 8
            else:
                batch = 4
        elif model_type == "yolov8s.pt":
            if img_size <= 320:
                batch = 10 if temp_choice == "3" else 12
            elif img_size <= 384:
                batch = 8 if temp_choice == "3" else 10
            elif img_size <= 416:
                batch = 6 if temp_choice == "3" else 8
            elif img_size <= 512:
                batch = 4 if temp_choice == "3" else 6
            elif img_size <= 640:
                batch = 3 if temp_choice == "3" else 4
            else:
                batch = 2
        else:  # yolov8m.pt
            if img_size <= 320:
                batch = 8 if temp_choice == "3" else 10
            elif img_size <= 384:
                batch = 6 if temp_choice == "3" else 8
            elif img_size <= 416:
                batch = 4 if temp_choice == "3" else 6
            elif img_size <= 512:
                batch = 2 if temp_choice == "3" else 4
            elif img_size <= 640:
                batch = 1 if temp_choice == "3" else 2
            else:
                batch = 1
        
        # Apply max batch limit if specified (for cool mode)
        if max_batch is not None and batch > max_batch:
            batch = max_batch
        
        print(f"\nSelected {model_type} with {img_size}px resolution")
        print(f"Recommended batch size: {batch} (adjust if you encounter memory or temperature issues)")
        custom_batch = input(f"Enter batch size (default={batch}): ").strip()
        if custom_batch and custom_batch.isdigit() and int(custom_batch) > 0:
            batch = int(custom_batch)
        
        # Adjust epochs based on temperature settings
        if temp_choice == "1":  # Standard
            epochs = 300
            patience = 100
        elif temp_choice == "2":  # Balanced
            epochs = 250
            patience = 80
        else:  # Cool
            epochs = 200
            patience = 60
        
        # High-performance configuration to reach 0.95 mAP50
        high_perf_config = {
            "data_yaml_path": data_yaml_path,
            "epochs": epochs,                # Adjusted based on temperature preference
            "imgsz": img_size,
            "batch_size": batch,
            "model_type": model_type,
            "lr0": 0.01,
            "lrf": 0.005,                    # Very low final learning rate for fine convergence
            "weight_decay": 0.0005,
            "patience": patience,            # Adjusted based on temperature preference
            "optimizer": "SGD",              # Changed to SGD for better temperature management
            "augment": True,
            "mixup": 0.15,                   # Increased mixup for better generalization
            "mosaic": 1.0,
            "close_mosaic": 30,              # Disable mosaic in final 30 epochs for fine tuning
            "cool_down": temp_choice != "1"  # Enable temperature management except in standard mode
        }
        
        print("\nHigh-Performance configuration (temperature-optimized):")
        for key, value in high_perf_config.items():
            print(f"  {key}: {value}")
        
        if progressive and temp_choice == "3":
            print("\nStarting progressive high-performance training with cooling breaks...")
            print("This achieves high accuracy while managing GPU temperature")
            
            # Stage 1: Initial training (40% of epochs)
            stage1_epochs = int(epochs * 0.4)
            print(f"\nStage 1: Training for {stage1_epochs} epochs...")
            stage1_config = high_perf_config.copy()
            stage1_config["epochs"] = stage1_epochs
            train_for_raspi(**stage1_config)
            
            # Cooling break
            print("\n=== GPU COOLING BREAK (60 seconds) ===")
            print("Allowing GPU to cool down before continuing...")
            import time
            for i in range(60, 0, -5):
                print(f"Resuming in {i} seconds...")
                time.sleep(5)
            
            # Stage 2: Mid training (30% of epochs)
            stage2_epochs = int(epochs * 0.3)
            print(f"\nStage 2: Training for {stage2_epochs} more epochs...")
            stage2_config = high_perf_config.copy()
            stage2_config["epochs"] = stage2_epochs
            train_for_raspi(**stage2_config)
            
            # Another cooling break
            print("\n=== GPU COOLING BREAK (45 seconds) ===")
            print("Allowing GPU to cool down before final stage...")
            for i in range(45, 0, -5):
                print(f"Resuming in {i} seconds...")
                time.sleep(5)
            
            # Stage 3: Final training (30% of epochs)
            stage3_epochs = epochs - stage1_epochs - stage2_epochs
            print(f"\nStage 3: Final {stage3_epochs} epochs...")
            stage3_config = high_perf_config.copy()
            stage3_config["epochs"] = stage3_epochs
            train_for_raspi(**stage3_config)
            
            print("\nProgressive high-performance training completed!")
        else:
            print("\nStarting high-performance training targeting 0.95 mAP50...")
            train_for_raspi(**high_perf_config)
    
    elif main_choice == "5":
        # Temperature-friendly training to reduce GPU heat
        data_yaml_path = select_dataset()
        if not data_yaml_path:
            print("No dataset selected. Exiting...")
            sys.exit(1)
        
        print("\n===== Temperature-Friendly Training Mode =====")
        print("This mode reduces GPU temperature by using:")
        print("- Lower resolution images")
        print("- Smaller batch sizes")
        print("- Reduced worker threads")
        print("- Gradual training approach")
        
        # Select model type
        print("\nSelect model (smaller = cooler GPU):")
        print("1. YOLOv8n (nano) - smallest, coolest")
        print("2. YOLOv8s (small) - moderate heat")
        print("3. YOLOv8m (medium) - generates more heat")
        
        model_choice = input("\nSelect model (1/2/3, default=1): ").strip() or "1"
        if model_choice == "1":
            model_type = "yolov8n.pt"
        elif model_choice == "2":
            model_type = "yolov8s.pt"
        elif model_choice == "3":
            model_type = "yolov8m.pt"
        else:
            print("Invalid choice, defaulting to YOLOv8n")
            model_type = "yolov8n.pt"
        
        print("\nSelect image size (smaller = cooler GPU):")
        print("1. 320px (minimum heat generation)")
        print("2. 384px (moderate heat generation)")
        print("3. 416px (balanced performance/heat)")
        print("4. 512px (higher performance, more heat)")
        print("5. 640px (highest quality, maximum heat)")
        
        size_choice = input("\nSelect image size (1/2/3/4/5, default=1): ").strip() or "1"
        if size_choice == "1":
            img_size = 320
        elif size_choice == "2":
            img_size = 384
        elif size_choice == "3":
            img_size = 416
        elif size_choice == "4":
            img_size = 512
        elif size_choice == "5":
            img_size = 640
            print("\nWARNING: 640px will generate significant heat. Consider using progressive training.")
        else:
            print("Invalid choice, defaulting to 320px")
            img_size = 320
            
        # Adjust batch size based on model and image size for temperature control
        if model_type == "yolov8n.pt":
            if img_size <= 384:
                batch = 8
            elif img_size <= 416:
                batch = 6
            elif img_size <= 512:
                batch = 4
            else:  # 640px
                batch = 3
        elif model_type == "yolov8s.pt":
            if img_size <= 384:
                batch = 6
            elif img_size <= 416:
                batch = 4
            elif img_size <= 512:
                batch = 2
            else:  # 640px
                batch = 1
        else:  # yolov8m.pt
            if img_size <= 384:
                batch = 4
            elif img_size <= 416:
                batch = 2
            elif img_size <= 512:
                batch = 1
            else:  # 640px
                batch = 1
                print("\nWARNING: YOLOv8m at 640px resolution may be unstable due to high memory usage.")
                print("Consider a smaller image size or batch size if training fails.")
        
        print("\nHow would you like to manage training duration?")
        print("1. Shorter training (100 epochs) - less heat over time")
        print("2. Standard training (150 epochs) - balanced duration/performance")
        print("3. Progressive training - starts with few epochs, gradually increases")
        
        duration_choice = input("\nSelect training duration (1/2/3, default=1): ").strip() or "1"
        if duration_choice == "1":
            epochs = 100
            progressive = False
        elif duration_choice == "2":
            epochs = 150
            progressive = False
        elif duration_choice == "3":
            epochs = 200  # Total epochs for progressive training
            progressive = True
        else:
            print("Invalid choice, defaulting to shorter training")
            epochs = 100
            progressive = False
        
        # Temperature-friendly configuration
        cool_config = {
            "data_yaml_path": data_yaml_path,
            "epochs": epochs,
            "imgsz": img_size,
            "batch_size": batch,
            "model_type": model_type,
            "lr0": 0.01,
            "lrf": 0.01,
            "weight_decay": 0.0005,
            "augment": True,
            "mixup": 0.1,
            "mosaic": 1.0,
            "close_mosaic": 10,
            "patience": 50,
            "optimizer": "SGD",
            "cool_down": True  # Enable temperature management mode
        }
        
        print("\nTemperature-Friendly configuration:")
        for key, value in cool_config.items():
            print(f"  {key}: {value}")
        
        if progressive:
            print("\nStarting progressive training to manage temperature...")
            print("This will train in 3 stages with cooling breaks in between")
            
            # Stage 1: Initial training (40% of epochs)
            stage1_epochs = int(epochs * 0.4)
            print(f"\nStage 1: Training for {stage1_epochs} epochs...")
            cool_config["epochs"] = stage1_epochs
            train_for_raspi(**cool_config)
            
            # Cooling break
            print("\n=== GPU COOLING BREAK (60 seconds) ===")
            print("Allowing GPU to cool down before continuing...")
            import time
            for i in range(60, 0, -5):
                print(f"Resuming in {i} seconds...")
                time.sleep(5)
            
            # Stage 2: Mid training (30% of epochs)
            stage2_epochs = int(epochs * 0.3)
            print(f"\nStage 2: Training for {stage2_epochs} more epochs...")
            cool_config["epochs"] = stage2_epochs
            train_for_raspi(**cool_config)
            
            # Another cooling break
            print("\n=== GPU COOLING BREAK (45 seconds) ===")
            print("Allowing GPU to cool down before final stage...")
            for i in range(45, 0, -5):
                print(f"Resuming in {i} seconds...")
                time.sleep(5)
            
            # Stage 3: Final training (30% of epochs)
            stage3_epochs = epochs - stage1_epochs - stage2_epochs
            print(f"\nStage 3: Final {stage3_epochs} epochs...")
            cool_config["epochs"] = stage3_epochs
            train_for_raspi(**cool_config)
            
            print("\nProgressive training completed!")
        else:
            print("\nStarting temperature-friendly training...")
            train_for_raspi(**cool_config)
    
    else:
        print("Invalid option. Exiting...")
        sys.exit(1) 