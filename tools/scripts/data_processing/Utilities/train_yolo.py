from ultralytics import YOLO
import torch
from pathlib import Path
import re
import gc

def train_yolo(
    data_yaml_path: str,
    epochs: int = 100,
    imgsz: int = 640,  # Increased to 640 for better accuracy
    batch_size: int = 32,  # Increased batch size to utilize more VRAM
    model_type: str = "yolov8n.pt"  # Using nano model
):
    # Force garbage collection to free up memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Print detailed GPU information
    print("\nGPU Information:")
    print("PyTorch version:", torch.__version__)
    
    # Check if PyTorch was built with ROCm
    has_rocm = "+rocm" in torch.__version__
    print("PyTorch built with ROCm:", has_rocm)
    
    # ROCm can be detected through CUDA API in PyTorch
    print("CUDA available:", torch.cuda.is_available())
    
    # Check for GPU
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print("GPU device:", device_name)
        print("GPU memory:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
        
        # Check if it's an AMD GPU
        if re.search(r'AMD|Radeon|RX|Vega', device_name):
            print("AMD GPU detected! Using ROCm for training.")
        else:
            print("NVIDIA GPU detected! Using CUDA for training.")
            
        device = 0  # Use GPU
    else:
        print("WARNING: No GPU detected! Training will be very slow on CPU.")
        print("Please check your ROCm or CUDA installation if you want to use GPU.")
        device = 'cpu'
    
    # Initialize model
    print("\nInitializing YOLO model...")
    model = YOLO(model_type)
    
    # Train the model with optimized settings for AMD GPUs with 12GB VRAM
    print("\nStarting training...")
    try:
        model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,  # Use GPU if available
            patience=50,  # Increased patience for better convergence
            save=True,  # Save best model
            plots=True,  # Generate training plots
            workers=4,  # Increased workers for faster data loading
            cache=False,  # Disable caching to save memory
            amp=False,  # Disable automatic mixed precision for better compatibility with ROCm
            optimizer="SGD",  # Use SGD optimizer for better memory efficiency
            lr0=0.01,  # Initial learning rate
            lrf=0.1,  # Final learning rate
            momentum=0.937,  # SGD momentum
            weight_decay=0.0005,  # L2 regularization
            warmup_epochs=3,  # Learning rate warmup
            warmup_momentum=0.8,  # Warmup momentum
            warmup_bias_lr=0.1,  # Warmup bias learning rate
            close_mosaic=10,  # Enable mosaic augmentation but close it for last 10 epochs
            fraction=1.0,  # Use the full dataset
            rect=False,  # Don't use rectangular training
            cos_lr=True,  # Use cosine learning rate scheduler
            verbose=True,  # Show verbose output
            exist_ok=True,  # Overwrite existing experiment
            nbs=64,  # Increased nominal batch size
            overlap_mask=False,  # Disable mask overlap for faster training
            val=True,  # Enable validation during training
            deterministic=False  # Disable deterministic training for speed
        )
        
        print("\nTraining completed!")
        print(f"Best model saved at: runs/detect/train/weights/best.pt")
        
    except Exception as e:
        print(f"\nTraining error: {e}")
        print("Model weights are still saved at runs/detect/train/weights/")
    
    # Export to ONNX format
    print("\nExporting model to ONNX format...")
    try:
        model.export(format="onnx", imgsz=imgsz, simplify=True)
        print(f"Model exported to ONNX format for deployment")
    except Exception as e:
        print(f"Export error: {e}")
        print("You can manually export the model later.")

def export_for_jetson(model_path, imgsz=320):
    """
    Export a YOLO model to formats optimized for Jetson Nano to achieve higher FPS
    
    Args:
        model_path: Path to the PyTorch model (.pt file)
        imgsz: Image size for the exported models (smaller is faster on Jetson)
    """
    print(f"\nLoading model from {model_path}...")
    model = YOLO(model_path)
    
    # 1. Export to TensorRT (best performance on Jetson)
    try:
        print(f"\nExporting model to TensorRT format with image size {imgsz}...")
        engine_path = model.export(format="engine", imgsz=imgsz, half=True, device=0)
        print(f"TensorRT engine exported to: {engine_path}")
    except Exception as e:
        print(f"TensorRT export error: {e}")
        print("Note: TensorRT export must be done on the Jetson device itself")
    
    # 2. Export to ONNX with optimization
    try:
        print(f"\nExporting model to optimized ONNX format with image size {imgsz}...")
        onnx_path = model.export(format="onnx", imgsz=imgsz, simplify=True, opset=11)
        print(f"Optimized ONNX model exported to: {onnx_path}")
    except Exception as e:
        print(f"ONNX export error: {e}")
    
    # 3. Export to TFLite with INT8 quantization (good for Jetson)
    try:
        print(f"\nExporting model to TFLite format with INT8 quantization...")
        tflite_path = model.export(format="tflite", imgsz=imgsz, int8=True)
        print(f"TFLite model exported to: {tflite_path}")
    except Exception as e:
        print(f"TFLite export error: {e}")
    
    print("\n=== Jetson Nano Optimization Guide ===")
    print("1. For best performance (10+ FPS), use TensorRT engine on Jetson")
    print("   - Install JetPack with TensorRT support")
    print("   - Convert model on the Jetson device itself")
    print("2. Alternative formats:")
    print("   - ONNX with TensorRT backend")
    print("   - TFLite with XNNPACK delegate")
    print("3. Performance tips:")
    print("   - Reduce input resolution to 320x320 or even 256x256")
    print("   - Use FP16 precision (half=True)")
    print("   - Set Jetson to maximum performance mode:")
    print("     sudo nvpmodel -m 0")
    print("     sudo jetson_clocks")
    print("   - Consider using YOLOv8n or YOLOv8n-ultralight variants")

if __name__ == "__main__":
    # Path to your data.yaml file
    data_yaml = str(Path("weapon_detection/data.yaml").absolute())
    
    # Training configuration optimized for AMD GPU with ROCm
    config = {
        "data_yaml_path": data_yaml,
        "epochs": 100,
        "imgsz": 640,
        "batch_size": 32,  # Increased for AMD GPU with 12GB VRAM
        "model_type": "yolov8n.pt"
    }
    
    train_yolo(**config) 