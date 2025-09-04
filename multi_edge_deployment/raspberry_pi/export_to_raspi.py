#!/usr/bin/env python3
from ultralytics import YOLO
import os
import argparse

def export_for_raspi(model_path, imgsz=256):
    """
    Export a YOLO model to formats optimized for Raspberry Pi
    
    Args:
        model_path: Path to the PyTorch model (.pt file)
        imgsz: Image size for the exported models (smaller is faster on Raspberry Pi)
    """
    # Set the HSA_OVERRIDE_GFX_VERSION environment variable for AMD GPU compatibility
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
    
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # 1. Export to ONNX (good general format)
    try:
        print(f"\nExporting model to ONNX format with image size {imgsz}...")
        onnx_path = model.export(format="onnx", imgsz=imgsz, simplify=True)
        print(f"✅ ONNX model exported to: {onnx_path}")
    except Exception as e:
        print(f"❌ ONNX export error: {e}")
    
    # 2. Export to TFLite (great for Raspberry Pi)
    try:
        print(f"\nExporting model to TFLite format with image size {imgsz}...")
        tflite_path = model.export(format="tflite", imgsz=imgsz, int8=True)
        print(f"✅ TFLite model exported to: {tflite_path}")
    except Exception as e:
        print(f"❌ TFLite export error: {e}")
    
    # 3. Export to OpenVINO (optimized for edge devices)
    try:
        print(f"\nExporting model to OpenVINO format with image size {imgsz}...")
        openvino_path = model.export(format="openvino", imgsz=imgsz, half=True)
        print(f"✅ OpenVINO model exported to: {openvino_path}")
    except Exception as e:
        print(f"❌ OpenVINO export error: {e}")
    
    # 4. Export to CoreML (for Apple devices)
    try:
        print(f"\nExporting model to CoreML format with image size {imgsz}...")
        coreml_path = model.export(format="coreml", imgsz=imgsz, half=True)
        print(f"✅ CoreML model exported to: {coreml_path}")
    except Exception as e:
        print(f"❌ CoreML export error: {e}")
    
    print("\n=== Raspberry Pi Deployment Guide ===")
    print("1. Copy the exported models to your Raspberry Pi")
    print("2. Install the appropriate runtime:")
    print("   - For ONNX: pip install onnxruntime")
    print("   - For TFLite: pip install tflite-runtime")
    print("   - For OpenVINO: pip install openvino")
    print("3. Use the appropriate model based on your needs:")
    print("   - OpenVINO: Best performance on Raspberry Pi 4")
    print("   - TFLite: Good balance of performance and compatibility")
    print("   - ONNX: Maximum compatibility across platforms")
    print("   - CoreML: For Apple devices only")
    print("\nFor optimal performance on Raspberry Pi 4:")
    print("- Use OpenVINO with half precision (FP16)")
    print("- Set image size to 320x320 or 416x416")
    print("- Consider running inference at 1-2 FPS for real-time applications")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO models for Raspberry Pi")
    parser.add_argument("--model", type=str, default="runs/detect/train/weights/best.pt", 
                       help="Path to the PyTorch model (.pt file)")
    parser.add_argument("--imgsz", type=int, default=256, 
                       help="Image size for the exported models (smaller is faster)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found!")
        exit(1)
    
    export_for_raspi(args.model, args.imgsz) 