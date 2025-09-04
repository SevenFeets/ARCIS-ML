#!/usr/bin/env python3
from ultralytics import YOLO
import os

# Set the HSA_OVERRIDE_GFX_VERSION environment variable for AMD GPU compatibility
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

def export_to_onnx(model_path, img_size=640):
    """
    Export a YOLO model to ONNX format
    
    Args:
        model_path: Path to the PyTorch model (.pt file)
        img_size: Image size for the ONNX model
    """
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    print(f"Exporting model to ONNX format with image size {img_size}...")
    success = model.export(format="onnx", imgsz=img_size, simplify=True)
    
    if success:
        # The export function returns the path to the exported model
        onnx_path = model_path.replace('.pt', '.onnx')
        print(f"Model successfully exported to: {onnx_path}")
    else:
        print("Export failed. Check error messages above.")

if __name__ == "__main__":
    # Path to your trained model
    model_path = "runs/detect/train/weights/best.pt"
    
    # Export the model
    export_to_onnx(model_path, img_size=640) 