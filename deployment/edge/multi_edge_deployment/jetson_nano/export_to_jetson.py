#!/usr/bin/env python3
from ultralytics import YOLO
import os
import argparse
import sys

def export_for_jetson(model_path, imgsz=256):
    """
    Export a YOLO model to formats optimized for Jetson Nano to achieve 10+ FPS
    
    Args:
        model_path: Path to the PyTorch model (.pt file)
        imgsz: Image size for the exported models (smaller is faster on Jetson)
    """
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # 1. Export to ONNX with optimization for TensorRT
    try:
        print(f"\nExporting model to ONNX format optimized for TensorRT with image size {imgsz}...")
        onnx_path = model.export(format="onnx", imgsz=imgsz, simplify=True, opset=11)
        print(f"✅ ONNX model exported to: {onnx_path}")
        print("  - This model can be converted to TensorRT on the Jetson device")
    except Exception as e:
        print(f"❌ ONNX export error: {e}")
    
    # 2. Export to TFLite with INT8 quantization
    try:
        print(f"\nExporting model to TFLite format with INT8 quantization and image size {imgsz}...")
        tflite_path = model.export(format="tflite", imgsz=imgsz, int8=True)
        print(f"✅ TFLite model exported to: {tflite_path}")
        print("  - This model can be used with TFLite runtime on Jetson")
    except Exception as e:
        print(f"❌ TFLite export error: {e}")
    
    # 3. Export to OpenVINO (works well on some Jetson configurations)
    try:
        print(f"\nExporting model to OpenVINO format with FP16 precision and image size {imgsz}...")
        openvino_path = model.export(format="openvino", imgsz=imgsz, half=True)
        print(f"✅ OpenVINO model exported to: {openvino_path}")
        print("  - This model can be used with OpenVINO runtime on Jetson")
    except Exception as e:
        print(f"❌ OpenVINO export error: {e}")
    
    print("\n=== Jetson Nano Deployment Guide ===")
    print("1. Copy the exported models to your Jetson Nano")
    print("2. For best performance (10+ FPS):")
    print("   - Use TensorRT: Convert ONNX model to TensorRT engine on the Jetson")
    print("     $ trtexec --onnx=model.onnx --saveEngine=model.engine --fp16")
    print("3. Set Jetson to maximum performance mode:")
    print("   $ sudo nvpmodel -m 0")
    print("   $ sudo jetson_clocks")
    print("4. Additional optimization tips:")
    print("   - Use smaller input resolution (256x256)")
    print("   - Use FP16 precision instead of FP32")
    print("   - Consider batch size of 1 for inference")
    print("   - Disable unnecessary post-processing")

def optimize_for_jetson(model_path, imgsz=256):
    """Optimize an existing model for Jetson Nano by reducing size and precision"""
    print(f"\nOptimizing {model_path} for Jetson Nano...")
    
    # Load the original model
    model = YOLO(model_path)
    
    # 1. Export to smaller size ONNX with optimization for TensorRT
    try:
        print(f"\nExporting to smaller ONNX format (size={imgsz}x{imgsz})...")
        onnx_path = model.export(format="onnx", imgsz=imgsz, simplify=True, opset=11)
        print(f"✅ Optimized ONNX model exported to: {onnx_path}")
    except Exception as e:
        print(f"❌ ONNX export error: {e}")
    
    # 2. Export to TFLite with INT8 quantization (best for Jetson Nano)
    try:
        print(f"\nExporting to TFLite with INT8 quantization (size={imgsz}x{imgsz})...")
        tflite_path = model.export(format="tflite", imgsz=imgsz, int8=True)
        print(f"✅ INT8 quantized TFLite model exported to: {tflite_path}")
    except Exception as e:
        print(f"❌ TFLite export error: {e}")
    
    print("\n=== Performance Optimization Tips for Jetson Nano ===")
    print("1. Convert ONNX to TensorRT on the Jetson device:")
    print(f"   $ trtexec --onnx={os.path.basename(onnx_path)} --saveEngine=model.engine --fp16")
    print("2. Set Jetson to maximum performance mode:")
    print("   $ sudo nvpmodel -m 0")
    print("   $ sudo jetson_clocks")
    print("3. Use the INT8 TFLite model for best performance")
    print("4. Consider these additional optimizations:")
    print("   - Process at 256x256 or 192x192 resolution")
    print("   - Reduce detection confidence threshold")
    print("   - Skip frames (process every 2nd or 3rd frame)")
    print("   - Disable NMS or simplify post-processing")

def train_mobilenet_yolo(data_yaml_path, epochs=50, imgsz=256):
    """Train a MobileNetV3-based YOLOv8 model for maximum speed on Jetson Nano"""
    print("\nTraining MobileNetV3-based YOLOv8 model for Jetson Nano...")
    
    # Check if YOLOv8n-mobile.pt exists, if not download it
    mobile_model_path = "yolov8n-mobile.pt"
    if not os.path.exists(mobile_model_path):
        print("Downloading MobileNetV3-based YOLOv8 model...")
        try:
            # Try to use a pre-trained MobileNet-based model if available
            model = YOLO("yolov8n.pt")
            print("Downloaded base YOLOv8n model, will train with reduced size")
        except:
            print("Failed to download model. Check your internet connection.")
            return
    else:
        print(f"Using existing MobileNet model: {mobile_model_path}")
        model = YOLO(mobile_model_path)
    
    # Train the model with settings optimized for Jetson Nano
    try:
        print(f"\nTraining model on {data_yaml_path} with size {imgsz}x{imgsz}...")
        model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=16,
            patience=20,
            optimizer="SGD",  # SGD uses less memory
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            close_mosaic=10,
            project="runs/detect",
            name="jetson_mobile"
        )
        
        print("\nMobileNet model training completed!")
        best_model_path = "runs/detect/jetson_mobile/weights/best.pt"
        print(f"Best model saved at: {best_model_path}")
        
        # Export the model for Jetson Nano
        export_for_jetson(best_model_path, imgsz)
        
    except Exception as e:
        print(f"\nMobileNet model training error: {e}")

def convert_to_tensorrt(onnx_path):
    """
    Provide instructions for converting ONNX to TensorRT on Jetson Nano
    Note: This function doesn't actually perform the conversion as it must be done on the Jetson
    """
    print("\n=== TensorRT Conversion Instructions for Jetson Nano ===")
    print("To convert the ONNX model to TensorRT on your Jetson Nano, follow these steps:")
    print("\n1. Copy the ONNX model to your Jetson Nano")
    print(f"   $ scp {onnx_path} user@jetson:/home/user/")
    print("\n2. On the Jetson Nano, install TensorRT if not already installed")
    print("   $ sudo apt-get update")
    print("   $ sudo apt-get install tensorrt")
    print("\n3. Convert the ONNX model to TensorRT engine")
    print(f"   $ trtexec --onnx={os.path.basename(onnx_path)} --saveEngine=model.engine --fp16")
    print("\n4. Use the TensorRT engine in your application")
    print("   - Python: Use tensorrt and pycuda packages")
    print("   - C++: Use the TensorRT C++ API")
    print("\nNote: The conversion must be done on the Jetson device itself with the same")
    print("TensorRT version that will be used for inference.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO models for Jetson Nano")
    parser.add_argument("--model", type=str, default="runs/detect/train/weights/best.pt", 
                       help="Path to the PyTorch model (.pt file)")
    parser.add_argument("--imgsz", type=int, default=256, 
                       help="Image size for the exported models (smaller is faster)")
    parser.add_argument("--data", type=str, default="database/data.yaml",
                       help="Path to data.yaml file (needed for training)")
    parser.add_argument("--train-mobile", action="store_true",
                       help="Train a MobileNetV3-based model for maximum speed")
    
    args = parser.parse_args()
    
    # Set HSA_OVERRIDE_GFX_VERSION for AMD GPU compatibility
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
    
    if args.train_mobile:
        if not os.path.exists(args.data):
            print(f"Error: Data file {args.data} not found for training!")
            sys.exit(1)
        train_mobilenet_yolo(args.data, epochs=50, imgsz=args.imgsz)
    else:
        if not os.path.exists(args.model):
            print(f"Error: Model file {args.model} not found!")
            sys.exit(1)
        optimize_for_jetson(args.model, args.imgsz) 