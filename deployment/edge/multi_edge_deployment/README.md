# Edge Deployment Tools

This directory contains specialized scripts and tools for optimizing and deploying YOLO models to edge devices.

## Directory Structure

- **raspberry_pi/**: Files for Raspberry Pi 4 optimization and deployment
- **jetson_nano/**: Files for NVIDIA Jetson Nano optimization and deployment
- **check_onnx.py**: Utility to validate ONNX models
- **train_with_rocm.py**: Script for training with AMD GPUs using ROCm
- **train_with_rocm_fixed.py**: Fixed version with optimized settings for AMD GPUs

## Raspberry Pi Tools

Files in the `raspberry_pi/` directory:

- **train_for_raspi.py**: Trains YOLOv8 models optimized for Raspberry Pi (smaller size)
- **export_to_raspi.py**: Converts existing models to Raspberry Pi-friendly formats (ONNX, TFLite, OpenVINO)

## Jetson Nano Tools

Files in the `jetson_nano/` directory:

- **train_for_jetson.py**: Trains YOLOv8 models optimized for Jetson Nano (smaller input size and ultralight models)
- **export_to_jetson.py**: Converts existing models to Jetson-optimized formats (ONNX for TensorRT, TFLite with INT8 quantization)
- **jetson_inference.py**: Specialized inference script for high performance on Jetson Nano
- **JETSON_OPTIMIZATION.md**: Detailed guide for optimizing YOLO models to achieve 10+ FPS on Jetson Nano
- **yolov8n-ultralight.yaml**: Configuration for ultralight YOLOv8n model with reduced channels for maximum speed

## Training Utilities

- **train_with_rocm.py**: Script for training on AMD GPUs with ROCm support
  - Sets necessary environment variables for AMD compatibility (HSA_OVERRIDE_GFX_VERSION)
  - Optimized for 12GB VRAM with batch_size=32 and imgsz=640

- **train_with_rocm_fixed.py**: Enhanced version with more robust AMD GPU settings
  - Adds additional environment variables for better memory management
  - Uses smaller batch_size=4 and imgsz=320 for better compatibility
  - Reduces epochs to 50 for faster training

- **check_onnx.py**: Utility to validate and inspect ONNX models
  - Verifies model structure and validity
  - Displays metadata, inputs, and outputs
  - Reports model size

## Performance Comparison

| Device | Model | Format | Resolution | Precision | FPS |
|--------|-------|--------|------------|-----------|-----|
| Raspberry Pi 4 | YOLOv8n | TFLite | 320x320 | INT8 | ~2-3 FPS |
| Raspberry Pi 4 | YOLOv8n | OpenVINO | 320x320 | FP16 | ~3-4 FPS |
| Jetson Nano | YOLOv8n | PyTorch | 640x640 | FP32 | ~3 FPS |
| Jetson Nano | YOLOv8n | TensorRT | 256x256 | FP16 | ~10-12 FPS |
| Jetson Nano | YOLOv8n-ultralight | TensorRT | 256x256 | FP16 | ~12-15 FPS |

## Usage Instructions

### Raspberry Pi Deployment

1. Train an optimized model for Raspberry Pi:
   ```bash
   ./train_for_raspi.py
   ```

2. Or convert an existing model for Raspberry Pi:
   ```bash
   ./export_to_raspi.py --model path/to/best.pt --imgsz 320
   ```

### Jetson Nano Deployment

1. Train an optimized model for Jetson Nano:
   ```bash
   ./train_for_jetson.py
   ```

2. Or convert an existing model for Jetson Nano:
   ```bash
   ./export_to_jetson.py --model path/to/best.pt --imgsz 256
   ```

3. On the Jetson Nano device, convert the ONNX model to TensorRT:
   ```bash
   trtexec --onnx=best.onnx --saveEngine=best.engine --fp16
   ```

4. Run inference using the optimized script:
   ```bash
   ./jetson_inference.py --model best.engine --source 0 --imgsz 256 --skip 1 --show
   ```

### Training with AMD GPUs

1. Train using standard settings (optimized for 12GB VRAM):
   ```bash
   ./train_with_rocm.py
   ```

2. Train using more conservative settings for better compatibility:
   ```bash
   ./train_with_rocm_fixed.py
   ```

### Checking ONNX Models

Validate an exported ONNX model:
```bash
./check_onnx.py
```

## Common Optimization Techniques

1. Reduce input resolution (256x256 or 320x320)
2. Use smaller models (YOLOv8n, MobileNet-based models)
3. Use hardware-specific acceleration (TensorRT, OpenVINO)
4. Quantize to reduced precision (INT8, FP16)
5. Skip frames for higher FPS in video applications
6. Use ultralight model variants with reduced channels 