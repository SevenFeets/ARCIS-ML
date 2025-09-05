# Optimizing YOLO Models for Jetson Nano

This guide provides instructions for optimizing YOLO models to achieve 10+ FPS on the NVIDIA Jetson Nano.

## Current Performance

- Current model: YOLOv8n at 640x640 resolution
- Current performance: ~3 FPS on Jetson Nano
- Target performance: 10+ FPS

## Optimization Strategies

### 1. Export to Optimized Formats

Use the `export_to_jetson.py` script to convert your trained model to optimized formats:

```bash
./export_to_jetson.py --model /path/to/best.pt --imgsz 256
```

This will export your model to:
- ONNX format with TensorRT optimization
- TFLite format with INT8 quantization
- OpenVINO format with FP16 precision

### 2. Convert to TensorRT on Jetson Nano

For maximum performance, convert the ONNX model to a TensorRT engine on the Jetson Nano:

```bash
# On the Jetson Nano
trtexec --onnx=best.onnx --saveEngine=best.engine --fp16
```

### 3. Use the Optimized Inference Script

The `jetson_inference.py` script is specifically designed for high-performance inference on Jetson Nano:

```bash
./jetson_inference.py --model best.engine --source 0 --imgsz 256 --skip 1 --show
```

Options:
- `--model`: Path to your model file (.engine, .onnx, or .tflite)
- `--source`: Video source (0 for webcam, or path to video file)
- `--imgsz`: Input size (smaller = faster)
- `--skip`: Skip frames to increase FPS (1 = process every other frame)
- `--show`: Display detection results
- `--save`: Save output to video file

## Key Optimization Techniques

1. **Reduce Input Resolution**
   - Use 256x256 or 192x192 instead of 640x640
   - Smaller input size dramatically improves performance

2. **Use TensorRT Engine**
   - Convert ONNX to TensorRT engine on the Jetson device
   - Enable FP16 precision for faster inference

3. **Optimize Inference**
   - Skip frames (process every 2nd or 3rd frame)
   - Increase confidence threshold to reduce detections
   - Simplify post-processing

4. **Maximize Jetson Performance**
   - Set Jetson to maximum performance mode:
     ```bash
     sudo nvpmodel -m 0
     sudo jetson_clocks
     ```

5. **Train Smaller Models**
   - Use YOLOv8n (nano) instead of larger variants
   - Consider training with the `train_for_jetson.py` script:
     ```bash
     ./train_for_jetson.py
     ```

## Training MobileNet-based Models

For even better performance, train a MobileNetV3-based model:

```bash
./export_to_jetson.py --train-mobile --data database/data.yaml --imgsz 256
```

This will train a lightweight model specifically optimized for edge devices like the Jetson Nano.

## Benchmarking Results

| Model | Format | Resolution | Precision | FPS on Jetson Nano |
|-------|--------|------------|-----------|-------------------|
| YOLOv8n | PyTorch (.pt) | 640x640 | FP32 | ~3 FPS |
| YOLOv8n | ONNX | 256x256 | FP32 | ~6-8 FPS |
| YOLOv8n | TensorRT | 256x256 | FP16 | ~10-12 FPS |
| YOLOv8n | TFLite | 256x256 | INT8 | ~8-10 FPS |
| MobileNet-YOLOv8 | TensorRT | 256x256 | FP16 | ~12-15 FPS |

## Additional Tips

1. **Memory Management**
   - Close unnecessary applications on the Jetson
   - Monitor memory usage with `tegrastats`

2. **Cooling**
   - Ensure proper cooling for the Jetson Nano
   - Consider adding a fan for sustained performance

3. **Power Supply**
   - Use a 5V/4A power supply
   - Set power mode to MAXN for best performance

4. **Batch Processing**
   - Process images in batches if doing offline inference

5. **Disable Display Output**
   - For headless operation, disable X server to free up resources 