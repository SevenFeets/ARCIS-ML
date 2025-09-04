# Edge Device Training for Weapon Detection

This repository contains scripts for training optimized YOLOv8 models for edge devices like Jetson Nano and Raspberry Pi using the merged weapon detection dataset.

## Available Scripts

1. **train_for_jetson.py** - Optimizes models for Jetson Nano with two options:
   - Standard YOLOv8n with reduced image size
   - Ultralight YOLOv8n (faster but less accurate)

2. **train_for_raspi.py** - Optimizes models for Raspberry Pi

3. **train_merged_dataset.py** - Generic training script (not specifically optimized for edge devices)

## Merged Dataset

The merged dataset contains 4 standardized weapon classes:
- `Knife`
- `Pistol`
- `weapon` (general category for various weapons)
- `rifle`

Two versions of the merged dataset are available:
- 80/10/10 split (train/val/test)
- 75/15/15 split (train/val/test)

## Training for Jetson Nano

```bash
# Basic usage with 80/10/10 dataset split
./train_for_jetson.py

# Using 75/15/15 dataset split
./train_for_jetson.py --dataset 75_15_15

# Custom configuration
./train_for_jetson.py --dataset 80_10_10 --epochs 150 --imgsz 256 --batch 8 --model yolov8n.pt
```

The script will prompt you to choose between:
1. Standard YOLOv8n with reduced image size (256x256)
2. Ultralight YOLOv8n model (even faster but less accurate)

The script will automatically export the trained model to formats optimized for Jetson Nano:
- ONNX (optimized for TensorRT)
- TFLite with INT8 quantization

## Training for Raspberry Pi

```bash
# Basic usage with 80/10/10 dataset split
./train_for_raspi.py

# Using 75/15/15 dataset split
./train_for_raspi.py --dataset 75_15_15

# Custom configuration
./train_for_raspi.py --dataset 80_10_10 --epochs 150 --imgsz 320 --batch 8 --model yolov8n.pt
```

The script will automatically export the trained model to formats optimized for Raspberry Pi:
- ONNX (general format)
- TFLite with INT8 quantization (good for Pi)
- OpenVINO (best performance on Pi 4)

## Command-Line Arguments

Both scripts accept the following arguments:

- `--dataset`: Dataset split to use (`80_10_10` or `75_15_15`)
- `--epochs`: Number of training epochs
- `--imgsz`: Image size for training (smaller is faster on edge devices)
- `--batch`: Batch size for training
- `--model`: Base model to use (yolov8n.pt recommended for edge devices)

## Deployment Recommendations

### Jetson Nano
- Use 256x256 image size
- Use TensorRT for inference (convert ONNX model on the device)
- For maximum speed, use the ultralight model

### Raspberry Pi 4
- Use 320x320 image size
- Use OpenVINO for best performance
- TFLite is a good alternative with decent compatibility 