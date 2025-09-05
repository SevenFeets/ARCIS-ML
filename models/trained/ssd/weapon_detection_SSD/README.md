# Weapon Detection System

A MobileNet-SSD v2 based weapon detection system optimized for edge devices like Raspberry Pi.

## Project Structure

```
weapon_detection/
├── datasets/             # Dataset preparation scripts and dataset links
│   ├── create_merged_ssd_dataset.py   # Script to create SSD-specific dataset
│   ├── merged_dataset -> ../../merged_dataset
│   ├── merged_dataset_75_15 -> ../../merged_dataset_75_15
│   └── merged_ssd_dataset -> ../../merged_ssd_dataset
│
├── models/               # Model architecture definitions
│   ├── create_mobilenet_ssd_model.py  # Script to create the SSD model
│   └── download_pretrained_model.py   # Script to download pretrained models
│
├── training/             # Training scripts and training output
│   ├── runs -> ../../runs            # Training results directory
│   ├── train_fixed_v2.py             # Fixed training script version 2
│   ├── train_ssd_model.py            # Main SSD model training script
│   ├── train_ssd_pipeline.sh         # Full training pipeline script
│   ├── train_ssd_small.py            # Training with reduced dataset for testing
│   ├── train_with_dataset.py         # Original training script
│   └── train_with_dataset_fixed.py   # Fixed training script
│
├── inference/            # Inference and model testing
│   ├── detection_result.jpg          # Sample detection result
│   ├── detection_result_tflite.jpg   # TFLite detection result
│   ├── inference_tflite.py           # TFLite inference script
│   ├── test_images -> ../../test_images  # Test images directory
│   ├── test_inference.py             # Inference testing script
│   └── tflite_info.py                # TFLite model information script
│
└── utils/                # Utility scripts and tools
```

## Getting Started

### 1. Dataset Preparation

```bash
cd datasets
python create_merged_ssd_dataset.py
```

This will create a merged dataset specifically formatted for SSD architecture.

### 2. Training

To train the full model:
```bash
cd ../training
bash train_ssd_pipeline.sh
```

For a faster test with a smaller dataset:
```bash
python train_ssd_small.py
```

### 3. Inference

To test the trained model:
```bash
cd ../inference
python test_inference.py --image test_images/test_image.jpg --model ../training/runs/ssd_trained/models/model.h5
```

For TFLite inference (for edge devices):
```bash
python inference_tflite.py --image test_images/test_image.jpg --model ../training/runs/ssd_trained/models/model.tflite
```

## Model Details

- Architecture: MobileNet-SSD v2
- Input size: 320x320
- Classes: pistol, rifle, knife, grenade, person
- Feature map sizes: 10x10, 20x20, 40x40
- Total prediction boxes: 6,300

## Deployment

For Raspberry Pi deployment:
1. Copy the `training/runs/ssd_trained/models` directory to your Raspberry Pi
2. Install TensorFlow Lite on Raspberry Pi
3. Run inference using the included `raspi_inference.py` script

## Overview

The project provides multiple approaches for creating and deploying weapon detection models:

1. **Full Dataset Pipeline**: Merges multiple weapon detection datasets and trains a complete SSD model
2. **Small Dataset Training**: Trains on a limited dataset for faster iteration and testing
3. **Model Export**: Exports trained models in multiple formats (Keras, TFLite, Quantized TFLite)

## Dataset

The merged dataset is created from 5 different weapon detection datasets, with classes harmonized into a unified structure:

- **pistol**: Handguns, pistols, and generic weapons
- **rifle**: Rifles, shotguns, submachine guns, and heavy weapons
- **knife**: Knives and bladed weapons
- **grenade**: Grenades and explosive devices
- **person**: Person detection (can be used for person-with-weapon detection)

## Training Options

### Full Pipeline Training

The full training pipeline processes all datasets and trains the complete model:

```bash
bash train_ssd_pipeline.sh
```

This script:
1. Creates the merged dataset from all source datasets
2. Trains the SSD model (10 epochs by default)
3. Exports the model in multiple formats
4. Tests inference on a sample image

### Small Dataset Training

For faster iteration or testing, you can use the small dataset training:

```bash
./train_ssd_small.py --max_samples 500 --epochs 2
```

Options:
- `--max_samples`: Number of samples to use (default: 1000)
- `--epochs`: Number of training epochs (default: 5)
- `--batch`: Batch size (default: 4)
- `--imgsz`: Image size (default: 320)

## Model Architecture

The model uses a MobileNet-SSD v2 architecture:

- **Base Network**: MobileNetV2 pre-trained on ImageNet
- **Detection Head**: SSD (Single Shot MultiBox Detector)
- **Input Size**: 320x320 pixels
- **Output**: 6300 prediction boxes with class probabilities for 5 weapon classes

## Exported Models

After training, the following models are available in the `runs/ssd_trained/models/` directory:

- `model.h5`: Full Keras model
- `model.tflite`: TensorFlow Lite model
- `model_quantized.tflite`: Quantized TensorFlow Lite model (optimized for edge devices)
- `class_names.txt`: List of class names
- `inference.py`: Inference script for testing

## Deployment on Edge Devices

### Raspberry Pi Deployment

1. Copy the entire `models` directory to the Raspberry Pi
2. Install the required dependencies:
   ```bash
   pip install tensorflow-lite numpy opencv-python
   ```
3. Run inference:
   ```bash
   cd models
   python inference.py --image test_image.jpg --quantized
   ```

### Performance Optimization

For best performance on edge devices:
- Use the quantized TFLite model
- Set appropriate confidence thresholds
- Consider reducing input image resolution

## Troubleshooting

If you encounter dimension mismatch issues:
- Ensure the dataset and model both use the same number of boxes (6300)
- Check that feature map sizes are calculated correctly (10x10, 20x20, 40x40)
- Use a smaller batch size if running out of memory

## SSD Architecture Advantages

The SSD (Single Shot MultiBox Detector) architecture is particularly well-suited for edge devices because:

1. **Single-stage detection**: Faster inference than two-stage detectors
2. **Multi-scale feature maps**: Better detection of objects at different sizes
3. **Mobile-optimized backbone**: MobileNetV2 is designed for mobile devices
4. **Quantization-friendly**: Works well with INT8 quantization for edge deployment

## Dimension Matching

This implementation fixes the dimension mismatch issues encountered in previous attempts by:

1. Calculating the correct feature map sizes based on the input image size
2. Ensuring the model's output shape matches the training targets (25,200 boxes)
3. Correctly formatting the dataset with padded boxes to match SSD architecture
4. Adding SSD-specific metadata to the dataset configuration

## Performance Optimization

For optimal performance on Raspberry Pi:

- Use the quantized TensorFlow Lite model
- Reduce input image size to 320×320 or smaller if needed
- Consider using Coral USB Accelerator for enhanced performance

## References

- MobileNetV2: https://arxiv.org/abs/1801.04381
- SSD: https://arxiv.org/abs/1512.02325
- TensorFlow Lite: https://www.tensorflow.org/lite 