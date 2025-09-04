# MobileNet-SSD v2 for Raspberry Pi

This repository contains code and models for deploying MobileNet-SSD v2 object detection on Raspberry Pi devices. The models are optimized for edge deployment with TensorFlow Lite.

## Project Structure

- `mobilenet_ssd_model/` - Contains a custom-trained MobileNet-SSD v2 model for weapon detection
  - `model.h5` - Full Keras model
  - `model.tflite` - TensorFlow Lite model
  - `model_quantized.tflite` - Quantized TensorFlow Lite model
  - `class_names.txt` - List of class names
  - `raspi_inference.py` - Inference script for Raspberry Pi
  - `test_image.jpg` - Test image for inference
  - `README.md` - Instructions for using the model

- `create_mobilenet_ssd_model.py` - Script to create and export a MobileNet-SSD v2 model
- `download_pretrained_model.py` - Script to download a pre-trained MobileNet-SSD v2 model from TensorFlow Hub
- `train_forraspi_mobilenet.py` - Script to train a MobileNet-SSD v2 model on custom data

## Getting Started

### Option 1: Use the pre-built model

The `mobilenet_ssd_model` directory contains a pre-built MobileNet-SSD v2 model trained for weapon detection (knife, pistol, weapon, rifle). You can use this model directly on your Raspberry Pi.

```bash
# Test the model with the provided test image
python mobilenet_ssd_model/raspi_inference.py --image mobilenet_ssd_model/test_image.jpg

# Use the model with your Raspberry Pi camera
python mobilenet_ssd_model/raspi_inference.py
```

### Option 2: Download a pre-trained COCO model

If you want to detect the 90 classes from the COCO dataset (person, car, dog, etc.), you can download a pre-trained model:

```bash
# Download the pre-trained model
python download_pretrained_model.py

# Test the model with the provided test image
python pretrained_mobilenet_ssd/inference.py --image pretrained_mobilenet_ssd/test_image.jpg
```

### Option 3: Train your own model

If you have your own dataset in YOLO format, you can train a custom model:

```bash
# Train a model on your dataset
python train_forraspi_mobilenet.py --dataset 80_10_10 --epochs 100 --imgsz 320 --batch 8
```

## Deployment on Raspberry Pi

1. Transfer the model directory to your Raspberry Pi:
   ```bash
   scp -r mobilenet_ssd_model/ pi@your-raspberry-pi-ip:~/
   ```

2. Install dependencies on your Raspberry Pi:
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3-pip python3-opencv
   pip3 install tensorflow-lite numpy
   ```

3. Run the inference script:
   ```bash
   python3 mobilenet_ssd_model/raspi_inference.py
   ```

## Performance Optimization

For better performance on Raspberry Pi:

1. Use the quantized model:
   ```bash
   python3 mobilenet_ssd_model/raspi_inference.py --model model_quantized.tflite
   ```

2. Consider using a Coral USB Accelerator for even better performance.

3. Adjust the detection threshold to reduce false positives:
   ```bash
   python3 mobilenet_ssd_model/raspi_inference.py --threshold 0.6
   ```

## License

This project is provided for educational purposes only. Please ensure you comply with all applicable laws when using object detection for weapons or other restricted items. 