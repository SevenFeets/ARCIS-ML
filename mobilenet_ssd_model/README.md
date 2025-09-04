# MobileNet-SSD v2 Model for Raspberry Pi

This directory contains a MobileNet-SSD v2 object detection model optimized for deployment on Raspberry Pi devices. The model is pre-trained to detect weapons (knife, pistol, weapon, rifle).

## Contents

- `model.h5` - Full Keras model
- `model.tflite` - TensorFlow Lite model
- `model_quantized.tflite` - Quantized TensorFlow Lite model (optimized for Raspberry Pi)
- `class_names.txt` - List of class names
- `raspi_inference.py` - Inference script for Raspberry Pi
- `test_image.jpg` - Test image for inference

## Setup on Raspberry Pi

1. Transfer this entire directory to your Raspberry Pi

2. Install required dependencies:
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3-pip python3-opencv
   pip3 install tensorflow-lite numpy
   ```

3. For better performance with camera, you may want to install picamera:
   ```bash
   sudo apt-get install -y python3-picamera
   ```

## Running Inference

### Using an image file:
```bash
python3 raspi_inference.py --image test_image.jpg
```

### Using the Raspberry Pi camera:
```bash
python3 raspi_inference.py
```

### Using the quantized model (better performance):
```bash
python3 raspi_inference.py --model model_quantized.tflite
```

### Adjusting detection threshold:
```bash
python3 raspi_inference.py --threshold 0.6
```

## Performance Tips

1. The quantized model (`model_quantized.tflite`) will run much faster than the standard model.

2. For even better performance, consider using a Coral USB Accelerator with TensorFlow Lite.

3. Reducing the input resolution can improve performance at the cost of accuracy.

4. On Raspberry Pi 4 or newer, you can enable OpenGL acceleration:
   ```bash
   sudo apt-get install -y python3-opengl
   export DISPLAY=:0
   ```

## Troubleshooting

- If you encounter memory errors, try closing other applications or increasing swap space.
- If the camera doesn't work, ensure your camera is properly connected and enabled in raspi-config.
- For GPU acceleration issues, make sure you have the latest Raspberry Pi OS and drivers.

## License

This model is provided for educational purposes only. Please ensure you comply with all applicable laws when using object detection for weapons. 