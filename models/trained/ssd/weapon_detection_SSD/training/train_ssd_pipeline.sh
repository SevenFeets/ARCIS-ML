#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Training pipeline for MobileNet-SSD weapon detection model
# This script runs the entire pipeline:
# 1. Creates a merged dataset specifically formatted for SSD
# 2. Trains an SSD model on the merged dataset
# 3. Exports the trained model in multiple formats

echo "===== MobileNet-SSD Weapon Detection Training Pipeline ====="

# Step 1: Create merged SSD dataset
echo -e "\n[Step 1/3] Creating merged SSD dataset..."
python create_merged_ssd_dataset.py

# Check if dataset creation was successful
if [ ! -f "merged_ssd_dataset/data.yaml" ]; then
    echo "Error: Failed to create merged SSD dataset (data.yaml not found)!"
    exit 1
fi

echo "Dataset creation completed successfully."

# Step 2: Train SSD model
echo -e "\n[Step 2/3] Training SSD model..."
# Use smaller batch size (4) and fewer epochs (10) to avoid memory issues
python train_ssd_model.py --epochs 10 --batch 4 --imgsz 320

# Check if training was successful
if [ ! -d "runs/ssd_trained/models" ]; then
    echo "Error: Failed to train SSD model (models directory not found)!"
    exit 1
fi

if [ ! -f "runs/ssd_trained/models/model.h5" ]; then
    echo "Error: Failed to train SSD model (model.h5 not found)!"
    exit 1
fi

echo "Model training completed successfully."

# Step 3: Test inference
echo -e "\n[Step 3/3] Testing inference..."
if [ -f "runs/ssd_trained/models/inference.py" ]; then
    # Create a sample test image if one doesn't exist
    if [ ! -f "test_image.jpg" ]; then
        echo "Creating a test image..."
        # Try using ImageMagick if available, otherwise create a simple image with python
        if command -v convert >/dev/null 2>&1; then
            convert -size 640x480 xc:white -fill black -draw "rectangle 100,100 300,300" test_image.jpg
        else
            # Create a simple test image with Python
            python -c "
import numpy as np
import cv2
img = np.ones((480, 640, 3), dtype=np.uint8) * 255
cv2.rectangle(img, (100, 100), (300, 300), (0, 0, 0), -1)
cv2.imwrite('test_image.jpg', img)
print('Created test image using OpenCV')
"
        fi
    fi
    
    echo "Running inference test..."
    if [ -f "runs/ssd_trained/models/model.h5" ]; then
        cd runs/ssd_trained/models
        python inference.py --image ../../../test_image.jpg --output detection_result.jpg
        INFERENCE_STATUS=$?
        cd ../../..
        
        if [ $INFERENCE_STATUS -eq 0 ]; then
            echo "Inference test completed successfully. Check detection_result.jpg for results."
            # Copy the result to the main directory for easy viewing
            cp runs/ssd_trained/models/detection_result.jpg ./
        else
            echo "Warning: Inference test failed with status $INFERENCE_STATUS."
        fi
    else
        echo "Warning: Model file not found. Skipping inference test."
    fi
else
    echo "Warning: Inference script not found. Skipping inference test."
fi

echo -e "\n===== Training Pipeline Completed ====="
echo "The trained models are available in: runs/ssd_trained/models/"
echo "- model.h5: Full Keras model"
echo "- model.tflite: TensorFlow Lite model"
echo "- model_quantized.tflite: Quantized TensorFlow Lite model for edge devices"
echo "- inference.py: Script for running inference"
echo "- class_names.txt: List of class names"

echo -e "\nTo deploy to edge devices (like Raspberry Pi):"
echo "1. Copy the entire 'models' directory to your device"
echo "2. Run: python inference.py --image your_image.jpg --quantized"
echo "3. For best performance on edge devices, use the quantized model" 