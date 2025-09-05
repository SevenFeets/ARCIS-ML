#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Complete pipeline script for weapon detection model
# This script runs the entire process from dataset creation to inference

echo "===== Weapon Detection Pipeline ====="
echo "This script will run the complete pipeline:"
echo "1. Create merged SSD dataset"
echo "2. Train SSD model"
echo "3. Run inference on test images"

# Ask for confirmation
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Operation cancelled."
    exit 1
fi

# Step 1: Dataset preparation
echo -e "\n[Step 1/3] Creating merged SSD dataset..."
cd datasets
python create_merged_ssd_dataset.py
cd ..

# Check if dataset creation was successful
if [ ! -d "datasets/merged_ssd_dataset/images" ]; then
    echo "Error: Failed to create merged SSD dataset!"
    exit 1
fi

# Step 2: Training
echo -e "\n[Step 2/3] Training SSD model..."
cd training
bash train_ssd_pipeline.sh
cd ..

# Check if training was successful
if [ ! -f "training/runs/ssd_trained/models/model.h5" ]; then
    echo "Error: Training failed or model not found!"
    exit 1
fi

# Step 3: Inference
echo -e "\n[Step 3/3] Running inference..."
cd inference
python test_inference.py --image test_images/test_image.jpg --model ../training/runs/ssd_trained/models/model.h5 --threshold 0.3

# Also try TFLite model
echo -e "\nRunning TFLite inference..."
python inference_tflite.py --image test_images/test_image.jpg --model ../training/runs/ssd_trained/models/model.tflite --threshold 0.3
cd ..

echo -e "\n===== Pipeline completed successfully! ====="
echo "Detection results are available in the inference directory."
echo "Trained models are available in training/runs/ssd_trained/models/" 