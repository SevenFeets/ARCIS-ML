# Transferring Your Weapon Detection Model to Another Laptop

This guide provides step-by-step instructions for transferring your trained weapon detection model to another laptop and running inference using the built-in webcam.

## What You Need to Transfer

1. **Model weights file**:
   - The trained model file: `runs/detect/train/weights/best.pt` (approximately 6 MB)

2. **Python scripts**:
   - `webcam_inference.py` (the script for running inference on a webcam)
   
3. **Supporting files** (if needed):
   - `requirements.txt` (for installing dependencies)

## Step-by-Step Transfer Guide

### 1. Copy Files to a USB Drive or Cloud Storage

Copy the following files to a USB drive or cloud storage (like Google Drive or OneDrive):

- `runs/detect/train/weights/best.pt`
- `webcam_inference.py`
- `requirements.txt`

### 2. Set Up the Other Laptop

#### If Python is not installed:
1. Download and install Python 3.10 or newer from [python.org](https://www.python.org/downloads/)
2. Make sure to check "Add Python to PATH" during installation

#### Install Required Packages:
1. Open a command prompt or terminal on the other laptop
2. Create a new folder for your project
3. Copy the files from your USB drive to this folder
4. Navigate to this folder in the command prompt/terminal
5. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   ```
6. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
7. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
   
   If you don't have a requirements.txt file, install the necessary packages:
   ```bash
   pip install ultralytics opencv-python numpy
   ```

### 3. Run the Model on the Webcam

1. Make sure the webcam is connected/enabled
2. In the command prompt/terminal with the virtual environment activated, run:
   ```bash
   python webcam_inference.py --model best.pt
   ```
3. Press 'q' to quit the application when finished

### Additional Options

The script supports several command-line options:

- `--camera NUM`: Select a specific camera (default: 0)
- `--conf-thres VALUE`: Set the confidence threshold (default: 0.25)
- `--save`: Save the output video
- `--device cpu/cuda`: Choose CPU or GPU for inference

Example with options:
```bash
python webcam_inference.py --model best.pt --camera 1 --conf-thres 0.4 --save
```

### Troubleshooting

1. **Camera not found**:
   - The script will automatically list available cameras
   - Try a different camera index (0, 1, 2, etc.) using the `--camera` parameter

2. **Model file not found**:
   - Make sure the path to the model file is correct
   - If in the same folder, use `--model best.pt`

3. **Missing packages**:
   - If you get a "module not found" error, install the missing package using pip:
   ```bash
   pip install [package-name]
   ```

4. **Performance issues**:
   - If inference is slow, try reducing the image size: `--imgsz 320`
   - On laptops without a good GPU, performance may be limited 