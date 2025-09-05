#!/usr/bin/env python3
from pathlib import Path
import sys
import os
import argparse

# Add the Utilities directory to the path so we can import train_yolo
utilities_path = Path(__file__).parent / "Utilities"
sys.path.append(str(utilities_path))

from ultralytics import YOLO
import torch
import gc

def train_for_cloud_raspi_system(
    data_yaml_path: str,
    epochs: int = 100,
    imgsz: int = 320,  # Standard image size for cloud inference
    batch_size: int = 16,
    model_type: str = "yolov8n.pt"  # Using nano model for balance of speed and accuracy
):
    """
    Train a YOLOv8 model for a Raspberry Pi + Cloud system where:
    - Raspberry Pi captures video and sends frames to cloud
    - Google Cloud VM runs inference with a T4 GPU
    - Results are sent back to Raspberry Pi for alerting
    """
    # Force garbage collection to free up memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Print detailed GPU information
    print("\nGPU Information:")
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print("GPU device:", device_name)
        print("GPU memory:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
        device = 0  # Use GPU for training
    else:
        print("WARNING: No GPU detected! Training will be very slow on CPU.")
        device = 'cpu'
    
    # Initialize model
    print("\nInitializing YOLO model...")
    model = YOLO(model_type)
    
    # Train the model with settings optimized for cloud inference
    print("\nStarting training for Raspberry Pi + Cloud system...")
    try:
        model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,  # Using standard image size for better accuracy
            batch=batch_size,
            device=device,
            patience=50,
            save=True,
            plots=True,
            workers=4,
            cache=False,
            amp=True,  # Mixed precision for faster training on T4 GPU
            optimizer="AdamW",  # Better optimizer for cloud inference
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            close_mosaic=10,
            fraction=1.0,
            rect=False,
            cos_lr=True,
            verbose=True,
            exist_ok=True,
            nbs=64,
            overlap_mask=False,
            val=True,
            deterministic=False,
            project="runs/detect",
            name="cloud_raspi"  # Save in a different folder
        )
        
        print("\nTraining completed!")
        print(f"Best model saved at: runs/detect/cloud_raspi/weights/best.pt")
        
    except Exception as e:
        print(f"\nTraining error: {e}")
        print("Model weights are still saved at runs/detect/cloud_raspi/weights/")
    
    # Export to optimized formats for cloud deployment
    export_for_cloud_raspi(model, imgsz)

def export_for_cloud_raspi(model, imgsz):
    """Export the model to formats optimized for Cloud VM with T4 GPU"""
    try:
        # 1. Export to ONNX (standard format)
        print("\nExporting model to ONNX format...")
        onnx_path = model.export(format="onnx", imgsz=imgsz, simplify=True)
        print(f"ONNX model exported to: {onnx_path}")
        
        # 2. Export to TensorRT for T4 GPU acceleration
        print("\nExporting model to TensorRT format...")
        try:
            tensorrt_path = model.export(format="engine", imgsz=imgsz, half=True)
            print(f"TensorRT engine exported to: {tensorrt_path}")
        except Exception as e:
            print(f"TensorRT export error: {e}")
            print("TensorRT export should be done on the target Google Cloud VM with the same GPU")
        
        # 3. Export PyTorch model (best for direct deployment)
        print("\nExporting optimized PyTorch model...")
        pytorch_path = model.export(format="torchscript", imgsz=imgsz)
        print(f"TorchScript model exported to: {pytorch_path}")
        
        print("\nAll exports completed successfully!")
        print("\nDeployment Recommendations:")
        print("1. Upload 'best.pt' to your Google Cloud VM")
        print("2. Install Flask and ultralytics on your VM")
        print("3. Set up the Flask server as per your requirements")
        print("4. For maximum performance, consider using TensorRT on the T4 GPU")
        
        # Generate Flask server code
        generate_flask_server_code()
        generate_raspi_client_code()
        
    except Exception as e:
        print(f"\nExport error: {e}")

def generate_flask_server_code():
    """Generate optimized Flask server code for the Google Cloud VM"""
    server_code = '''#!/usr/bin/env python3
from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import time
import torch

app = Flask(__name__)

# Load the YOLOv8 model
model = YOLO('best.pt')

# Configure for maximum performance on T4 GPU
if torch.cuda.is_available():
    model.to('cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: No GPU detected! Inference will be slower.")

# Store the last frame processing time for performance monitoring
last_process_time = 0

@app.route('/infer', methods=['POST'])
def infer():
    global last_process_time
    start_time = time.time()
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    # Get image from request
    file = request.files['image']
    image_data = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    
    # Run inference
    results = model(image, conf=0.25)  # Lower confidence threshold for weapon detection
    
    # Process results
    detections = []
    alert = False  # Flag to indicate if an alert should be triggered
    
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls]
            
            # Check if the detected object is a weapon class
            if class_name in ['Knife', 'Pistol', 'weapon', 'rifle']:
                alert = True
            
            xyxy = box.xyxy[0].tolist()
            detections.append({
                'class': class_name,
                'confidence': round(conf, 3),
                'box': list(map(int, xyxy))
            })
    
    # Calculate processing time
    process_time = time.time() - start_time
    last_process_time = process_time
    
    # Return results
    return jsonify({
        'detections': detections,
        'alert': alert,
        'process_time': round(process_time, 3)
    })

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'status': 'online',
        'gpu': str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'None',
        'last_process_time': round(last_process_time, 3)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, threaded=True)
'''
    with open("flask_server.py", "w") as f:
        f.write(server_code)
    
    print("\nCreated flask_server.py - Upload this to your Google Cloud VM")
    print("Run with: python flask_server.py")

def generate_raspi_client_code():
    """Generate Raspberry Pi client code for video streaming and alerting"""
    client_code = '''#!/usr/bin/env python3
import cv2
import requests
import json
import numpy as np
import time
import pygame
import threading
from io import BytesIO
import argparse

# Initialize pygame for audio
pygame.mixer.init()

# Load alarm sound
ALARM_SOUND_PATH = "/home/barvaz/Desktop/alarm.mp3"
alarm_sound = pygame.mixer.Sound(ALARM_SOUND_PATH)

# Global variables
alarm_playing = False
alarm_lock = threading.Lock()

def play_alarm():
    global alarm_playing
    with alarm_lock:
        if not alarm_playing:
            alarm_playing = True
            alarm_sound.play(-1)  # Play in loop

def stop_alarm():
    global alarm_playing
    with alarm_lock:
        if alarm_playing:
            alarm_sound.stop()
            alarm_playing = False

def process_frame(frame, server_url):
    # Send frame to server and process the response
    # Encode frame as JPEG
    _, img_encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    # Send to server
    try:
        files = {'image': BytesIO(img_encoded.tobytes())}
        response = requests.post(f"{server_url}/infer", files=files, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            
            # Check if we should alert
            if result.get('alert', False):
                play_alarm()
            else:
                stop_alarm()
            
            # Draw bounding boxes on the frame
            for detection in result.get('detections', []):
                box = detection['box']
                label = f"{detection['class']} {detection['confidence']:.2f}"
                
                # Draw rectangle
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                
                # Draw label background
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (box[0], box[1] - text_size[1] - 5), 
                              (box[0] + text_size[0], box[1]), (0, 255, 0), -1)
                
                # Draw text
                cv2.putText(frame, label, (box[0], box[1] - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Add server processing time
            process_time = result.get('process_time', 0)
            cv2.putText(frame, f"Server: {process_time*1000:.0f}ms", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
        else:
            print(f"Error: Server returned status code {response.status_code}")
            cv2.putText(frame, "Server Error", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}")
        cv2.putText(frame, "Connection Error", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return frame

def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi client for weapon detection")
    parser.add_argument("--server", type=str, default="http://34.0.85.5:8000",
                       help="Server URL (default: http://34.0.85.5:8000)")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=640,
                       help="Frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480,
                       help="Frame height (default: 480)")
    parser.add_argument("--fps", type=int, default=15,
                       help="Target FPS (default: 15)")
    args = parser.parse_args()
    
    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print(f"Connected to camera. Sending frames to {args.server}")
    
    # Set target time between frames based on FPS
    frame_delay = 1.0 / args.fps
    
    try:
        # Check server status
        try:
            response = requests.get(f"{args.server}/status", timeout=5)
            if response.status_code == 200:
                print(f"Server status: {response.json()}")
            else:
                print(f"Warning: Server status check failed with code {response.status_code}")
        except requests.exceptions.RequestException:
            print("Warning: Could not connect to server for status check")
    
        while True:
            start_time = time.time()
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Process frame (send to server and draw results)
            processed_frame = process_frame(frame.copy(), args.server)
            
            # Display frame
            cv2.imshow('Weapon Detection', processed_frame)
            
            # Calculate time to wait to maintain target FPS
            elapsed = time.time() - start_time
            wait_time = max(1, int((frame_delay - elapsed) * 1000))
            
            # Exit on 'q' key press
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break
    
    finally:
        # Clean up
        stop_alarm()
        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()

if __name__ == "__main__":
    main()
'''
    with open("raspi_client.py", "w") as f:
        f.write(client_code)
    
    print("\nCreated raspi_client.py - Transfer this to your Raspberry Pi")
    print("Run with: python raspi_client.py")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train YOLOv8 for Raspberry Pi + Cloud system")
    parser.add_argument('--dataset', choices=['80_10_10', '75_15_15'], default=None,
                       help='Which dataset split to use (80/10/10 or 75/15/15)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size for training (standard size for cloud inference)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Base model to use (yolov8n.pt recommended for balance of speed/accuracy)')
    parser.add_argument('--generate-code-only', action='store_true',
                       help='Only generate client-server code without training')
    parser.add_argument('--interactive', action='store_true',
                       help='Use interactive menu for choosing options')
    args = parser.parse_args()
    
    # If we only want to generate code
    if args.generate_code_only:
        print("Generating client-server code only (no training)...")
        generate_flask_server_code()
        generate_raspi_client_code()
        sys.exit(0)
    
    # Interactive mode to choose dataset and other options
    if args.interactive or args.dataset is None:
        print("\n=== Raspberry Pi + Cloud Training System ===")
        
        # Dataset selection
        print("\nAvailable datasets:")
        print("1. 80/10/10 split (80% train, 10% validation, 10% test)")
        print("2. 75/15/15 split (75% train, 15% validation, 15% test)")
        
        dataset_choice = input("\nSelect dataset (1/2): ").strip()
        if dataset_choice == "2":
            data_yaml = 'merged_dataset_75_15/data.yaml'
            dataset_name = "75/15/15 split"
        else:
            data_yaml = 'merged_dataset/data.yaml'
            dataset_name = "80/10/10 split"
        
        print(f"\nSelected dataset: {dataset_name}")
        
        # Additional options
        print("\nTraining options (press Enter to use defaults):")
        
        epochs_input = input(f"Number of epochs [{args.epochs}]: ").strip()
        if epochs_input and epochs_input.isdigit():
            args.epochs = int(epochs_input)
        
        imgsz_input = input(f"Image size [{args.imgsz}]: ").strip()
        if imgsz_input and imgsz_input.isdigit():
            args.imgsz = int(imgsz_input)
        
        batch_input = input(f"Batch size [{args.batch}]: ").strip()
        if batch_input and batch_input.isdigit():
            args.batch = int(batch_input)
        
        model_input = input(f"Model type [{args.model}]: ").strip()
        if model_input:
            args.model = model_input
    else:
        # Select the appropriate dataset based on the argument
        if args.dataset == '80_10_10':
            data_yaml = 'merged_dataset/data.yaml'
            dataset_name = "80/10/10 split"
        else:
            data_yaml = 'merged_dataset_75_15/data.yaml'
            dataset_name = "75/15/15 split"
    
    # Check if the data.yaml file exists
    if not os.path.exists(data_yaml):
        print(f"Error: Dataset file {data_yaml} not found!")
        print("Please make sure the merged datasets have been created.")
        sys.exit(1)
    
    print(f"Using data configuration from: {data_yaml}")
    
    # Configure training parameters optimized for Cloud-Raspberry Pi system
    config = {
        "data_yaml_path": data_yaml,
        "epochs": args.epochs,
        "imgsz": args.imgsz,  # Standard size for cloud inference
        "batch_size": args.batch,
        "model_type": args.model
    }
    
    print("\nTraining configuration for Raspberry Pi + Cloud system:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Confirm training
    if args.interactive or args.dataset is None:
        confirm = input("\nStart training with these settings? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Training cancelled.")
            sys.exit(0)
    
    print("\nStarting training with settings optimized for Raspberry Pi + Cloud deployment...")
    
    # Set HSA_OVERRIDE_GFX_VERSION for AMD GPU compatibility
    os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
    
    train_for_cloud_raspi_system(**config) 