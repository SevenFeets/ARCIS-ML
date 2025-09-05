#!/usr/bin/env python3
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
