#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
import cv2
import time
import argparse
from pathlib import Path

def load_labels(label_path):
    """Load class labels"""
    with open(label_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Resize image to expected shape [1, height, width, 3]
    input_shape = input_details[0]['shape']
    input_height, input_width = input_shape[1], input_shape[2]
    img_resized = cv2.resize(image, (input_width, input_height))
    input_data = np.expand_dims(img_resized, axis=0)
    
    # Normalize pixel values if using a floating model
    if input_details[0]['dtype'] == np.float32:
        input_data = (np.float32(input_data) / 255.0)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensors
    # First output is bounding boxes, second is class probabilities
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    
    results = []
    for i in range(min(100, len(boxes))):  # Limit to first 100 detections
        # Get class with highest probability
        class_id = np.argmax(classes[i])
        score = classes[i][class_id]
        
        if score > threshold:
            # Get bounding box
            box = boxes[i]
            
            # Convert normalized coordinates to pixel coordinates
            xmin = max(0, int(box[0] * image.shape[1]))
            ymin = max(0, int(box[1] * image.shape[0]))
            xmax = min(image.shape[1], int(box[2] * image.shape[1]))
            ymax = min(image.shape[0], int(box[3] * image.shape[0]))
            
            results.append({
                'box': (xmin, ymin, xmax, ymax),
                'class_id': int(class_id),
                'score': float(score)
            })
    
    return results

def main():
    parser = argparse.ArgumentParser(description='TFLite Object Detection on Raspberry Pi')
    parser.add_argument('--model', default='model.tflite',
                      help='Name of the TFLite model file')
    parser.add_argument('--labels', default='class_names.txt',
                      help='Name of the labels file')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='Detection threshold')
    parser.add_argument('--image', default=None,
                      help='Path to image file (if not provided, will use camera)')
    parser.add_argument('--camera', type=int, default=0,
                      help='Camera index (default: 0)')
    args = parser.parse_args()
    
    # Get path to current directory
    current_dir = Path(__file__).parent
    
    # Load TFLite model and labels
    model_path = current_dir / args.model
    labels_path = current_dir / args.labels
    
    print(f"Loading model: {model_path}")
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    
    print(f"Loading labels: {labels_path}")
    labels = load_labels(str(labels_path))
    
    if args.image:
        # Process a single image
        print(f"Processing image: {args.image}")
        image = cv2.imread(args.image)
        if image is None:
            print(f"Error: Could not load image {args.image}")
            return
        
        # Run detection
        start_time = time.time()
        results = detect_objects(interpreter, image, args.threshold)
        inference_time = time.time() - start_time
        
        print(f"Inference time: {inference_time*1000:.1f}ms")
        print(f"Found {len(results)} objects")
        
        # Draw results
        for obj in results:
            # Draw bounding box
            box = obj['box']
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            
            # Draw label
            class_id = obj['class_id']
            score = obj['score']
            if class_id < len(labels):
                label = f"{labels[class_id]}: {score:.2f}"
            else:
                label = f"Class {class_id}: {score:.2f}"
            
            cv2.putText(image, label, (box[0], box[1] - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save output image
        output_path = "detection_result.jpg"
        cv2.imwrite(output_path, image)
        print(f"Detection result saved to {output_path}")
        
        # Display image (if running in GUI environment)
        try:
            cv2.imshow('Object Detection', image)
            print("Press any key to exit...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except:
            print("Could not display image (no GUI environment)")
    else:
        # Use camera
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Starting detection...")
        
        try:
            while True:
                start_time = time.time()
                
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    break
                
                # Detect objects
                results = detect_objects(interpreter, frame, args.threshold)
                
                # Draw results
                for obj in results:
                    # Draw bounding box
                    box = obj['box']
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                    
                    # Draw label
                    class_id = obj['class_id']
                    score = obj['score']
                    if class_id < len(labels):
                        label = f"{labels[class_id]}: {score:.2f}"
                    else:
                        label = f"Class {class_id}: {score:.2f}"
                    
                    cv2.putText(frame, label, (box[0], box[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Calculate FPS
                fps = 1.0 / (time.time() - start_time)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Display frame
                cv2.imshow('Object Detection', frame)
                
                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            # Clean up
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
