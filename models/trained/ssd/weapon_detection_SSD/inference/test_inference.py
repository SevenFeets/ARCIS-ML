#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
import cv2
import argparse
from pathlib import Path

def load_model(model_path):
    """Load the saved model"""
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

def load_class_names(labels_path):
    """Load class names"""
    with open(labels_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def preprocess_image(image_path, target_size=(320, 320)):
    """Preprocess image for model input"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    img_resized = cv2.resize(img, target_size)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img, img_batch

def detect_objects(model, img_batch, class_names, conf_threshold=0.5):
    """Detect objects using the model"""
    # Run inference
    outputs = model.predict(img_batch)
    boxes, class_probs = outputs
    
    # Process results
    results = []
    
    # Get the first image's predictions
    image_boxes = boxes[0]
    image_class_probs = class_probs[0]
    
    for i in range(len(image_boxes)):
        # Get confidence score and class ID
        class_id = np.argmax(image_class_probs[i])
        confidence = image_class_probs[i][class_id]
        
        # Filter low confidence detections
        if confidence > conf_threshold:
            # Get coordinates
            xmin, ymin, xmax, ymax = image_boxes[i]
            
            # Ensure coordinates are within image bounds
            if xmin < 0 or ymin < 0 or xmax > 1 or ymax > 1:
                continue
                
            if xmin >= xmax or ymin >= ymax:
                continue
            
            results.append({
                'class_id': int(class_id),
                'class_name': class_names[class_id] if class_id < len(class_names) else "unknown",
                'confidence': float(confidence),
                'box': [float(xmin), float(ymin), float(xmax), float(ymax)]
            })
    
    return results

def draw_results(image, results):
    """Draw detection results on the image"""
    h, w = image.shape[:2]
    
    for result in results:
        # Get coordinates
        xmin, ymin, xmax, ymax = result['box']
        
        # Convert to pixel coordinates
        xmin = int(xmin * w)
        ymin = int(ymin * h)
        xmax = int(xmax * w)
        ymax = int(ymax * h)
        
        # Draw bounding box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        
        # Draw label
        label = f"{result['class_name']}: {result['confidence']:.2f}"
        cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

def main():
    parser = argparse.ArgumentParser(description='Object Detection Inference')
    parser.add_argument('--image', default='test_images/test_image.jpg', help='Path to the input image')
    parser.add_argument('--model', default='runs/ssd_small/models/model.h5', help='Path to the model file')
    parser.add_argument('--threshold', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--output', default='detection_result.jpg', help='Path to the output image')
    args = parser.parse_args()
    
    # Get model directory
    model_dir = Path(args.model).parent
    
    # Load class names
    class_names = load_class_names(model_dir / 'class_names.txt')
    print(f"Loaded {len(class_names)} classes: {class_names}")
    
    # Load the model
    print(f"Loading model: {args.model}")
    model = load_model(args.model)
    
    # Handle cases where model.input is a list
    if isinstance(model.input, list):
        print(f"Model has multiple inputs with shapes: {[i.shape for i in model.input]}")
    else:
        print(f"Model input shape: {model.input.shape}")
    
    # Handle cases where model.outputs is a list
    print(f"Model output shapes: {[o.shape for o in model.outputs]}")
    
    # Load and preprocess image
    print(f"Processing image: {args.image}")
    img, img_batch = preprocess_image(args.image)
    
    # Detect objects
    results = detect_objects(model, img_batch, class_names, args.threshold)
    
    print(f"Found {len(results)} objects")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result['class_name']}: {result['confidence']:.2f}")
    
    # Draw results on image
    output_img = draw_results(img.copy(), results)
    
    # Save output image
    cv2.imwrite(args.output, output_img)
    print(f"Detection result saved to: {args.output}")
    
    # Try to display image
    try:
        cv2.imshow('Detection Result', output_img)
        print("Press any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("Could not display image (no GUI environment)")

if __name__ == '__main__':
    main() 