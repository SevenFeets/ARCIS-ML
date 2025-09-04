#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
import cv2
import argparse
from pathlib import Path

def load_tflite_model(tflite_path):
    """Load TFLite model"""
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    return interpreter

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

def detect_objects_tflite(interpreter, img_batch, class_names, conf_threshold=0.5):
    """Detect objects using TFLite model"""
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Print model details
    print(f"Input details: {input_details}")
    print(f"Output details: {output_details}")
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_batch)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensors - from the TFLite info:
    # Output 0: StatefulPartitionedCall_1:1 = [1, 6300, 5] (class probabilities)
    # Output 1: StatefulPartitionedCall_1:0 = [1, 6300, 4] (bounding boxes)
    class_probs = interpreter.get_tensor(output_details[0]['index'])
    boxes = interpreter.get_tensor(output_details[1]['index'])
    
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
    parser = argparse.ArgumentParser(description='TFLite Object Detection Inference')
    parser.add_argument('--image', default='test_images/test_image.jpg', help='Path to the input image')
    parser.add_argument('--model', default='runs/ssd_small/models/model.tflite', help='Path to the TFLite model file')
    parser.add_argument('--quantized', action='store_true', help='Use quantized model')
    parser.add_argument('--threshold', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--output', default='detection_result_tflite.jpg', help='Path to the output image')
    args = parser.parse_args()
    
    # Use quantized model if specified
    if args.quantized:
        model_path = Path(args.model).parent / "model_quantized.tflite"
    else:
        model_path = args.model
    
    # Get model directory
    model_dir = Path(model_path).parent
    
    # Load class names
    class_names = load_class_names(model_dir / 'class_names.txt')
    print(f"Loaded {len(class_names)} classes: {class_names}")
    
    # Load the model
    print(f"Loading TFLite model: {model_path}")
    interpreter = load_tflite_model(model_path)
    
    # Load and preprocess image
    print(f"Processing image: {args.image}")
    img, img_batch = preprocess_image(args.image)
    
    # Detect objects
    results = detect_objects_tflite(interpreter, img_batch, class_names, args.threshold)
    
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