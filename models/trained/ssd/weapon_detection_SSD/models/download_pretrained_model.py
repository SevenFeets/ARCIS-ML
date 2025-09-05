#!/usr/bin/env python3
import os
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from pathlib import Path

def download_model():
    """Download a pre-trained MobileNet SSD v2 model from TensorFlow Hub"""
    print("Downloading pre-trained MobileNet SSD v2 model from TensorFlow Hub...")
    
    # Create output directory
    output_dir = "pretrained_mobilenet_ssd"
    os.makedirs(output_dir, exist_ok=True)
    
    # Download model from TensorFlow Hub
    model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
    model = hub.load(model_url)
    
    # Save model
    tf.saved_model.save(model, output_dir)
    print(f"Model saved to {output_dir}")
    
    # Create class names file (COCO dataset classes)
    coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    with open(os.path.join(output_dir, "coco_classes.txt"), "w") as f:
        for class_name in coco_classes:
            f.write(f"{class_name}\n")
    
    # Convert to TFLite
    print("Converting to TFLite format...")
    converter = tf.lite.TFLiteConverter.from_saved_model(output_dir)
    tflite_model = converter.convert()
    
    with open(os.path.join(output_dir, "model.tflite"), "wb") as f:
        f.write(tflite_model)
    
    # Convert to quantized TFLite
    print("Converting to quantized TFLite format...")
    converter = tf.lite.TFLiteConverter.from_saved_model(output_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    def representative_dataset_gen():
        for _ in range(100):
            data = np.random.rand(1, 320, 320, 3).astype(np.float32)
            yield [data]
    
    converter.representative_dataset = representative_dataset_gen
    
    try:
        quantized_tflite_model = converter.convert()
        
        with open(os.path.join(output_dir, "model_quantized.tflite"), "wb") as f:
            f.write(quantized_tflite_model)
    except Exception as e:
        print(f"Warning: Could not create quantized model: {e}")
        print("Continuing with standard TFLite model...")
    
    # Create a test image
    create_test_image(output_dir)
    
    # Copy the inference script
    create_inference_script(output_dir)
    
    print("\nPre-trained model download complete!")
    print(f"Model files are in the '{output_dir}' directory")
    print(f"You can test the model with: python {output_dir}/inference.py --image {output_dir}/test_image.jpg")

def create_test_image(output_dir):
    """Create a test image for inference testing"""
    # Create a simple test image
    img = np.zeros((320, 320, 3), dtype=np.uint8)
    
    # Draw some shapes
    cv2.rectangle(img, (50, 50), (150, 150), (0, 255, 0), 2)
    cv2.rectangle(img, (200, 100), (250, 200), (0, 0, 255), 2)
    cv2.circle(img, (100, 250), 30, (255, 0, 0), -1)
    
    # Save the test image
    test_image_path = os.path.join(output_dir, "test_image.jpg")
    cv2.imwrite(test_image_path, img)
    print(f"Created test image: {test_image_path}")

def create_inference_script(output_dir):
    """Create an inference script for the pre-trained model"""
    script = '''#!/usr/bin/env python3
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
    
    # Get output tensors - SSD MobileNet has 4 output tensors
    # Output details depend on the specific model
    # Typically: boxes, classes, scores, num_detections
    boxes = None
    classes = None
    scores = None
    count = None
    
    # Find the right output tensors
    for i, output in enumerate(output_details):
        tensor = interpreter.get_tensor(output['index'])[0]
        if output['name'].find('location') >= 0 or output['name'].find('box') >= 0:
            boxes = tensor
        elif output['name'].find('class') >= 0:
            classes = tensor
        elif output['name'].find('score') >= 0:
            scores = tensor
        elif output['name'].find('num_detections') >= 0 or output['name'].find('count') >= 0:
            count = int(tensor) if tensor.dtype == np.float32 else tensor
    
    # If we couldn't identify the outputs by name, try by shape and type
    if boxes is None and len(output_details) >= 4:
        # Try to identify outputs by their typical shapes
        for tensor in [interpreter.get_tensor(out['index'])[0] for out in output_details]:
            if len(tensor.shape) == 2 and tensor.shape[1] == 4:  # boxes: [N, 4]
                boxes = tensor
            elif len(tensor.shape) == 1 and tensor.dtype == np.float32:  # scores: [N]
                scores = tensor
            elif len(tensor.shape) == 1 and tensor.dtype == np.int64:  # classes: [N]
                classes = tensor
            elif len(tensor.shape) == 0:  # num_detections: scalar
                count = int(tensor)
    
    # If we still don't have what we need, use default output order
    if boxes is None and len(output_details) >= 4:
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
        count = int(interpreter.get_tensor(output_details[3]['index'])[0])
    
    # If we still don't have scores or count, make some assumptions
    if scores is None:
        for i, output in enumerate(output_details):
            if len(output['shape']) >= 2 and output['shape'][-1] > 10:  # Assume this is the class scores
                scores = np.max(interpreter.get_tensor(output['index'])[0], axis=1)
                classes = np.argmax(interpreter.get_tensor(output['index'])[0], axis=1)
                break
    
    if count is None:
        count = len(scores) if scores is not None else 0
    
    results = []
    for i in range(min(100, count) if count is not None else min(100, len(scores))):
        if scores is not None and scores[i] >= threshold:
            # Get bounding box
            box = boxes[i]
            
            # The model might output normalized coordinates
            ymin, xmin, ymax, xmax = box
            
            # Convert normalized coordinates to pixel coordinates
            xmin = max(0, int(xmin * image.shape[1]))
            ymin = max(0, int(ymin * image.shape[0]))
            xmax = min(image.shape[1], int(xmax * image.shape[1]))
            ymax = min(image.shape[0], int(ymax * image.shape[0]))
            
            # Get class
            class_id = int(classes[i])
            
            results.append({
                'box': (xmin, ymin, xmax, ymax),
                'class_id': class_id,
                'score': float(scores[i])
            })
    
    return results

def main():
    parser = argparse.ArgumentParser(description='TFLite Object Detection')
    parser.add_argument('--model', default='model.tflite',
                      help='Name of the TFLite model file')
    parser.add_argument('--labels', default='coco_classes.txt',
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
'''
    
    with open(os.path.join(output_dir, "inference.py"), "w") as f:
        f.write(script)
    
    # Make the script executable
    os.chmod(os.path.join(output_dir, "inference.py"), 0o755)
    
    print(f"Created inference script: {os.path.join(output_dir, 'inference.py')}")

if __name__ == "__main__":
    # Check if tensorflow_hub is installed
    try:
        import tensorflow_hub as hub
    except ImportError:
        print("TensorFlow Hub not found. Installing...")
        os.system("pip install tensorflow-hub")
        import tensorflow_hub as hub
    
    # Check if OpenCV is installed
    try:
        import cv2
    except ImportError:
        print("OpenCV not found. Installing...")
        os.system("pip install opencv-python")
        import cv2
    
    download_model() 