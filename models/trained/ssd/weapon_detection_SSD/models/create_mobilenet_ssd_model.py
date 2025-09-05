#!/usr/bin/env python3
import os
import tensorflow as tf
import numpy as np
from pathlib import Path
import yaml
import gc

def create_mobilenet_ssd_model(num_classes, imgsz=320):
    """Create MobileNet-SSD v2 model architecture"""
    # Base feature extractor: MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(imgsz, imgsz, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze early layers for transfer learning
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # Get the output of specific layers for feature extraction
    feature_maps = [
        base_model.get_layer('block_13_expand_relu').output,  # Higher level features
        base_model.get_layer('block_6_expand_relu').output,   # Mid level features
        base_model.get_layer('block_3_expand_relu').output    # Lower level features
    ]
    
    # Apply 1x1 convolution to reduce channel dimensions
    fpn_features = []
    for i, feature in enumerate(feature_maps):
        name = f'fpn_{i}'
        fpn_features.append(
            tf.keras.layers.Conv2D(256, kernel_size=1, padding='same', name=f'{name}_conv')(feature)
        )
    
    # Create detection heads (simplified for Raspberry Pi)
    regression_layers = []
    classification_layers = []
    
    for i, feature in enumerate(fpn_features):
        name = f'detection_{i}'
        
        # Add some convolutional layers for feature extraction
        x = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', name=f'{name}_conv1')(feature)
        x = tf.keras.layers.BatchNormalization(name=f'{name}_bn1')(x)
        x = tf.keras.layers.ReLU(name=f'{name}_relu1')(x)
        
        # Regression head (bounding boxes)
        regression = tf.keras.layers.Conv2D(4, kernel_size=3, padding='same', name=f'{name}_regression')(x)
        regression_reshape = tf.keras.layers.Reshape((-1, 4), name=f'{name}_regression_reshape')(regression)
        regression_layers.append(regression_reshape)
        
        # Classification head (class probabilities)
        classification = tf.keras.layers.Conv2D(num_classes, kernel_size=3, padding='same', name=f'{name}_classification')(x)
        classification_reshape = tf.keras.layers.Reshape((-1, num_classes), name=f'{name}_classification_reshape')(classification)
        classification_softmax = tf.keras.layers.Softmax(name=f'{name}_classification_softmax')(classification_reshape)
        classification_layers.append(classification_softmax)
    
    # Concatenate all predictions
    regression_output = tf.keras.layers.Concatenate(axis=1, name='regression')(regression_layers)
    classification_output = tf.keras.layers.Concatenate(axis=1, name='classification')(classification_layers)
    
    # Create model
    model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[regression_output, classification_output]
    )
    
    print(f"Created MobileNet-SSD v2 model with {len(model.layers)} layers")
    return model

def export_for_raspi(model, imgsz, output_dir, class_names):
    """Export the model to formats optimized for Raspberry Pi"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save class names
    with open(os.path.join(output_dir, "class_names.txt"), "w") as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    # Save the model directly (no need to create a new model)
    model.save(os.path.join(output_dir, "model.h5"))
    print(f"Saved model to {os.path.join(output_dir, 'model.h5')}")
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open(os.path.join(output_dir, "model.tflite"), "wb") as f:
        f.write(tflite_model)
    print(f"Saved TFLite model to {os.path.join(output_dir, 'model.tflite')}")
    
    # Convert to quantized TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    def representative_dataset_gen():
        for _ in range(100):
            data = np.random.rand(1, imgsz, imgsz, 3).astype(np.float32)
            yield [data]
    
    converter.representative_dataset = representative_dataset_gen
    
    try:
        quantized_tflite_model = converter.convert()
        
        with open(os.path.join(output_dir, "model_quantized.tflite"), "wb") as f:
            f.write(quantized_tflite_model)
        print(f"Saved quantized TFLite model to {os.path.join(output_dir, 'model_quantized.tflite')}")
    except Exception as e:
        print(f"Warning: Could not create quantized model: {e}")
        print("Continuing with standard TFLite model...")
    
    # Generate inference script
    generate_inference_script(output_dir, imgsz, class_names)
    
    print("\nExport completed successfully!")
    print("\nDeployment Recommendations:")
    print("1. Transfer the models directory to your Raspberry Pi")
    print("2. Install TensorFlow Lite runtime on your Raspberry Pi")
    print("3. Run the inference script with: python raspi_inference.py")

def generate_inference_script(output_dir, imgsz, class_names):
    """Generate a simple inference script for Raspberry Pi"""
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
'''
    
    with open(os.path.join(output_dir, "raspi_inference.py"), "w") as f:
        f.write(script)
    
    print(f"Created Raspberry Pi inference script: {os.path.join(output_dir, 'raspi_inference.py')}")

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
    return test_image_path

def main():
    # Define parameters
    imgsz = 320
    output_dir = "mobilenet_ssd_model"
    
    # Define class names based on your dataset
    class_names = ["Knife", "Pistol", "weapon", "rifle"]  # Replace with your actual classes
    num_classes = len(class_names)
    
    print(f"Creating MobileNet-SSD v2 model for {num_classes} classes: {class_names}")
    
    # Create model
    model = create_mobilenet_ssd_model(num_classes, imgsz)
    
    # Export model for Raspberry Pi
    export_for_raspi(model, imgsz, output_dir, class_names)
    
    # Create a test image
    test_image_path = create_test_image(output_dir)
    
    print(f"\nModel created and exported to {output_dir}/")
    print(f"You can test the model with: python {output_dir}/raspi_inference.py --image {test_image_path}")
    print("You can now transfer this directory to your Raspberry Pi and run the inference script")

if __name__ == "__main__":
    # Force garbage collection to free up memory
    gc.collect()
    
    # Print system information
    print("\nSystem Information:")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    
    try:
        import cv2
    except ImportError:
        print("OpenCV not found. Installing...")
        os.system("pip install opencv-python")
        import cv2
    
    main() 