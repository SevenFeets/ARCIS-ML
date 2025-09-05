#!/usr/bin/env python3
from pathlib import Path
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import shutil
import yaml
import cv2
import gc

def train_mobilenet_ssd_for_raspi(
    data_yaml_path: str,
    epochs: int = 100,
    imgsz: int = 320,  # Standard image size for Raspberry Pi
    batch_size: int = 16,
    model_type: str = "mobilenet_ssd_v2"  # Using MobileNet-SSD v2 for edge deployment
):
    """
    Train a MobileNet-SSD v2 model for TensorFlow Lite deployment on Raspberry Pi
    """
    # Force garbage collection to free up memory
    gc.collect()
    
    # Print system information
    print("\nSystem Information:")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    
    # Create output directories
    output_dir = Path("runs/mobilenet_ssd")
    model_dir = output_dir / "models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Load dataset information from YAML
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    num_classes = len(data_config['names'])
    class_names = data_config['names']
    
    print(f"\nLoaded dataset with {num_classes} classes: {class_names}")
    
    # Create TF dataset from YOLO format
    train_dataset, val_dataset = create_tf_datasets_from_yolo(
        data_yaml_path, 
        imgsz=imgsz, 
        batch_size=batch_size
    )
    
    # Create MobileNet-SSD v2 model
    model = create_mobilenet_ssd_model(num_classes, imgsz)
    
    # Train the model
    print("\nStarting training for MobileNet-SSD v2...")
    try:
        # Compile model with appropriate loss and optimizer
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'regression': smooth_l1_loss,
                'classification': tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            },
            loss_weights={'regression': 1.0, 'classification': 1.0},
            metrics=['accuracy']
        )
        
        # Define callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(model_dir / "best_model.h5"),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=str(output_dir / "logs")
            )
        ]
        
        # Train the model
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining completed!")
        print(f"Best model saved at: {model_dir}/best_model.h5")
        
        # Save model and export to TFLite
        model.save(str(model_dir / "final_model.h5"))
        
    except Exception as e:
        print(f"\nTraining error: {e}")
        print("Check if partial model weights were saved")
    
    # Export to optimized formats for Raspberry Pi deployment
    export_for_raspi(model, imgsz, model_dir, class_names)

def smooth_l1_loss(y_true, y_pred):
    """Smooth L1 loss for bounding box regression"""
    abs_diff = tf.abs(y_true - y_pred)
    smooth_l1 = tf.where(abs_diff < 1, 0.5 * abs_diff ** 2, abs_diff - 0.5)
    return tf.reduce_mean(smooth_l1, axis=-1)

def create_mobilenet_ssd_model(num_classes, imgsz):
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
    
    # SSD detection head
    x = base_model.output
    
    # Feature Pyramid Network (FPN) for multi-scale detection
    fpn_features = []
    
    # Use different feature maps from MobileNetV2
    feature_maps = [
        base_model.get_layer('block_13_expand_relu').output,  # Higher level features
        base_model.get_layer('block_6_expand_relu').output,   # Mid level features
        base_model.get_layer('block_3_expand_relu').output    # Lower level features
    ]
    
    for i, feature in enumerate(feature_maps):
        name = f'fpn_{i}'
        fpn_features.append(
            tf.keras.layers.Conv2D(256, kernel_size=1, padding='same', name=f'{name}_conv')(feature)
        )
    
    # Detection heads for each feature map
    regression_layers = []
    classification_layers = []
    
    for i, feature in enumerate(fpn_features):
        name = f'detection_{i}'
        
        # Regression head (bounding boxes)
        regression = tf.keras.layers.Conv2D(4, kernel_size=3, padding='same', name=f'{name}_regression')(feature)
        regression = tf.keras.layers.Reshape((-1, 4), name=f'{name}_regression_reshape')(regression)
        regression_layers.append(regression)
        
        # Classification head (class probabilities)
        classification = tf.keras.layers.Conv2D(num_classes, kernel_size=3, padding='same', name=f'{name}_classification')(feature)
        classification = tf.keras.layers.Reshape((-1, num_classes), name=f'{name}_classification_reshape')(classification)
        classification = tf.keras.layers.Softmax(name=f'{name}_classification_softmax')(classification)
        classification_layers.append(classification)
    
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

def create_tf_datasets_from_yolo(data_yaml_path, imgsz=320, batch_size=16):
    """Convert YOLO format dataset to TensorFlow dataset"""
    # Load dataset configuration
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    train_images_dir = Path(data_config['train']).parent / 'images' / 'train'
    val_images_dir = Path(data_config['val']).parent / 'images' / 'val'
    
    train_labels_dir = Path(data_config['train']).parent / 'labels' / 'train'
    val_labels_dir = Path(data_config['val']).parent / 'labels' / 'val'
    
    num_classes = len(data_config['names'])
    
    # Create dataset generators
    def process_yolo_sample(image_path, label_path, num_classes, imgsz):
        # Read and resize image
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [imgsz, imgsz])
        img = img / 255.0  # Normalize to [0,1]
        
        # Read YOLO format labels and convert to SSD format
        try:
            with open(label_path.numpy().decode('utf-8'), 'r') as f:
                labels = f.read().strip().split('\n')
            
            boxes = []
            classes = []
            
            for label in labels:
                if label.strip():
                    parts = label.strip().split()
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Convert YOLO format (x_center, y_center, width, height) to 
                    # absolute coordinates (xmin, ymin, xmax, ymax)
                    xmin = max(0, x_center - width/2)
                    ymin = max(0, y_center - height/2)
                    xmax = min(1, x_center + width/2)
                    ymax = min(1, y_center + height/2)
                    
                    boxes.append([xmin, ymin, xmax, ymax])
                    
                    # One-hot encode class
                    class_onehot = [0] * num_classes
                    class_onehot[class_id] = 1
                    classes.append(class_onehot)
            
            # Pad boxes and classes if needed (for fixed-size output)
            max_boxes = 100  # Maximum number of boxes per image
            
            if not boxes:
                # No objects in image, create dummy data
                boxes = [[0, 0, 0, 0]] * max_boxes
                classes = [[0] * num_classes] * max_boxes
            else:
                # Pad to max_boxes
                if len(boxes) > max_boxes:
                    boxes = boxes[:max_boxes]
                    classes = classes[:max_boxes]
                else:
                    padding_boxes = [[0, 0, 0, 0]] * (max_boxes - len(boxes))
                    padding_classes = [[0] * num_classes] * (max_boxes - len(classes))
                    boxes.extend(padding_boxes)
                    classes.extend(padding_classes)
            
            boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
            classes = tf.convert_to_tensor(classes, dtype=tf.float32)
            
            return img, (boxes, classes)
            
        except Exception as e:
            print(f"Error processing label {label_path}: {e}")
            # Return dummy data on error
            dummy_boxes = tf.zeros((max_boxes, 4), dtype=tf.float32)
            dummy_classes = tf.zeros((max_boxes, num_classes), dtype=tf.float32)
            return img, (dummy_boxes, dummy_classes)
    
    # Create training dataset
    train_image_paths = sorted([str(path) for path in train_images_dir.glob('*.jpg')])
    train_label_paths = [str(train_labels_dir / f"{Path(img_path).stem}.txt") for img_path in train_image_paths]
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_label_paths))
    
    # Use py_function to wrap the process_yolo_sample function
    def tf_process_yolo_sample(img_path, label_path):
        def process_fn(img_p, label_p):
            result = process_yolo_sample(img_p, label_p, num_classes, imgsz)
            return result[0], result[1][0], result[1][1]  # img, boxes, classes
        
        # Use py_function with flat outputs
        img, boxes, classes = tf.py_function(
            process_fn,
            [img_path, label_path],
            [tf.float32, tf.float32, tf.float32]
        )
        
        # Set shapes since py_function doesn't preserve them
        img.set_shape([imgsz, imgsz, 3])
        boxes.set_shape([100, 4])
        classes.set_shape([100, num_classes])
        
        return img, (boxes, classes)
    
    train_dataset = train_dataset.map(
        tf_process_yolo_sample,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Create validation dataset
    val_image_paths = sorted([str(path) for path in val_images_dir.glob('*.jpg')])
    val_label_paths = [str(val_labels_dir / f"{Path(img_path).stem}.txt") for img_path in val_image_paths]
    
    val_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths, val_label_paths))
    val_dataset = val_dataset.map(
        tf_process_yolo_sample,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    print(f"Created TF datasets from YOLO format: {len(train_image_paths)} training, {len(val_image_paths)} validation images")
    
    return train_dataset, val_dataset

def export_for_raspi(model, imgsz, model_dir, class_names):
    """Export the model to formats optimized for Raspberry Pi"""
    try:
        # Save class names
        with open(model_dir / "class_names.txt", "w") as f:
            for name in class_names:
                f.write(f"{name}\n")
        
        # 1. Export to TensorFlow SavedModel format
        saved_model_dir = model_dir / "saved_model"
        tf.saved_model.save(model, str(saved_model_dir))
        print(f"\nSaved model exported to: {saved_model_dir}")
        
        # 2. Export to TFLite format (standard)
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        tflite_model = converter.convert()
        
        tflite_file = model_dir / "model.tflite"
        with open(tflite_file, 'wb') as f:
            f.write(tflite_model)
        print(f"TFLite model exported to: {tflite_file}")
        
        # 3. Export to TFLite format (quantized for better performance on Raspberry Pi)
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quantized_tflite_model = converter.convert()
        
        quantized_file = model_dir / "model_quantized.tflite"
        with open(quantized_file, 'wb') as f:
            f.write(quantized_tflite_model)
        print(f"Quantized TFLite model exported to: {quantized_file}")
        
        # 4. Export to TFLite format with edge TPU support (for Coral USB Accelerator)
        try:
            converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.representative_dataset = lambda: representative_dataset_gen(imgsz)
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            converter.experimental_new_converter = True
            
            edge_tpu_model = converter.convert()
            
            edge_tpu_file = model_dir / "model_edgetpu.tflite"
            with open(edge_tpu_file, 'wb') as f:
                f.write(edge_tpu_model)
            print(f"Edge TPU compatible model exported to: {edge_tpu_file}")
        except Exception as e:
            print(f"Edge TPU export error (not critical): {e}")
        
        # Generate inference script for Raspberry Pi
        generate_raspi_inference_script(model_dir, imgsz, class_names)
        
        print("\nAll exports completed successfully!")
        print("\nDeployment Recommendations:")
        print("1. Transfer the 'models' directory to your Raspberry Pi")
        print("2. Install TensorFlow Lite runtime on your Raspberry Pi")
        print("3. Run the inference script with: python raspi_inference.py")
        print("4. For best performance, use the quantized model")
        print("5. If you have a Coral USB Accelerator, use the Edge TPU model")
        
    except Exception as e:
        print(f"\nExport error: {e}")

def representative_dataset_gen(imgsz):
    """Generate representative dataset for quantization"""
    # This is a dummy implementation - in a real scenario, you would use actual data
    for _ in range(100):
        data = np.random.rand(1, imgsz, imgsz, 3).astype(np.float32)
        yield [data]

def generate_raspi_inference_script(model_dir, imgsz, class_names):
    """Generate TFLite inference script for Raspberry Pi"""
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
    # Note: This assumes the model outputs bounding boxes and class probabilities
    # You may need to adjust based on your specific model outputs
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    
    results = []
    for i in range(len(boxes)):
        if np.max(classes[i]) > threshold:
            class_id = np.argmax(classes[i])
            box = boxes[i]
            
            # Convert normalized coordinates to pixel coordinates
            ymin, xmin, ymax, xmax = box
            xmin = int(xmin * image.shape[1])
            xmax = int(xmax * image.shape[1])
            ymin = int(ymin * image.shape[0])
            ymax = int(ymax * image.shape[0])
            
            results.append({
                'box': (xmin, ymin, xmax, ymax),
                'class_id': class_id,
                'score': np.max(classes[i])
            })
    
    return results

def main():
    parser = argparse.ArgumentParser(description='TFLite Object Detection on Raspberry Pi')
    parser.add_argument('--model', default='model_quantized.tflite',
                        help='Name of the TFLite model file')
    parser.add_argument('--labels', default='class_names.txt',
                        help='Name of the labels file')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection threshold')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index (default: 0)')
    parser.add_argument('--width', type=int, default=640,
                        help='Camera width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                        help='Camera height (default: 480)')
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
    
    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
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
                label = f"{labels[class_id]}: {score:.2f}"
                
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
    
    with open(model_dir / "raspi_inference.py", "w") as f:
        f.write(script)
    
    print(f"\nCreated Raspberry Pi inference script: {model_dir}/raspi_inference.py")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train MobileNet-SSD v2 for TensorFlow Lite on Raspberry Pi")
    parser.add_argument('--dataset', choices=['80_10_10', '75_15_15'], default=None,
                       help='Which dataset split to use (80/10/10 or 75/15/15)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=320,
                       help='Image size for training (smaller is faster on Raspberry Pi)')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--model', type=str, default='mobilenet_ssd_v2',
                       help='Model type (currently only mobilenet_ssd_v2 supported)')
    parser.add_argument('--generate-code-only', action='store_true',
                       help='Only generate inference code without training')
    parser.add_argument('--interactive', action='store_true',
                       help='Use interactive menu for choosing options')
    args = parser.parse_args()
    
    # If we only want to generate code
    if args.generate_code_only:
        print("Generating inference code only (no training)...")
        model_dir = Path("runs/mobilenet_ssd/models")
        os.makedirs(model_dir, exist_ok=True)
        
        # Load class names from the dataset
        if args.dataset == '75_15_15':
            data_yaml = 'merged_dataset_75_15/data.yaml'
        else:
            data_yaml = 'merged_dataset/data.yaml'
        
        try:
            with open(data_yaml, 'r') as f:
                data_config = yaml.safe_load(f)
                class_names = data_config['names']
        except:
            class_names = ["class1", "class2", "class3"]  # Default if can't load
        
        generate_raspi_inference_script(model_dir, 320, class_names)
        sys.exit(0)
    
    # Interactive mode to choose dataset and other options
    if args.interactive or args.dataset is None:
        print("\n=== MobileNet-SSD v2 Training for Raspberry Pi ===")
        
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
    
    # Configure training parameters optimized for Raspberry Pi
    config = {
        "data_yaml_path": data_yaml,
        "epochs": args.epochs,
        "imgsz": args.imgsz,  # Smaller size for Raspberry Pi
        "batch_size": args.batch,
        "model_type": args.model
    }
    
    print("\nTraining configuration for MobileNet-SSD v2 on Raspberry Pi:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Confirm training
    if args.interactive or args.dataset is None:
        confirm = input("\nStart training with these settings? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Training cancelled.")
            sys.exit(0)
    
    print("\nStarting training with MobileNet-SSD v2 for Raspberry Pi deployment...")
    
    train_mobilenet_ssd_for_raspi(**config) 