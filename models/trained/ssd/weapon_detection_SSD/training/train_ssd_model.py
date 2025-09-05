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
import time
import glob

def train_ssd_model(
    data_yaml_path: str,
    epochs: int = 50,
    imgsz: int = 320,
    batch_size: int = 8,
    learning_rate: float = 0.001
):
    """
    Train a MobileNet-SSD v2 model with the merged dataset specifically formatted for SSD
    """
    # Force garbage collection to free up memory
    gc.collect()
    
    # Print system information
    print("\nSystem Information:")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    
    # Create output directories
    output_dir = Path("runs/ssd_trained")
    model_dir = output_dir / "models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Load dataset information from YAML
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    num_classes = data_config['nc']
    class_names = data_config['names']
    
    # Get SSD-specific configuration if available
    ssd_config = data_config.get('ssd_config', {})
    img_size = ssd_config.get('image_size', imgsz)
    
    # Calculate feature map sizes based on the MobileNetV2 architecture
    # For a 320x320 input, MobileNetV2 produces feature maps of sizes 10x10, 20x20, 40x40
    # Each location on each feature map has 3 anchor boxes
    feature_map_sizes = [10, 20, 40]  # For 320x320 input
    total_boxes = sum([size * size * 3 for size in feature_map_sizes])  # 3 anchor boxes per cell = 6300
    
    print(f"\nLoaded dataset with {num_classes} classes: {class_names}")
    print(f"Using SSD configuration: image_size={img_size}, feature_map_sizes={feature_map_sizes}")
    print(f"Total prediction boxes: {total_boxes}")
    
    # Create TF datasets from YOLO format
    train_dataset, val_dataset = create_tf_datasets_from_yolo(
        data_yaml_path, 
        imgsz=img_size, 
        batch_size=batch_size,
        num_classes=num_classes,
        total_boxes=total_boxes  # Pass the correct number of boxes
    )
    
    # Create MobileNet-SSD model
    model = create_ssd_model(num_classes, img_size, feature_map_sizes)
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Use custom SSD loss functions
    model.compile(
        optimizer=optimizer,
        loss={
            'regression': SSDBoxLoss(),
            'classification': SSDClassLoss(num_classes)
        },
        metrics={
            'regression': ['mse'],
            'classification': ['accuracy']
        }
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / "checkpoint-{epoch:02d}-{val_loss:.2f}.h5"),
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(output_dir / 'logs'),
            histogram_freq=1
        )
    ]
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(model_dir / "model.h5")
    print(f"\nSaved final model to {model_dir}/model.h5")
    
    # Export optimized models for deployment
    export_for_deployment(model, img_size, model_dir, class_names)
    
    print("\nTraining and export completed successfully!")

def create_tf_datasets_from_yolo(data_yaml_path, imgsz=320, batch_size=8, num_classes=5, total_boxes=6300):
    """Create TensorFlow datasets from YOLO format labels, formatted for SSD"""
    # Load dataset configuration
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get paths
    dataset_path = Path(data_yaml_path).parent
    train_images_dir = dataset_path / 'images' / 'train'
    val_images_dir = dataset_path / 'images' / 'val'
    train_labels_dir = dataset_path / 'labels' / 'train'
    val_labels_dir = dataset_path / 'labels' / 'val'
    
    # Check if directories exist
    if not train_images_dir.exists() or not val_images_dir.exists():
        print(f"Error: Image directories not found")
        sys.exit(1)
    
    if not train_labels_dir.exists() or not val_labels_dir.exists():
        print(f"Error: Label directories not found")
        sys.exit(1)
    
    # Get image files
    train_image_paths = sorted(glob.glob(str(train_images_dir / '*.jpg')) + 
                              glob.glob(str(train_images_dir / '*.jpeg')) + 
                              glob.glob(str(train_images_dir / '*.png')))
    
    val_image_paths = sorted(glob.glob(str(val_images_dir / '*.jpg')) + 
                            glob.glob(str(val_images_dir / '*.jpeg')) + 
                            glob.glob(str(val_images_dir / '*.png')))
    
    print(f"Found {len(train_image_paths)} training images")
    print(f"Found {len(val_image_paths)} validation images")
    
    # Function to process YOLO format samples
    def process_yolo_sample(image_path, num_classes, imgsz, total_boxes):
        # Extract image name without extension
        image_name = os.path.basename(image_path).split('.')[0]
        label_path = os.path.join(os.path.dirname(image_path).replace('images', 'labels'), f"{image_name}.txt")
        
        # Read and resize image
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [imgsz, imgsz])
        img = img / 255.0  # Normalize to [0,1]
        
        # Read YOLO format labels
        boxes = []
        classes = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                labels = f.read().strip().split('\n')
            
            for label in labels:
                if label.strip():
                    parts = label.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Convert YOLO format to [xmin, ymin, xmax, ymax]
                        xmin = max(0, x_center - width/2)
                        ymin = max(0, y_center - height/2)
                        xmax = min(1, x_center + width/2)
                        ymax = min(1, y_center + height/2)
                        
                        boxes.append([xmin, ymin, xmax, ymax])
                        
                        # One-hot encode class (handle if class_id is out of range)
                        if class_id < num_classes:
                            class_onehot = [0] * num_classes
                            class_onehot[class_id] = 1
                            classes.append(class_onehot)
                        else:
                            # Skip invalid class IDs
                            boxes.pop()  # Remove the corresponding box
        
        # Pad or truncate to match expected SSD output size
        if not boxes:
            # No objects in image, create dummy data
            boxes = [[0, 0, 0, 0]] * total_boxes
            classes = [[0] * num_classes] * total_boxes
        else:
            # Pad or truncate to total_boxes
            if len(boxes) > total_boxes:
                boxes = boxes[:total_boxes]
                classes = classes[:total_boxes]
            else:
                padding_boxes = [[0, 0, 0, 0]] * (total_boxes - len(boxes))
                padding_classes = [[0] * num_classes] * (total_boxes - len(classes))
                boxes.extend(padding_boxes)
                classes.extend(padding_classes)
        
        boxes = tf.convert_to_tensor(boxes, dtype=tf.float32)
        classes = tf.convert_to_tensor(classes, dtype=tf.float32)
        
        return img, (boxes, classes)
    
    # Create TensorFlow datasets
    def create_dataset(image_paths, num_classes, imgsz, batch_size, total_boxes):
        def generator():
            for image_path in image_paths:
                yield process_yolo_sample(image_path, num_classes, imgsz, total_boxes)
        
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(imgsz, imgsz, 3), dtype=tf.float32),
                (
                    tf.TensorSpec(shape=(total_boxes, 4), dtype=tf.float32),
                    tf.TensorSpec(shape=(total_boxes, num_classes), dtype=tf.float32)
                )
            )
        )
        
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Create datasets
    train_dataset = create_dataset(train_image_paths, num_classes, imgsz, batch_size, total_boxes)
    val_dataset = create_dataset(val_image_paths, num_classes, imgsz, batch_size, total_boxes)
    
    return train_dataset, val_dataset

def create_ssd_model(num_classes, imgsz=320, feature_map_sizes=None):
    """Create a MobileNet-SSD v2 model with correct output dimensions for SSD architecture"""
    # Base feature extractor: MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(imgsz, imgsz, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze early layers for transfer learning
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # Get feature maps at different scales
    feature_maps = [
        base_model.get_layer('block_13_expand_relu').output,  # Higher level features
        base_model.get_layer('block_6_expand_relu').output,   # Mid level features
        base_model.get_layer('block_3_expand_relu').output    # Lower level features
    ]
    
    # SSD Feature Pyramid Network
    fpn_features = []
    for i, feature in enumerate(feature_maps):
        name = f'fpn_{i}'
        x = tf.keras.layers.Conv2D(256, kernel_size=1, padding='same', name=f'{name}_conv1')(feature)
        x = tf.keras.layers.BatchNormalization(name=f'{name}_bn1')(x)
        x = tf.keras.layers.ReLU(name=f'{name}_relu1')(x)
        
        x = tf.keras.layers.Conv2D(256, kernel_size=3, padding='same', name=f'{name}_conv2')(x)
        x = tf.keras.layers.BatchNormalization(name=f'{name}_bn2')(x)
        x = tf.keras.layers.ReLU(name=f'{name}_relu2')(x)
        
        fpn_features.append(x)
    
    # Calculate total prediction boxes
    if feature_map_sizes is None:
        feature_map_sizes = [10, 20, 40]  # For 320x320 input
    
    total_boxes = sum([size * size * 3 for size in feature_map_sizes])  # 3 anchor boxes per cell
    print(f"Model will output {total_boxes} prediction boxes")
    
    # Create SSD detection heads
    regression_layers = []
    classification_layers = []
    
    for i, feature in enumerate(fpn_features):
        name = f'ssd_head_{i}'
        
        # Regression head for bounding boxes
        regression = tf.keras.layers.Conv2D(
            12,  # 3 anchors * 4 coordinates (xmin, ymin, xmax, ymax)
            kernel_size=3,
            padding='same',
            name=f'{name}_regression'
        )(feature)
        regression_reshape = tf.keras.layers.Reshape(
            (-1, 4),  # Reshape to [batch_size, num_boxes, 4]
            name=f'{name}_regression_reshape'
        )(regression)
        regression_layers.append(regression_reshape)
        
        # Classification head for class probabilities
        classification = tf.keras.layers.Conv2D(
            3 * num_classes,  # 3 anchors * num_classes
            kernel_size=3,
            padding='same',
            name=f'{name}_classification'
        )(feature)
        classification_reshape = tf.keras.layers.Reshape(
            (-1, num_classes),  # Reshape to [batch_size, num_boxes, num_classes]
            name=f'{name}_classification_reshape'
        )(classification)
        classification_softmax = tf.keras.layers.Softmax(
            name=f'{name}_classification_softmax'
        )(classification_reshape)
        classification_layers.append(classification_softmax)
    
    # Concatenate predictions from all feature maps
    regression_output = tf.keras.layers.Concatenate(axis=1, name='regression')(regression_layers)
    classification_output = tf.keras.layers.Concatenate(axis=1, name='classification')(classification_layers)
    
    # Create model
    model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[regression_output, classification_output]
    )
    
    print(f"Created SSD model with {len(model.layers)} layers")
    return model

# Custom loss functions for SSD
class SSDBoxLoss(tf.keras.losses.Loss):
    """Custom loss function for SSD bounding box regression"""
    
    def __init__(self, delta=1.0, **kwargs):
        super(SSDBoxLoss, self).__init__(**kwargs)
        self.delta = delta
    
    def call(self, y_true, y_pred):
        # Only compute loss for boxes with actual targets (non-zero)
        # Sum across the box coordinates to find valid boxes
        mask = tf.reduce_sum(tf.abs(y_true), axis=-1) > 0
        mask = tf.cast(mask, dtype=tf.float32)
        
        # Number of positive boxes
        num_positive = tf.maximum(tf.reduce_sum(mask), 1.0)
        
        # Smooth L1 loss
        # Make sure shapes are compatible for broadcasting
        loss = tf.keras.losses.huber(y_true, y_pred, delta=self.delta)
        
        # Apply mask and normalize by number of positive boxes
        # Reshape mask for proper broadcasting
        mask_expanded = tf.expand_dims(mask, axis=-1)  # Shape: [batch_size, num_boxes, 1]
        loss = tf.reduce_sum(loss * mask_expanded) / num_positive
        return loss

class SSDClassLoss(tf.keras.losses.Loss):
    """Custom loss function for SSD classification"""
    
    def __init__(self, num_classes, alpha=1.0, **kwargs):
        super(SSDClassLoss, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        # Only compute loss for boxes with actual targets (non-zero)
        # Sum across the classes to find valid boxes
        mask = tf.reduce_sum(y_true, axis=-1) > 0
        mask = tf.cast(mask, dtype=tf.float32)
        
        # Number of positive boxes
        num_positive = tf.maximum(tf.reduce_sum(mask), 1.0)
        
        # Focal loss to handle class imbalance
        ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_weight = tf.pow(1.0 - pt, self.alpha)
        
        # Apply mask and focal weight, normalize by number of positive boxes
        loss = tf.reduce_sum(ce_loss * mask * focal_weight) / num_positive
        return loss

def export_for_deployment(model, imgsz, model_dir, class_names):
    """Export the model to optimized formats for deployment"""
    try:
        # Save class names
        with open(model_dir / "class_names.txt", "w") as f:
            for name in class_names:
                f.write(f"{name}\n")
        
        # Save Keras H5 model (already saved in main function)
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        tflite_file = model_dir / "model.tflite"
        with open(tflite_file, 'wb') as f:
            f.write(tflite_model)
        print(f"TFLite model exported to: {tflite_file}")
        
        # Create quantized TFLite model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        def representative_dataset_gen():
            for _ in range(100):
                data = np.random.rand(1, imgsz, imgsz, 3).astype(np.float32)
                yield [data]
                
        converter.representative_dataset = representative_dataset_gen
        
        try:
            quantized_tflite_model = converter.convert()
            
            quantized_file = model_dir / "model_quantized.tflite"
            with open(quantized_file, 'wb') as f:
                f.write(quantized_tflite_model)
            print(f"Quantized TFLite model exported to: {quantized_file}")
        except Exception as e:
            print(f"Warning: Could not create quantized model: {e}")
            print("Continuing with standard TFLite model...")
        
        # Create a test inference script
        create_inference_script(model_dir, imgsz, class_names)
        
        print("\nExport completed successfully!")
        
    except Exception as e:
        print(f"\nExport error: {e}")
        import traceback
        traceback.print_exc()

def create_inference_script(model_dir, imgsz, class_names):
    """Create a simple inference script for testing the model"""
    script = '''#!/usr/bin/env python3
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

def load_tflite_model(tflite_path):
    """Load TFLite model"""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
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

def detect_objects(model, img_batch, class_names, conf_threshold=0.5):
    """Detect objects using the model"""
    # Run inference
    boxes, class_probs = model.predict(img_batch)
    
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
                'class_name': class_names[class_id],
                'confidence': float(confidence),
                'box': [float(xmin), float(ymin), float(xmax), float(ymax)]
            })
    
    return results

def detect_objects_tflite(interpreter, img_batch, class_names, conf_threshold=0.5):
    """Detect objects using TFLite model"""
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_batch)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensors
    boxes = interpreter.get_tensor(output_details[0]['index'])
    class_probs = interpreter.get_tensor(output_details[1]['index'])
    
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
                'class_name': class_names[class_id],
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
    parser.add_argument('--image', required=True, help='Path to the input image')
    parser.add_argument('--model', default='model.h5', help='Path to the model file')
    parser.add_argument('--tflite', action='store_true', help='Use TFLite model')
    parser.add_argument('--quantized', action='store_true', help='Use quantized TFLite model')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--output', default='detection_result.jpg', help='Path to the output image')
    args = parser.parse_args()
    
    # Get model directory (where this script is located)
    model_dir = Path(__file__).parent
    
    # Load class names
    class_names = load_class_names(model_dir / 'class_names.txt')
    print(f"Loaded {len(class_names)} classes: {class_names}")
    
    # Load the model
    if args.tflite or args.quantized:
        # Use TFLite model
        if args.quantized:
            model_path = model_dir / 'model_quantized.tflite'
        else:
            model_path = model_dir / 'model.tflite'
        
        print(f"Loading TFLite model: {model_path}")
        interpreter = load_tflite_model(str(model_path))
        use_tflite = True
    else:
        # Use full Keras model
        model_path = model_dir / args.model
        print(f"Loading model: {model_path}")
        model = load_model(str(model_path))
        use_tflite = False
    
    # Load and preprocess image
    print(f"Processing image: {args.image}")
    img, img_batch = preprocess_image(args.image)
    
    # Detect objects
    if use_tflite:
        results = detect_objects_tflite(interpreter, img_batch, class_names, args.threshold)
    else:
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
'''
    
    with open(model_dir / "inference.py", "w") as f:
        f.write(script)
    
    print(f"Created inference script: {model_dir}/inference.py")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train SSD model on merged dataset")
    parser.add_argument('--dataset', default='merged_ssd_dataset/data.yaml',
                       help='Path to the dataset YAML file')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=320,
                       help='Image size for training')
    parser.add_argument('--batch', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    args = parser.parse_args()
    
    # Check if the data.yaml file exists
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file {args.dataset} not found!")
        print("Please run create_merged_ssd_dataset.py first to create the merged dataset.")
        sys.exit(1)
    
    print(f"Using data configuration from: {args.dataset}")
    
    # Configure training parameters
    config = {
        "data_yaml_path": args.dataset,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch_size": args.batch,
        "learning_rate": args.lr
    }
    
    print("\nTraining configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nStarting SSD model training...")
    
    train_ssd_model(**config) 