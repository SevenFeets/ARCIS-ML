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

def train_ssd_small(
    data_yaml_path: str,
    epochs: int = 5,
    imgsz: int = 320,
    batch_size: int = 4,
    learning_rate: float = 0.001,
    max_samples: int = 1000  # Limit the number of samples for faster testing
):
    """
    Train a smaller version of MobileNet-SSD v2 with a reduced dataset for testing
    """
    # Force garbage collection to free up memory
    gc.collect()
    
    # Print system information
    print("\nSystem Information:")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    
    # Create output directories
    output_dir = Path("runs/ssd_small")
    model_dir = output_dir / "models"
    os.makedirs(model_dir, exist_ok=True)
    
    # Load dataset information from YAML
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    num_classes = data_config['nc']
    class_names = data_config['names']
    
    # Calculate feature map sizes based on the MobileNetV2 architecture
    feature_map_sizes = [10, 20, 40]  # For 320x320 input
    total_boxes = sum([size * size * 3 for size in feature_map_sizes])  # 3 anchor boxes per cell = 6300
    
    print(f"\nLoaded dataset with {num_classes} classes: {class_names}")
    print(f"Using smaller dataset for testing (max {max_samples} samples)")
    print(f"Feature map sizes: {feature_map_sizes}")
    print(f"Total prediction boxes: {total_boxes}")
    
    # Create TF datasets from YOLO format with reduced sample count
    train_dataset, val_dataset = create_small_datasets(
        data_yaml_path, 
        imgsz=imgsz, 
        batch_size=batch_size,
        num_classes=num_classes,
        total_boxes=total_boxes,
        max_samples=max_samples
    )
    
    # Create a smaller SSD model
    model = create_ssd_small_model(num_classes, imgsz, feature_map_sizes)
    
    # Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Use simple losses for the small model
    model.compile(
        optimizer=optimizer,
        loss={
            'regression': tf.keras.losses.MeanSquaredError(),
            'classification': tf.keras.losses.CategoricalCrossentropy()
        },
        metrics={
            'regression': ['mse'],
            'classification': ['accuracy']
        }
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / "checkpoint-{epoch:02d}.h5"),
            save_best_only=True,
            monitor='loss',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=3,
            verbose=1
        )
    ]
    
    # Train model
    print("\nStarting training with small dataset...")
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
    export_model(model, imgsz, model_dir, class_names)
    
    print("\nTraining and export completed successfully!")

def create_small_datasets(data_yaml_path, imgsz=320, batch_size=4, num_classes=5, total_boxes=6300, max_samples=1000):
    """Create smaller TensorFlow datasets for testing"""
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
    
    # Get image files (limited number)
    train_image_paths = sorted(glob.glob(str(train_images_dir / '*.jpg'))[:max_samples] + 
                               glob.glob(str(train_images_dir / '*.jpeg'))[:max_samples//2] + 
                               glob.glob(str(train_images_dir / '*.png'))[:max_samples//2])
    
    val_image_paths = sorted(glob.glob(str(val_images_dir / '*.jpg'))[:max_samples//5] + 
                             glob.glob(str(val_images_dir / '*.jpeg'))[:max_samples//10] + 
                             glob.glob(str(val_images_dir / '*.png'))[:max_samples//10])
    
    # Limit the total number of images
    train_image_paths = train_image_paths[:max_samples]
    val_image_paths = val_image_paths[:max_samples//5]
    
    print(f"Using {len(train_image_paths)} training images")
    print(f"Using {len(val_image_paths)} validation images")
    
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
        
        # Create simple dummy boxes and classes for testing
        # In a real scenario, we would load actual annotations
        boxes = [[0.1, 0.1, 0.9, 0.9]] * total_boxes  # Simple boxes covering most of the image
        classes = [[0, 0, 0, 0, 0]] * total_boxes      # Background class for all
        
        # If label file exists, try to read it
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    labels = f.read().strip().split('\n')
                
                real_boxes = []
                real_classes = []
                
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
                            
                            real_boxes.append([xmin, ymin, xmax, ymax])
                            
                            # One-hot encode class
                            if class_id < num_classes:
                                class_onehot = [0] * num_classes
                                class_onehot[class_id] = 1
                                real_classes.append(class_onehot)
                
                # If we successfully parsed some boxes, use them
                if real_boxes:
                    # Use only a few real boxes and pad the rest
                    num_real = min(len(real_boxes), 10)  # Use at most 10 real boxes
                    boxes = real_boxes[:num_real] + [[0, 0, 0, 0]] * (total_boxes - num_real)
                    classes = real_classes[:num_real] + [[0] * num_classes] * (total_boxes - num_real)
            except:
                # If anything goes wrong, just use dummy data
                pass
        
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

def create_ssd_small_model(num_classes, imgsz=320, feature_map_sizes=None):
    """Create a smaller SSD model for testing"""
    # Use a smaller backbone for faster training
    input_layer = tf.keras.layers.Input(shape=(imgsz, imgsz, 3))
    
    # Simple convolutional backbone
    x = tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu')(input_layer)
    x = tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    feature_map1 = tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding='same', activation='relu')(x)  # 40x40
    feature_map2 = tf.keras.layers.Conv2D(256, kernel_size=3, strides=2, padding='same', activation='relu')(feature_map1)  # 20x20
    feature_map3 = tf.keras.layers.Conv2D(512, kernel_size=3, strides=2, padding='same', activation='relu')(feature_map2)  # 10x10
    
    # Collect feature maps
    feature_maps = [feature_map3, feature_map2, feature_map1]  # From smallest to largest
    
    # Calculate total prediction boxes
    if feature_map_sizes is None:
        feature_map_sizes = [10, 20, 40]  # For 320x320 input
    
    total_boxes = sum([size * size * 3 for size in feature_map_sizes])  # 3 anchor boxes per cell
    print(f"Model will output {total_boxes} prediction boxes")
    
    # Create SSD detection heads
    regression_layers = []
    classification_layers = []
    
    for i, feature in enumerate(feature_maps):
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
        inputs=input_layer,
        outputs=[regression_output, classification_output]
    )
    
    print(f"Created simplified SSD model with {len(model.layers)} layers")
    return model

def export_model(model, imgsz, model_dir, class_names):
    """Export the model to optimized formats for deployment"""
    try:
        # Save class names
        with open(model_dir / "class_names.txt", "w") as f:
            for name in class_names:
                f.write(f"{name}\n")
        
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
            for _ in range(10):
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
        
        print("\nExport completed successfully!")
        
    except Exception as e:
        print(f"\nExport error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train smaller SSD model for testing")
    parser.add_argument('--dataset', default='merged_ssd_dataset/data.yaml',
                       help='Path to the dataset YAML file')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=320,
                       help='Image size for training')
    parser.add_argument('--batch', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='Maximum number of samples to use')
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
        "learning_rate": args.lr,
        "max_samples": args.max_samples
    }
    
    print("\nTraining configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nStarting small SSD model training...")
    
    train_ssd_small(**config) 