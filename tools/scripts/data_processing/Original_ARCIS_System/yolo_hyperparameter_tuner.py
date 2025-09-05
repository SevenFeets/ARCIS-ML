#!/usr/bin/env python3
"""
YOLO Hyperparameter Tuning for ARCIS Weapon Detection
Comprehensive parameter tuning interface for optimizing YOLO performance
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch

# YOLO Hyperparameter Presets
HYPERPARAMETER_PRESETS = {
    "default": {
        "name": "Default YOLO Settings",
        "description": "Standard YOLO parameters for general use",
        "params": {
            "lr0": 0.01,
            "lrf": 0.1,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
            "pose": 12.0,
            "kobj": 1.0,
            "label_smoothing": 0.0,
            "nbs": 64,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.0,
            "copy_paste": 0.0
        }
    },
    "weapon_optimized": {
        "name": "Weapon Detection Optimized",
        "description": "Optimized for small weapon detection with high precision",
        "params": {
            "lr0": 0.008,  # Slightly lower learning rate for stability
            "lrf": 0.05,   # Lower final learning rate
            "momentum": 0.95,  # Higher momentum for stability
            "weight_decay": 0.001,  # Higher weight decay to prevent overfitting
            "warmup_epochs": 5,  # More warmup epochs
            "warmup_momentum": 0.9,
            "warmup_bias_lr": 0.05,
            "box": 10.0,  # Higher box loss weight (important for weapon localization)
            "cls": 0.8,   # Higher classification loss weight
            "dfl": 2.0,   # Higher distribution focal loss
            "label_smoothing": 0.1,  # Add label smoothing for better generalization
            "hsv_h": 0.01,  # Reduced color augmentation (weapons have consistent colors)
            "hsv_s": 0.5,
            "hsv_v": 0.3,
            "degrees": 5.0,  # Small rotation augmentation
            "translate": 0.05,  # Reduced translation (weapons usually centered)
            "scale": 0.3,   # Reduced scale variation
            "fliplr": 0.3,  # Reduced horizontal flip (weapon orientation matters)
            "mosaic": 0.8,  # Slightly reduced mosaic
            "mixup": 0.1,   # Add mixup for better generalization
        }
    },
    "jetson_optimized": {
        "name": "Jetson Nano Optimized",
        "description": "Optimized for Jetson Nano deployment with speed focus",
        "params": {
            "lr0": 0.012,  # Higher learning rate for faster convergence
            "lrf": 0.15,
            "momentum": 0.9,
            "weight_decay": 0.0003,  # Lower weight decay for faster training
            "warmup_epochs": 2,  # Fewer warmup epochs
            "warmup_momentum": 0.7,
            "warmup_bias_lr": 0.15,
            "box": 6.0,   # Balanced loss weights
            "cls": 0.4,
            "dfl": 1.0,
            "label_smoothing": 0.05,
            "hsv_h": 0.02,  # Moderate augmentation
            "hsv_s": 0.6,
            "hsv_v": 0.4,
            "degrees": 10.0,
            "translate": 0.15,
            "scale": 0.6,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.0,  # No mixup for speed
        }
    },
    "high_accuracy": {
        "name": "High Accuracy (Slow Training)",
        "description": "Maximum accuracy settings for research/development",
        "params": {
            "lr0": 0.005,  # Lower learning rate for careful learning
            "lrf": 0.01,   # Very low final learning rate
            "momentum": 0.98,  # Very high momentum
            "weight_decay": 0.002,  # Higher regularization
            "warmup_epochs": 10,  # Extended warmup
            "warmup_momentum": 0.95,
            "warmup_bias_lr": 0.01,
            "box": 15.0,  # Very high box loss weight
            "cls": 1.0,   # High classification weight
            "dfl": 3.0,   # High distribution focal loss
            "label_smoothing": 0.15,  # Strong label smoothing
            "hsv_h": 0.005,  # Minimal color augmentation
            "hsv_s": 0.3,
            "hsv_v": 0.2,
            "degrees": 2.0,  # Minimal rotation
            "translate": 0.02,  # Minimal translation
            "scale": 0.2,   # Minimal scale variation
            "fliplr": 0.2,  # Minimal horizontal flip
            "mosaic": 0.5,  # Reduced mosaic for cleaner training
            "mixup": 0.2,   # Moderate mixup
            "copy_paste": 0.1,  # Add copy-paste augmentation
        }
    }
}

def display_parameter_info():
    """Display comprehensive information about YOLO parameters"""
    print("=== YOLO HYPERPARAMETER GUIDE ===\n")
    
    print("🎯 LEARNING RATE PARAMETERS:")
    print("  lr0: Initial learning rate (0.001-0.1)")
    print("    - Higher: Faster learning, risk of instability")
    print("    - Lower: Stable learning, slower convergence")
    print("  lrf: Final learning rate (lr0 * lrf)")
    print("    - Controls learning rate decay")
    
    print("\n⚡ OPTIMIZATION PARAMETERS:")
    print("  momentum: SGD momentum (0.8-0.99)")
    print("    - Higher: Better convergence, more stable")
    print("  weight_decay: L2 regularization (0.0001-0.01)")
    print("    - Higher: Prevents overfitting, may reduce accuracy")
    
    print("\n🔥 WARMUP PARAMETERS:")
    print("  warmup_epochs: Gradual learning rate increase (1-10)")
    print("    - More epochs: Stable start, slower initial training")
    print("  warmup_momentum: Initial momentum during warmup")
    print("  warmup_bias_lr: Bias learning rate during warmup")
    
    print("\n📦 LOSS FUNCTION WEIGHTS:")
    print("  box: Bounding box loss weight (1-20)")
    print("    - Higher: Better localization, important for weapons")
    print("  cls: Classification loss weight (0.1-2)")
    print("    - Higher: Better class prediction accuracy")
    print("  dfl: Distribution focal loss weight (0.5-5)")
    print("    - Higher: Better bounding box regression")
    
    print("\n🎨 DATA AUGMENTATION:")
    print("  hsv_h/s/v: Color space augmentation (0-1)")
    print("  degrees: Rotation augmentation (0-45)")
    print("  translate: Translation augmentation (0-0.5)")
    print("  scale: Scale augmentation (0-1)")
    print("  fliplr: Horizontal flip probability (0-1)")
    print("  mosaic: Mosaic augmentation probability (0-1)")
    print("  mixup: Mixup augmentation probability (0-1)")

def create_custom_hyperparameters():
    """Interactive hyperparameter customization"""
    print("\n=== CUSTOM HYPERPARAMETER SETUP ===")
    print("Enter values or press Enter for default:")
    
    custom_params = {}
    
    # Learning rate parameters
    print("\n📚 LEARNING RATE SETTINGS:")
    lr0 = input("Initial learning rate (default 0.01): ") or "0.01"
    custom_params["lr0"] = float(lr0)
    
    lrf = input("Final learning rate factor (default 0.1): ") or "0.1"
    custom_params["lrf"] = float(lrf)
    
    # Optimization parameters
    print("\n⚡ OPTIMIZATION SETTINGS:")
    momentum = input("Momentum (default 0.937): ") or "0.937"
    custom_params["momentum"] = float(momentum)
    
    weight_decay = input("Weight decay (default 0.0005): ") or "0.0005"
    custom_params["weight_decay"] = float(weight_decay)
    
    # Loss weights
    print("\n📦 LOSS WEIGHTS:")
    box_weight = input("Box loss weight (default 7.5): ") or "7.5"
    custom_params["box"] = float(box_weight)
    
    cls_weight = input("Classification loss weight (default 0.5): ") or "0.5"
    custom_params["cls"] = float(cls_weight)
    
    # Augmentation
    print("\n🎨 AUGMENTATION SETTINGS:")
    fliplr = input("Horizontal flip probability (default 0.5): ") or "0.5"
    custom_params["fliplr"] = float(fliplr)
    
    mosaic = input("Mosaic probability (default 1.0): ") or "1.0"
    custom_params["mosaic"] = float(mosaic)
    
    return custom_params

def train_with_hyperparameters(
    data_yaml_path,
    model_type="../Models/yolov8n.pt",
    preset="default",
    custom_params=None,
    epochs=100,
    imgsz=640,
    batch_size=16,
    run_name="hyperparameter_test"
):
    """Train YOLO model with specified hyperparameters"""
    
    # Get hyperparameters
    if custom_params:
        params = custom_params
        preset_name = "Custom"
    else:
        preset_info = HYPERPARAMETER_PRESETS[preset]
        params = preset_info["params"]
        preset_name = preset_info["name"]
    
    print(f"\n🚀 TRAINING WITH: {preset_name}")
    print(f"📁 Dataset: {data_yaml_path}")
    print(f"🤖 Model: {model_type}")
    print(f"⏱️ Epochs: {epochs}")
    
    # Initialize model
    model = YOLO(model_type)
    
    # Train with hyperparameters
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=0 if torch.cuda.is_available() else 'cpu',
        project="runs/hyperparameter_tuning",
        name=f"{run_name}_{preset}",
        **params  # Unpack hyperparameters
    )
    
    return results

def compare_presets(data_yaml_path, epochs=50):
    """Compare different hyperparameter presets"""
    print("\n=== HYPERPARAMETER PRESET COMPARISON ===")
    print("This will train models with different presets for comparison")
    print(f"Training for {epochs} epochs each (reduced for comparison)")
    
    results = {}
    
    for preset_key, preset_info in HYPERPARAMETER_PRESETS.items():
        print(f"\n🔄 Training with {preset_info['name']}...")
        print(f"📝 {preset_info['description']}")
        
        try:
            result = train_with_hyperparameters(
                data_yaml_path=data_yaml_path,
                preset=preset_key,
                epochs=epochs,
                run_name="comparison"
            )
            results[preset_key] = result
            print(f"✅ Completed: {preset_info['name']}")
        except Exception as e:
            print(f"❌ Failed: {preset_info['name']} - {e}")
            results[preset_key] = None
    
    return results

def main():
    print("=== ARCIS YOLO HYPERPARAMETER TUNER ===")
    
    # Check available datasets
    datasets = {
        "1": "../ARCIS_Dataset_80_10_10/data.yaml",
        "2": "../ARCIS_Dataset_70_15_15/data.yaml", 
        "3": "../ARCIS_Dataset_75_12_12/data.yaml"
    }
    
    print("\n📊 AVAILABLE DATASETS:")
    for key, path in datasets.items():
        status = "✅" if os.path.exists(path) else "❌"
        split_name = path.split('/')[-2].replace('ARCIS_Dataset_', '').replace('_', '/')
        print(f"{key}. {status} {split_name} Split")
    
    dataset_choice = input("\nSelect dataset (1-3): ") or "1"
    data_yaml = datasets.get(dataset_choice, datasets["1"])
    
    if not os.path.exists(data_yaml):
        print(f"❌ Dataset not found: {data_yaml}")
        return
    
    print("\n🎛️ HYPERPARAMETER TUNING OPTIONS:")
    print("1. Use preset hyperparameters")
    print("2. Create custom hyperparameters")
    print("3. Compare all presets")
    print("4. View parameter information")
    
    choice = input("Enter your choice (1-4): ")
    
    if choice == "1":
        # Use preset
        print("\n📋 AVAILABLE PRESETS:")
        for i, (key, preset) in enumerate(HYPERPARAMETER_PRESETS.items(), 1):
            print(f"{i}. {preset['name']}")
            print(f"   {preset['description']}")
        
        preset_choice = input(f"\nSelect preset (1-{len(HYPERPARAMETER_PRESETS)}): ") or "1"
        preset_keys = list(HYPERPARAMETER_PRESETS.keys())
        preset_key = preset_keys[int(preset_choice) - 1] if preset_choice.isdigit() else "default"
        
        epochs = int(input("Number of epochs (default 100): ") or "100")
        run_name = input("Run name (default 'hyperparameter_test'): ") or "hyperparameter_test"
        
        train_with_hyperparameters(
            data_yaml_path=data_yaml,
            preset=preset_key,
            epochs=epochs,
            run_name=run_name
        )
    
    elif choice == "2":
        # Custom hyperparameters
        custom_params = create_custom_hyperparameters()
        epochs = int(input("\nNumber of epochs (default 100): ") or "100")
        run_name = input("Run name (default 'custom_hyperparameters'): ") or "custom_hyperparameters"
        
        train_with_hyperparameters(
            data_yaml_path=data_yaml,
            custom_params=custom_params,
            epochs=epochs,
            run_name=run_name
        )
    
    elif choice == "3":
        # Compare presets
        epochs = int(input("Epochs for comparison (default 50): ") or "50")
        compare_presets(data_yaml, epochs)
    
    elif choice == "4":
        # View parameter info
        display_parameter_info()
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main() 